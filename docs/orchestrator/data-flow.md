# 异步训练数据流

> 训练全链路由编排器驱动：Prompt 预取 → 调度推理 → 生成/奖励 → 优势计算 → 组批 → 训练步 → 权重广播。关键在异步：推理与训练并发、批量与样本车道分层、off-policy 过期自动取消。

## 1. 全链路图

```
预取 prompt（_prepare_prompt_data :2394）
   ├─ 单轮：提前 get_initial_prompt + apply_chat_template → messages/prompt_str/token_count
   └─ _get_next_prompt_data :2373 从预取缓冲（prompt_prefetch_buffer_size）取，不阻塞采样
        │
        ▼
调度 dispatch（_schedule_dispatch :972）
   ├─ 标准 group：_start_next_prompt :996（_pick_server_with_capacity :949，跳过 eval 服务器）
   └─ 独立样本车道：_start_next_individual :1064（_pick_server_with_any_capacity :1133，free_lane_after_generation）
        │
        ▼
生成（generate.py）
   ├─ 单轮：generate_completion_with_tokens :439 / generate_completion :547
   │    请求体：return_token_ids=True, logprobs=1, skip_special_tokens=False（token_ids 与文本对齐）
   │    max_tokens = max_model_len - prompt_len（动态收缩）
   │    错误归一：PromptTooLongError / ContextExhaustedError / RolloutError（:395-542）
   └─ 多轮：run_multiturn_rollout :698（INTERLEAVED_ROLLOUTS 本地 tokenize，build_next_turn_prompt_ids :806）
        │
        ▼
奖励（env_worker_pool 子进程）→ RewardResult{sample_metrics, golden_answers, info_turns, sample_tags}
        │
        ▼
组批 + 优势（_assemble_single_turn_group :1651 / _assemble_multiturn_group :1752）
   compute_advantages 于 generate.py:1784-1819 按算法分发
        │
        ▼
过滤：零优势组丢弃（_should_discard_zero_advantage_group :1976）→ off-policy 过期取消
        │
        ▼
训练步（trainer）→ gather_weights_for_inference → weight_sync（NCCL 广播）
        │
        ▼
编排器 ray.get(weight_sync_load_refs) → InfraEvent(weight_sync) → 推进 inference_model_step（:2543-2553）
```

## 2. 异步与并发模型

| 机制 | 配置 | 实现 |
|------|------|------|
| **推理/训练重叠** | `max_async_rollout` | 训练不等待全部组：滚动批次投喂 |
| **Prompt 预取** | `enable_prompt_prefetch` / `prompt_prefetch_buffer_size` | `_prepare_prompt_data` 提前准备模板后的 prompt，采样不触碰 tokenizer 延迟 |
| **个体样本车道** | `enable_individual_sample_lanes` / `free_lane_after_generation` | 单个样本独立占用服务器槽位（`individual_sample_queues` 按 server 的 deque，`_dispatch_one_queued_sample` :1141）；适合超长 rollout 不被队头阻塞 |
| **off-policy 生命周期** | `max_off_policy_steps` | 每次权重更新后计数未完成 HTTP 样本（`sample_off_policy_steps`），超限取消请求与任务（:2555-2590） |
| **生成并发** | 后台线程池 2048 | `_sync_http_post` :42 用 stdlib `http.client` 绕开 async httpx 并发瓶颈（generate.py 头注释 :27-38），执行为 `loop.run_in_executor` |

## 3. 生命周期与取消路径

- 样本生命周期由 `_make_sample_lifecycle`(:1183) 构建；取消路径 `_handle_cancelled_sample`(:1453)、`_on_cancelled_group`(:1558)。
- 丢弃记录：`_log_kept_rollout_group`(:2117)、`_log_discarded_rollouts`(:2002)、`_log_cancelled_rollouts`(:2232) 分别生成 kept / discarded / cancelled 事件族（UI 与 eval 均按此分表）。

## 4. 优势计算（generate.py:1784-1819）

| 算法 | 优势公式 |
|------|---------|
| grpo / cispo / gspo / sapo（group 归一） | `(r - mean) / (std + 1e-4)`，ddof=1 |
| rloo | 留一基线 × n/(n-1) |
| dr_grpo | 仅减均值，不标准化 |
| reinforce_pp / `advantage_norm=="batch"` | 两段式：组减均值 → batch 白化 |

- 零优势组判定（`_should_discard_zero_advantage_group` :1976）：`discard_group_zero_advantage` 且组内全部 advantage 差值 < 1e-9——**不依赖标准化方式，对任意算法生效**。

## 5. 权重更新协调

- 初始化：`_init_weight_broadcast`(:2629) `world_size = 1 + num_inference × tp_size`；单节点共址时优先 loopback `127.0.0.1`（`ray_broadcast_prefer_loopback_if_single_node`）。
- 执行：训练完成 → `ray.get(weight_sync_load_refs)`（`asyncio.to_thread` 避免阻塞事件循环，:2522-2523）→ 起止时间写为 `weight_sync` InfraEvent（:2527-2541）。
- 状态推进：广播成功 → `inference_model_step = step`；group == `"full"` 时同步推进 `eval_server_model_step`（:2543-2553），维持 eval 服务器模型版本。

## 6. 批次重试与持久化

- `_try_start_pending_rollouts`(:2276)、`_flush_pending_batches`(:2285)、`_save_batch`(:2307)：批次投喂 trainer 失败时缓存，下一轮重试——不丢 batch。
- 每个训练批包含 micro_batches：`vllm_logprobs`、`advantages`、`position_ids`（按 `input_ids` 前缀对齐）。