# 面试准备 — qorix 深度要点

> 这不是入门文档,是「你能讲出什么故事」的能力清单。每条都是一道能展开聊半小时的题,配源码锚点。目标:读完能画出架构图、说清三个并发协程、解释 off-policy 的代价模型。

---

## 第一梯队:必须能秒答

### Q1. 训练、rollout、权重同步如何并行不打起来?

- 三类协程:`TrainingBackend.start_training()`、`start_orchestrator_rollout()`、`_weight_update_watcher()`——三者被 `asyncio.gather` 拉在一根事件循环上(orchestrator.py:613 run())。
- 同步点是 `weight_sync_ready` 事件。**trainer 永远不等 rollout**,它只管把结果推给 watcher;watcher 是唯一推进 `inference_model_step` 的人(:2479)。
- 代价模型见 Q3。

### Q2. 组的语义是什么?个体模式为何也按组?

- 组 = `group_size` 个 prompt 打包一次 HTTP 请求,shared 权重采样(同一步)。槽位 `lane_slots` 预分配(orchestrator.py:1029-1058)。
- 个体模式 `_start_next_individual`:1064 依然按 `group_size` 开组,只是**逐槽派发**(`_dispatch_one_queued_sample`:1141),这样"掉队样本"不占死整组,lane 逐个释放。

### Q3. max_async_rollout 与 max_off_policy_steps 分别守什么线?

- `max_async_rollout`:**在飞组数上限**。超了 `waiting_for_trainer=True` + 打 `rollout_paused_max_async` 事件(:1008-1014)。它守"同时有多少 rollout 在路上"。
- `max_off_policy_steps`:**单个样本的滞后步数上限**。watcher 每推进一步就给未完成的样本 +1(:2555-2566),超过就取消组(:2567)。它守"策略多旧就不能再练"。
- 一句话:**异步总量 vs 个体老化**,两个数量级不同。

### Q4. 权重怎么从 trainer 到每个 vLLM server?

1. trainer(rank0)**CPU staging 单副本**——`weight_sync.py:38`,`_build_flattened_bucket_chunks`:70 分桶;
2. 广播:`broadcast_weights_to_inference`:348 走 ray actor,event_logger 记 weight_sync(start/end)InfraEvent;
3. worker 端 `receive_state_dict`:108 收 → server `/init_broadcast`:75;
4. `group` 模型思考:`weight_sync_group="full"` → 连 eval server 也推进;`training_only` → eval 冻结(:2545-2547)。

### Q5. 有翻车样本怎么办?为什么丢数据的时候不 panic?

- 组内超时/失败样本 → **槽位留空**,`None` 防御(`_assemble_single_turn_group`:1651);丢弃进 `DiscardedGenerationRollout`(event_logger.py:794)留 `discard_reason`;
- rollout 失败不重试同一组,直接 `_flush_pending_batches`:2285;
- vLLM 断了 → `_pick_server_with_capacity` 的轮询绕开它;
- 核心哲学:off-policy 本身就允许数据不完美,**丢弃有因、样本可查**。这就是「可观测的失败」——比静默降级强。

---

## 第二梯队:面试官笑了就赢

### Q6. 为什么 HTTP 走 2048 线程池,而不是异步?

- 同步 `_sync_http_post`(generate.py:42)+ `loop.run_in_executor(_http_pool, ...)`(generate.py:511-515);
- 理由在文件头注释:2000+ 并发下 async httpx 的 JSON 解析+协议开销 ~3s/请求,线程池是经过实测的省事路径。
- 推论:框架**只有一个事件循环**,没有第二个;任何「改 async 就更快」的直觉在 2000 个并发请求前都是错的。

### Q7. mask 与 masked loss 是怎么回事?

- 训练损失是 **per-sample 的 masked mean**(loss.py:21),mask 各路来源按 step 对齐;
- `sampling` 里带 logprobs;序列版 IS / dapo 掩码让 off-policy 梯度不炸;
- 环境回喂(token 结果)与 n_turn 期间同一步内能不能 mask 到,取决于 tokenizer 一致性(见 Q10)。

### Q8. 多轮怎么保证 token 前缀一致?

- `InterleavedTokenizer`(generate.py:126)缓存对话历史 tokens,增量拼接而非重 tokenize;
- vLLM 返回 `skip_special_tokens:false` + `include_stop_str_in_output:false` → text 与 token_ids 逐位可逆(generate.py:495-496);
- 这样「回合间 prefix」在训练侧 mask/loss 不会偏移。

### Q9. 权重版本与 eval 版本如何分叉?

- `weight_sync_group="full"` 时 eval 与 train 同推进;`training_only` 时 **eval_server_model_step 冻结**(orchestrator.py:2545);
- 好处:超大 eval server 不用每步刷新,如大模型 eval 用 quantized server,只有 checkpoint 才同步;
- 代价:eval 时如果 switch 了 quantization,是独立的 server 生命周期 —— eval 结果必须标注 `model_step`。

### Q10. checkpoint 转 HF 的三个坑是什么?

- FSDP 的 `model.` 前缀要 strip(`_convert_fsdp`:62);
- Megatron TP>1 的 gate_up 交错 `[g0;u0;g1;u1...]` → `_deinterleave_tp_gated_mlp`:156 重排回 `[gate;up]`;
- TP size 探测依赖 `meta.tp_size` 或数 word_embeddings 的 chunk(`_detect_tp_size`:122)。
- **结构性 lesson**:训练侧(save_training_state=RNG+optimizer 可选)与部署侧(HF safetensors)是两套格式,当中必须架转换桥;转换是「按需 + 缓存](driver.py:140)。

### Q11. 为什么 eval 要造一个 compat config?

- 既有 EvalRunner/generate.py/server 全读全局 `config.cfg`;`_install_compat_config`(driver.py:175)拼一个兼容副本,**而不是**给 eval 重写一套推理路径。
- 这就是项目的「先兼容不再造」哲学。推论:standalone eval 与训练内 eval 共用 server 代码,bug 一修两好。

---

## 第三梯队:能让面试官点头的进阶

### Q12. 事件的 tail 帧与 block 如何做到「大数据不丢」?

- `tail_idx` 每上传周期 +1,`block` 按时间或 `rollout_block_size` 封口(event_logger.py:973-997);
- `_inflight_*` 快照:权重更新期间「正在生成」的事件不会丢,tail.zip 携带当前在飞 snapshot 继续累计(:952-958);
- 反观「一条事件一行」的做法在 2000 并发下必崩——帧上报让 wandb 与本地都只存「变化的尾部」。

### Q13. 你如何定位「样本卡住」?

1. 看 `_inflight_generations` 的时长(在飞太久 → HTTP 真断了);
2. 看 `RolloutEvent phase=start/end` 在 timeline 上的 gap;
3. 看 `sample_off_policy_steps` 计数——若是 `max_off_policy_steps` 卡在临界线,是策略老化不是网络问题;
4. `DiscardedGenerationRollout.discard_reason` 区分「被扬 vs 自然结束」。

### Q14. 为什么 `_pick_server_with_capacity` 要绕过 eval server?

- 计算服务器有自己的 lane 预算;eval server 只在 `weight_sync_group="full"` 时收到权重,平时不应挡主路径;
- 归队:独立 server 生命周期 — 启动 `eval_server.start` → `wait_for_avail`,`eval server` 每 step 做若干 batch,然后休整。

### Q15. 你打算怎么加「双语切换」这个 feature?(开放题)

- 环境层:`env_response` 注入语言切换指令(改 prompt 不改链路);
- 权重层:多个 server 不同 quantization,`weight_sync_group` 各自推进;
- 评测层:eval 每个 server 有独立 `model_step`,在 eval 表里打 `eval_name + model_step` 双列;
- 落点:**一切回到 Sample(group_id/sample_id)与事件,不需要改主循环** —— 这是「单样本优先 + 组槽位 + 可观测失败」设计给的红利。