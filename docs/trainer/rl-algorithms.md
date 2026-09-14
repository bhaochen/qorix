# RL 算法与损失

> 单个损失实现支撑 7 种算法：**GRPO / RLOO / REINFORCE++ / DR-GRPO / CISPO / GSPO / SAPO**，并可叠加 PPO Clip、双 Clip、截断重要性采样（TIS/ICE-POP）、KL (k2/k3)、熵奖励、序列级掩码等修饰（`trainer/loss.py` 头部注释 :5-12）。`algorithm` + 优势聚合在 `generate.py:1784-1819` 完成。

## 1. 算法矩阵

| 算法 | 配置值 | 优势/损失 | 备注 |
|------|--------|-----------|------|
| GRPO | `grpo` | 组内 (r-mean)/(std+1e-4) 归一化 | 默认 |
| RLOO | `rloo` | 留一基线 × n/(n-1) | 留一法基线 |
| REINFORCE++ | `reinforce_pp` | 两段式（组均值→batch 白化） | 需 `advantage_norm=="batch"` |
| DR-GRPO | `dr_grpo` | 仅减组均值，不标准化 | 无方差项 |
| CISPO | `cispo` | group 归一化 + clip | 与 `use_ppo_clip` 配合 |
| GSPO | `gspo` | group 归一化 + clip | GFlowNet 风格 |
| SAPO | `sapo` | group 归一化 + `sapo_tau_pos/neg` | 正负不等 tau |

- 优势汇聚函数位于 `generate.py`（单/多轮组批时按算法分发）。

## 2. 损失核心机制（`trainer/loss.py`）

- `_masked_mean_per_sample`(:21)：以 `position_ids==0` 为样本边界，把每样本 token 替换为样本掩码均值——**对齐 vLLM logprobs 的连续序列表示**（batch 内按 pos 切分）。
- `_masked_count_per_sample`(:47)：样本有效 token 计数。
- 序列打包下逐 token 加权累加 entropy/KL/额外指标：`_CORE_KEYS = {"loss","entropy","kl_divergence_inference","num_tokens"}`（fsdp.py:350），额外键自动收集（:376-382）。

## 3. 修饰项（开关与参数）

| 修饰 | 配置 | 说明 |
|------|------|------|
| **PPO Clip** | `use_ppo_clip=true`、`ppo_clip_ref_logprobs`(rollout/batch)、`clip_low`、`clip_high` | 对某一侧 ref 做 clip；`"batch"` 时训练前先 `_compute_batch_logprobs`（fsdp.py:238，no-grad 存 `[batch, seq_len]`） |
| **双 Clip** | `clip_ratio_c` | 上下限不对称 clip |
| **截断重要性采样** | `use_tis`、`tis_cap`、`tis_logprob_clamp`、`tis_mode`(truncate/icepop)、`tis_floor` | off-policy 截断；`max_off_policy_steps` 配合过期取消 |
| **KL 惩罚** | `kl_penalty_tau`、`kl_estimator`(k2/k3) | 参考策略距离惩罚 |
| **熵奖励** | `entropy_coef`、`entropy_chunk_size` | 探索项 |
| **序列级掩码** | `seq_is_masking`、`seq_is_mask_low/high` | IS 重要性掩码 |

## 4. 训练步内的计算顺序（fsdp.py `train_step` :294）

1. （可选）batch 级 ref logprobs：`_compute_batch_logprobs`(:238)。
2. 按 `number_of_minibatches` 轮转分组（:334-340）。
3. 每 minibatch：一次 `zero_grad(set_to_none=True)` → 逐微批 fwd/bwd。
4. **梯度累积优化**：仅组内最后一个微批 `set_requires_gradient_sync(True)`（:363-369）——FWD/BWD 仍逐微批，reduce-scatter 只在最后一次，省 NCCL 通信。
5. grad_norm → `grad_clip` 截断（未设则手算 L2，:384-397）。
6. `optimizer.step()`；grad_norm 非有限则跳过并清梯度（:399-408）。
7. LR scheduler 每 **step** 而非每 minibatch 步进（:415-417）。
8. 输出 token 加权平均的 entropy/KL 与平均 grad_norm（:419-423）。

## 5. 测量与诊断

- 稳定性过滤在**批次预处理**阶段：`filter_gibberish`（gibberish_token_threshold + logprob offset）、`filter_repetition`（repetition_compression_threshold）、`filter_overlong`（overlong_penalty_factor + buffer）。
- 零优势组丢弃独立于损失：`discard_group_zero_advantage`（advantage 差值 < 1e-9）。

## 6. 测试与验证建议

- 对同一 batch（固定 input_ids/advantages/vllm_logprobs）跑各 `algorithm`，断言损失与手工参考一致。
- 验证 `_masked_mean_per_sample` 对打包序列（多样本拼接、position_ids 归零）的分界正确性。