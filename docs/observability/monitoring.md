# 监控与可视化(观测金字塔)

> 这一篇把这些「日志到底在干嘛」一次性对齐,并给出**三层诊断路线**:先看输量,再看权重推进,最后落到样本。

## 1. 四个监控出口,各管一段

| 出口 | 谁写 | 管什么 | 观测面 |
|------|------|--------|--------|
| W&B(主) | orchestrator/wandb_logger.py | 训练曲线 + 事件帧 | run dashboard |
| events zip | EventLogger | 全量事件时序(独立于 wandb) | timeline 页面 / 离线分析 |
| tlog | tlog/logger.py:457 | 终端结构化日志 | 控制台/CI 消化 |
| GPU/vLLM/CPU 指标 | system_metrics_logger / vllm_metrics_logger | 硬件遥测 | blocks 表 / 曲线 |

**双链路原则**:训练(曲线)与推理(timeline)在 wandb 上不同命名空间,但共享同一 `tail_idx` 帧;离线上一份 zip 即可重建两轨。

## 2. W&B 三层命名(canon 结构)

`wandb_logger.log_trainer_metrics_payload`(:767)构造步骤键——以 step 为主题(训练模型的推进步),用 `section` 区隔:

```
step_0042/
  grad_norm          ← 训练核心
  entropy / kl_...
  loss                ← 掩码均值(masked mean,loss.py:21)
---
rollout/step_0042/
  avg_reward / sample_metrics / advantage
  rollout generation 时长(ttft/e2e_latency)
---
infra/
  weight_sync start/end × server_id
  sandbox 阶段
```

- 训练侧指标 key 以 `step` 为准;**rollout 侧同一 step 也挂 0042** —— 便于「训练走到 42 步时模型在干嘛」;
- `gpu.parquet` 用 `node_id+gpu_index` 与 `Event` 关联 → 把每个 forward/replication 钉到物理卡。

## 3. 三类「暂停/减载」信号,一图看懂

| 你需要观察 | event_type | 出现含义 |
|------------|-----------|---------|
| rollout 压不上去 | `rollout_paused_max_async` | `async_level > max_async_rollout`:rollout 在等 trainer(`waiting_for_trainer=True`) |
| 权重迟迟不动 | `weight_sync` 稀疏/缺帧 | watcher 睡着:trainer 步不推进,broadcast 卡 CPU staging |
| 样本被扬了 | `DiscardedGenerationRollout.discard_reason` | off_policy / max_async / zero_advantage / checkpoint_pending |

UI 操作:`filter` 里选 `rollout_paused_max_async`,出现次数与时长 = 异步余量的直观显示。

## 4. 铁手诊断三问(先总量后单体)

1. **吞吐对不对**:`generations/step` 与 `event_upload` 帧的间隔——若帧间隔 > taill window 常驻,说明系统 metrics 积压;
2. **权重推进是否线性**:`weight_update` 事件的 step 间距——若某 step 拉的 gap 特别大,八成是 checkpoint_save 或 sandbox 打头;
3. **老样本死法**:`off_policy_steps` 计数曲线 / `_inflight_generations` 里残留时长 — 若残留上百,`max_off_policy_steps` 该调大或开 `discard_group_zero_advantage`(default_train.yaml:43)丢弃无信号组。

## 5. vLLM 侧打点(server 侧)

- `/torch_memory`(server.py:95)遥测进 `VllmMetricSample`,看显存制高点;
- `max_concurrent_prompts_per_server` 直接由 lane 分配体现,和 `available_server_lane_slots` 对齐可判断「容量 vs 堆积」;
- GPU 遥测 `GpuMetricSample` 以 blocks 聚合,step 帧随 tail 上载——**显存曲线与训练 step 对齐**。

## 6. 必要的"个性化"设置

- `event_tail_window_seconds / event_upload_interval_seconds` —— 帧的多细;
- `rollout_block_size` —— 按样本数封块的频率(默认 4096);
- 想关某类日志:`event_logger.py` 里的 `_inflight_*` 开关(可留痕)。