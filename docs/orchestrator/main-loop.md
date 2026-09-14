# 主循环与派单 — 端到端数据流详解

> 目标:单步理解「一条 prompt 从进到出要穿过哪些函数」。配合 `docs/architecture/overview.md` 一起读。

## 1. `run()` 的三座并发火山

orchestrator.py:613 `run()` 是唯一的主线程骨架:

```python
trainer = TrainingBackend.init(...)        # 训练器就绪(FSDP or Megatron)
server_running = InferenceServerActor.start(...)
process_list = ... await self.start_processes()   # instance + eval servers

if not trainer.paused:
    trainer.start_training()               # 训练协程 A
orchestrator_task = asyncio.ensure_future(
    self.start_orchestrator_rollout()      # rollout 协程 B
)
weight_sync_task = asyncio.ensure_future(
    self._weight_update_watcher()          # 权重 watcher 协程 C
)
await asyncio.gather(*trainer.awaits, orchestrator_task, weight_sync_task)
```

关键认知:

| 协程 | 做了什么 | 永远在做的 |
|------|---------|-----------|
| 训练(A) | `fsdp.train_step` 微批循环 | 只喂权重给 watcher |
| rollout(B) | `_rollout_loop` 怼 prompt 进 vLLM | 只消费权重 + 采样 |
| watcher(C) | 等 `weight_sync_ready` → `broadcast` → `_update_inference_weights` | 唯一移动 `inference_model_step` 的地方 |

三者以 **`weight_sync_ready` 事件** 为唯一同步点。模型版本编号 = `inference_model_step`,由 watcher 单调递增。

## 2. 派单:从「有prompt」到「vLLM的lane」

`_start_next_prompt`:996(组模式)是调度心脏,完整路径:

```
_rollout_loop
  └─ _start_next_prompt()
       ├─ _checkpoint_pending?          → 停(PRE 占位)
       ├─ _has_pending_prompt_data()?   → 没数据就停
       ├─ async_level = _compute_async_level()      # 异步水位
       │    > max_async_rollout → 暂停并打 rollout_paused_max_async 事件
       ├─ _pick_server_with_capacity()              # 轮询,绕开 eval server
       ├─ 分配 group_size 个 lane_slots(组播)
       ├─ 预分配 sample_ids(next_sample_idx++)
       └─ asyncio.create_task(_run_prompt(...))     # 每条跑一个任务
```

派单快照记录在 `inflight_rollout_info[request_id]`(**事件溯源式的"在飞发货单"**):server、prompt、每个样本的 off-policy 步数、每个样本的 timing。它不只是一个记录——off-policy 取消、timeout 统计全靠翻这张表。

### 个体样本模式

`individual_sample_lanes`(default_train.yaml)开启后走 `_start_next_individual`:1064:

1. 优先 `_dispatch_one_queued_sample`:1141——从 per-server 队列弹一件,有 lane 立刻派;
2. 否则新建一组(`pending_individual_groups[request_id]`,`remaining=group_size`),把 `group_size` 件全塞进 deque(==「为这组占位」);
3. 立即派一件返车(撞 lane),省一个调度周期。

什么时候用:**环境要独立跟踪每个样本**(如 tool-use 环境,每个样本一问一答),且单个样本的完成时间参差。整数组的「一次性打完」在此时是浪费——逐个 lane 释放,逐个派。

## 3. 回合环:组内如何走完一轮

`safe_rollout_loop` 每轮:
- 从 `prompt_data` 构造 `sample_timings` 与 lane_slots;
- 对不可并行项(工具调用 / 多轮)用 `non_parallel[0].set()` 串行化;
- 组装 `group_samples` → 调 `generate_completion`(单轮)或 `run_multiturn_rollout`:698(多轮);
- 汇编 flip 结果:`_assemble_reach` 在 `_assemble_single_turn_group`:1651(单轮)/ `_assemble_multiturn_group`:1752(多轮)里做优势折算与 `turn_metrics` 统计;
- 打 `step_metrics` 事件出帧。

**一个翻转点(容易误读)**:`sample_id` 在文件里出现多次,含义不同——
- `next_sample_idx`(orchestrator.py:1047)**run 级全局唯一**;
- `agent_id` 与 `sample_id` 是两组 ID,不要混。

## 4. 权重追权重:watch → stage → broadcast → commit

watcher `_weight_update_watcher`:776 的循环:

```
等待 trainer 事件 step 完成
  └─ _update_inference_weights:2479
       ├─ 收集 weight_sync_load_refs(远端对象的引用)
       ├─ 广播(GATHER_TARGET 用 CPU staging 单副本)
       ├─ 解析 timing → event_logger 记 weight_sync start/end InfraEvent
       ├─ 推进 inference_model_step = step(weight_sync_group=="full" 时 eval 也推进)
       ├─ off-policy 扫描:超 max_off_policy_steps 的组 → _log_cancelled_rollouts + cancel 任务
       ├─ 释放 step_batches / 打 rollout_reward_metrics
       └─ trainer 收到 ACK → 解 finalize
```

## 5. 故障与重试哲学

- 组内样本超时:**不重试同一组**,取消后等 `_flush_pending_batches`:2285 丢弃输出;
- vLLM 无响应:`_pick_server_with_capacity` 里的 server cycle 会绕开它;
- 被取消的样本进 **DiscardedPrompt / DiscardedGenerationRollout**(event_logger.py:778 / 794),而非静默丢失:
- **不写磁盘的文件对象**:中间产物全在 W&B events 帧里,`tail_idx` 升帧即归档。

## 6. 读代码时按此顺序

1. `main()` :3279 → `run()` :613(骨架)
2. `_start_next_prompt` :996 → `_run_prompt`(派单)
3. `_update_inference_weights` :2479(watcher 缴费)
4. `_assemble_single_turn_group` :1651(rire 出 batch)
5. `_build_byte_chunk` / ray 下发 batch(训练侧喂饭)