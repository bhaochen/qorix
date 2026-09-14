# 事件体系 — 统一事件总线(EVENTS)

> "每一条发生过的事,都要能以最小单位去查。" 这是 `EventLogger` 的承诺。本篇把全部事件类型、字段、身份语义一次性讲清,并给出 timeline 查询指南。

## 1. 四层事件金字塔

| 层 | 数据类 | 负责谁 | 物化 |
|----|--------|--------|------|
| 瞬时事件 | `Event:656` | trainer + orchestrator 的「开关」 | `events/` zip 内 events 表 |
| 样本级 | `GenerationRollout:742` | 一个样本的全生命周期 | `events_rollouts` 表 |
| 生命周期 | `RolloutEvent:821`(start/end)+ `InfraEvent:838` | 粒度 timeline(UI) | 每 2 帧刷新 |
| 帧度量 | `GpuMetricSample` / `CpuMetricSample` / `VllmMetricSample` | 系统遥测 | blocks 聚合 |

### Event:656 字段表

| 字段 | 语义 |
|------|------|
| `event_type` | forward / backward / weight_sync / rollout_paused_max_async / inference_call / … |
| `source` | `"trainer"` 或 `"orchestrator"` |
| `step` | 训练步(-1 表示不适用) |
| `rank` / `local_rank` / `node_id` / `gpu_index` / `node_ip` / `hostname` | 硬件归属(join gpu 表靠 `node_id+gpu_index`) |
| `start_time` / `end_time` | 持续事件的区间 |
| `parent` / `depth` / `microbatch` / `minibatch` | 层级与粒度(forward 的子事件) |
| `group_id` / `sample_id` | 关联的流程身份 |
| `tail_idx` | 第几个上传帧(see §3) |

## 2. 样本身份的统一语义

整条链路只靠两个 ID 站稳:`group_id`(组)与 `sample_id`(样本)。

- **生成**:`GenerationRecord:681` —— 一次 vLLM 往返(`content/tokens/prompt_tokens/tool_call_count/stop_reason` + 六项时延)。
- **环境回喂**:`EnvResponseRecord:700` —— `turn_type`(tool_result / env_response / feedback / context)区分「谁插的话」。
- **工具**:`ToolCallRecord:711` —— `success/exit_code/truncated/sandbox_id` 全记录,**generation.sample 的回环用 `generation_idx` 指认**。
- **prompt**:`Prompt:729`(发布)+ `DiscardedPrompt:778`(被丢弃)——丢弃有 `discard_reason`,不静默。

聚合:一条 `GenerationRollout` = `step + group + sample + agent + generations + env_responses + tool_calls + reward/advantage + total_tokens + stop_reason + off_policy_steps + tail_idx`。所有分析(advantage、off-policy、turn_metrics)都从这个结构读取。

## 3. tail 帧与 block:events 为什么是「卷筒纸」

```
tail_idx  0   1   2   3 ...   (每次上传周期 +1)
         ──帧──
         [0]→上传 delta=pending
         [1]  [1]   ← 同一数据可横跨多帧(block 未结尾)
         ──block──
         block 按时间(最大 BLOCK_DURATION)或步数(rollout_block_size)封口:
         封口 = 写盘 → 下一个 block 全帧都带新的 block_idx
```

`EventLogger.__init__`(event_logger.py:943)里:

- `TAIL_WINDOW_SECONDS / BLOCK_DURATION_SECONDS / ROLLOUT_BLOCK_SIZE / UPLOAD_INTERVAL_SECONDS` 四种节奏被折叠成 `tail_idx` 与 `block_idx` 两个递增篱笆(:973, :976-979);
- `_inflight_*`(磁盘 memory? 译为"在飞件"):`inflight_generations / tool_executions / env_responses / rewards / weight_syncs / sandbox_ops` 每帧 snapshot 进 `tail.zip` —— **权重更新期间「正在生成」的事件不会丢,而是所在帧携带当前 snapshot 继续累计**;
- 为什么能串:每条记录带 `tail_idx`,UI 按 `tail_idx` 拉窗,orc 端到端「看到的是同一个事件在不同帧中的持续状态」。

## 4. 训练 vs 推理的墙(wall)

`Event.source` 把立体事件分侧:
- **trainer 侧**:前/反向、microbatch、rank 级事件的 parent/depth 层级;
- **orchestrator 侧**:inference_call、weight_update、rollout_paused_max_async、cancelled(off-policy)。

`StepMetric:768`(grad_norm、entropy、kl)与 rollout metrics 各自流向 `_pending_step_metrics` / `_pending_rollouts` 表,二帧合一的 `tail_idx` 让它们能按 step 对齐。

## 5. 查询命令(本机)

```bash
ls experiments/<run>/events/                # zip 帧
unzip -l <run>/events/tail_010.zip          # 看帧内容
```

UI 上:Timeline 选 `event_type = weight_sync` 即见权重推进;`RolloutEvent event_type=generation phase=start/end` 即见样本的推理热区。