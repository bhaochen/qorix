# 事件日志与可观测性

> 编排器把训练全链路留给自己的证据写成一个**字符串流事件日志 + 结构化表**，再经 W&B 上传：事件 → `tail_idx` 分窗 Arrow 表 → zstd 压缩 → W&B artifact。指标在 W&B 按 **section/group/metric** 三层命名；qorix-ui 正是消费这套格式。

## 1. 事件记录器（`orchestrator/loggers/event_logger.py`，4011 行）

**核心数据类**：

| 类 | 行号 | 字段要点 |
|----|------|---------|
| `Event` | :656 | timestamp, event_type, source(trainer/orchestrator), step, rank, local_rank, node_id, gpu_index, node_ip, hostname, ray_node_id, start_time, end_time, parent, depth, microbatch, minibatch, tail_idx, group_id, sample_id |
| `GenerationRecord` | :681 | generation_idx, content, tokens, prompt_tokens, tool_call_count, stop_reason, queue_time, ttft, prefill_time, decode_time, inference_time, e2e_latency, server_id, vllm_request_id |
| `EnvResponseRecord` | :701 | 环境响应 |
| `ToolCallRecord` | :711 | 工具调用 |
| `Prompt` | :729 | prompt 快照 |
| `GenerationRollout` | :742 | rollout 汇总 |
| `StepMetric` | :768 | **`{step, metric, value, section, group}` —— W&B 三层命名来源** |
| `DiscardedPrompt` / `DiscardedGenerationRollout` | :794/:822 | 丢弃样本 |
| `RolloutEvent` | :838 | rollout 事件 |
| `InfraEvent` | :850 | 基础设施事件（weight_sync、服务器起停） |
| `EvalPrompt` / `EvalGenerationRollout` | :866/:888 | 评测样本 |

**写入 API**（`EventLogger` :888）：
- `log_event`(:1074) / `log_instant_event`(:1121)
- `log_generation_rollout`(:1159，签名 :1161-1185：step/group_id/sample_id/prompt/generations/env_responses/tool_calls/reward/advantage/env/sample_metrics/golden_answers/info_turns/sample_tags/turn_metrics/tokens_prompt/system_prompt/total_tokens/raw_string/compute_reward_time/stop_reason/off_policy_steps/agent_id)
- `log_rollout_event`(:1311)、`log_infra_event`(:1394)、`log_step_metric(step, metric, value, section, group)`(:1439)
- `set_trainer_steps_done`(:1448)、`log_discarded_generation_rollout`(:1453)、`log_cancelled_generation_rollout`(:1578)
- `log_eval_prompt`(:1627)、`log_eval_rollout`(:1635)、`add_gpu_metrics`(:1640)、`add_cpu_metrics`(:1655)

**Arrow 表转换与压缩**：
- `_orchestrator_events_to_table`(:1664)、`_trainer_events_to_table`(:1688)、`_rollout_events_to_table`(:1724)、`_infra_events_to_table`(:1743)、`_gpu_metrics_to_table`(:1758)、`_cpu_metrics_to_table`(:1786)、`_logs_to_table`(:1808)、`_vllm_metrics_to_table`(:1829，列：timestamp/server/node_id/tp_group_id/tp_size/metric_name/value/tail_idx)、`_thread_pool_metrics_to_table`(:1855)、`_prompts_to_table`(:1875)、`_generations_to_table`(:1899，**content 用 `_zstd_compress` 压缩** :1917)、`_env_responses_to_table`(:1941)。
- `_zstd_compress`(:92)。

**上传窗口策略**（常量 :924-939）：
- `TAIL_WINDOW_SECONDS`、`BLOCK_DURATION_SECONDS`、`ROLLOUT_BLOCK_SIZE`、`UPLOAD_INTERVAL_SECONDS`。
- partition 表按 `(partition_type, tail_idx)` 落盘+上传；`tail_idx` 标记事件首次进入哪个 tail 窗口；超窗时间后老事件不再上传——**幂等增量**（qorix-ui 侧以 `ingested_tails` 去重）。

## 2. W&B 桥（`orchestrator/loggers/wandb_logger.py`）

- **节点分组**：setup 时注册 `inference` section(:344-347) 与 `trainer` section(:372-374)。
- `log_trainer_metrics_payload`(:767)：把 trainer 每步在内存聚合的 payload（step_metrics/timeline_events/torch_memory）按序写入事件日志；**并同时写 wandb 原生图表**——key 由 `section/group/metric` 三节拼接（`"/".join(parts)`，:788-794）。
- `log_rollout_reward_metrics`(:835)：写 `rollout/reward_mean`（:850）。
- `_resolve_trainer_location`(:700-761)：rank/local_rank → node/gpu（解析 `CUDA_VISIBLE_DEVICES` 索引映射物理 GPU，:744-753）。
- `log_generation_rollout` 便捷封装、`set_trainer_steps_done`(:763)。
- 上传循环：`start_event_upload_loop`（orchestrator.py:637 启动）。

## 3. 系统监控器（`orchestrator/loggers/`）

| 文件 | 职责 |
|------|------|
| `system_info.py`(910) | 主机 CPU/GPU/IO/温度采样 |
| `system_metrics_logger.py`(606) | 系统指标汇聚 |
| `vllm_metrics_logger.py` | vLLM 指标 |
| `thread_pool_metrics_logger.py`(265) | 2048 线程池水位 |
| `otlp_receiver.py`(246) | 本地 OTLP gRPC collector，承接 vLLM metrics（torch.profiler/telegraf），推理服务器启动前拉起 |

## 4. 终端日志（`utils/tlog/`）

- `logger.py`：`QorixLogger`(:161)、`LogRecord`(:265)、`BufferedLogHandler`(:274，缓冲避免高频日志拖慢训练)、`_strip_ansi`(:307)、`FileTailer`(:322)、`DirTailer`(:365)、`drain_all_log_buffers`(:408)、`setup_file_tailers`(:419)、`setup_logging`(:457，创建 `logs/{orchestrator,trainer,inference}/{comp}.log` 与 `.detailed.log`，全局防重入)、`get_logger`(:564)。
- `noise_filter.py`：抑制重复告警。
- 路径约定：`QORIX_RUN_DIR/logs`（`utils/paths.py`，含 `sandboxes/`、`stdout/` 等子目录）。

## 5. 消费端契约（qorix-ui 侧）

- 事件经 zip artifact 上传，DB 按 family 分表：`events_orchestrator/events_trainer/events_rollout/events_infra/prompts/generations/env_responses/tool_calls/samples_data/rollouts_metrics/golden_answers/sample_tags/info_turns` + `_eval/_discarded/_cancelled` 变体。
- `section/group/metric` 三层 → UI Metrics 页的指标树/图表。
- zstd 压缩列 → UI `decompress_blob` 读取（qorix-ui db.py:30）。
- `tail_idx` → `ingested_tails` 幂等去重保证增量同步正确性。