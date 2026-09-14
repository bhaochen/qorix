# 模块地图与符号索引

> 34K LOC / 82 文件的导航手册。需要找某段逻辑时,按「功能 → 模块 → 符号:行号」直达源码。本表所有符号均在 `src/qorix/` 下。

## 1. 帮你看懂缩略语

| 缩略 | 全称 | 含义 |
|------|------|------|
| TP / PP / CP / EP / DP | Tensor / Pipeline / Context / Expert / Data Parallel | Megatron 并行维度 |
| FSDP(DCP) | Fully Sharded Data Parallel / Distributed Checkpointing | 权重分片 + 分布式检查点 |
| TIS / ICE-POP | Truncated Importance Sampling / Ice-POP | off-policy 截断技术 |
| TTFT | Time To First Token | 首 token 延迟 |
| tail_idx | tail window index | 事件上传窗口标号 |
| IS (masking) | Importance Sampling | 序列级重要性采样掩码 |

## 2. 功能 → 模块 → 符号

### 训练主流程
| 你要找的内容 | 位置 |
|--------------|------|
| 主循环 `run()` 三段并发 | orchestrator/orchestrator.py:613 |
| 权重更新 watcher | orchestrator/orchestrator.py `_weight_update_watcher`:776 |
| rollout 事件循环 | orchestrator/orchestrator.py `_rollout_loop` |
| 启动/关停 | orchestrator/orchestrator.py `start_processes`:281 / `stop_processes`:515 |
| 入口 train.py | 根目录 train.py:1 → orchestrator.py `main()`:3279 |
| 线程池注册（非编译探针日志） | `_register_thread_pools`（:635 调用） |

### 调度与异步
| 你要找的内容 | 位置 |
|--------------|------|
| 延迟聚合调度 | `_schedule_dispatch`:972 / `_deferred_dispatch_loop`:984 |
| 组模式派单 | `_start_next_prompt`:996（lane 预分配 :1029-1058） |
| 个体样本车道派单 | `_start_next_individual`:1064 / `_dispatch_one_queued_sample`:1141 |
| 服务器挑选(绕过 eval) | `_pick_server_with_capacity`:949 / `_pick_server_with_any_capacity`:1133 |
| 样本生命周期回调 | `_make_sample_lifecycle`:1183 |
| 异步水位计算 | `_compute_async_level`（在 _start_next_prompt 中调用） |
| off-policy 取消 | `_update_inference_weights` :2555-2590 |
| 组批+优势 | `_assemble_single_turn_group`:1651 / `_assemble_multiturn_group`:1752 |

### 生成
| 你要找的内容 | 位置 |
|--------------|------|
| 同步 HTTP（绕 async 瓶颈） | orchestrator/generate.py `_sync_http_post`:42（2048 线程） |
| 单轮生成入口 | orchestrator/generate.py `generate_completion_with_tokens`:439 / `generate_completion`:547 |
| 多轮 interleaved | orchestrator/generate.py `run_multiturn_rollout`:698 |
| 多轮样本驱动 | orchestrator/generate.py `process_multiturn_sample`:1258 |
| 优点计算 | orchestrator/generate.py `compute_advantages`:1784 |
| 权重广播落库 | orchestrator.py `_update_inference_weights`:2479（weight_sync InfraEvent :2527-2541） |
| 重试封装 | generate.py 顶部 `_retry_request` |

### 权重同步
| 你要找的内容 | 位置 |
|--------------|------|
| 广播根（FSDP rank0） | trainer/backends/fsdp.py `is_weight_broadcast_rank`:559 |
| Megatron gather→HF | trainer/backends/megatron.py `gather_weights_for_inference`:2458 |
| CPU staging 单副本 | trainer/weight_sync.py `_stage_state_dict_to_cpu`:38 |
| 分组桶构造 | weight_sync.py `_build_flattened_bucket_chunks`:70 |
| 元数据增量协议 | weight_sync.py `_broadcast_metadata`:123 |
| 广播发送 | weight_sync.py `broadcast_weights_to_inference`:348 |
| 接收侧 | inference/worker.py `receive_state_dict`:108 / `_receive_metadata`:148 |

## 3. 训练后端

| 你要找的内容 | 位置 |
|--------------|------|
| 抽象接口 | trainer/backends/__init__.py `TrainingBackend`:23 |
| FSDP2 初始化（2D mesh） | trainer/backends/fsdp.py `init`:84 |
| 逐模块 fully_shard | fsdp.py `_build_model`:165 |
| 微批循环 + gradient sync 优化 | fsdp.py `train_step`:294（末微批同步 :363-369） |
| Megatron 初始化 | trainer/backends/megatron.py `init`:1620 |
| Megatron HF↔TransformerConfig | megatron.py `_hf_to_transformer_config`:503 / `_convert_qwen_to_hf`:647 |
| 微批装箱 | trainer/micro_batch.py `MicroBatch`:15（FFD 装箱） |
| 上下文并行 shard | trainer/context_parallel.py `shard_for_cp`:13 |
| 损失核心 | trainer/loss.py `_masked_mean_per_sample`:21 |
| GPU 时间线 | trainer/metrics/timeline.py `GPUTimelineLogger`:54 |

## 4. 环境与评测

| 你要找的内容 | 位置 |
|--------------|------|
| 自动发现（AST 依赖预检） | environments/registry.py `_find_env_class`:41 / `_read_packages`:57 |
| 环境基类 | environments/base.py（Sample:27 / RewardResult:35） |
| Reward 组合 | environments/rewards.py `Rubric`:89 / `maybe_await`:36 |
| 工具环境 | environments/tool_env.py `ToolEnvironment`（829 行） |
| 沙箱抽象 | environments/_sandbox/base.py `SandboxProvider` ABC + `SandboxHandle`:19 / `SandboxConfig`:32 |
| 沙箱池 | environments/_sandbox/pool.py:388 |
| Eval 注册 | evals/registry.py `_load_eval_class`:55 |
| 独立评测 driver | eval_standalone/driver.py `run_eval`:827 |

## 5. 配置 / 运行时 / 日志

| 你要找的内容 | 位置 |
|--------------|------|
| 三层合并 | utils/config_loader.py `_deep_merge`:40 / `_coerce_value`:54 |
| Schema 字段 | utils/config_schema.py（EnvironmentEntry:20 / QorixConfig） |
| Ray actor | utils/ray_runtime/runtime.py `InferenceServerActor`:1271 |
| 终端日志 | utils/tlog/logger.py `setup_logging`:457 |
| 事件数据类 | orchestrator/loggers/event_logger.py（Event:656 / GenerationRollout:742 / StepMetric:768） |
| W&B 三层命名 | orchestrator/loggers/wandb_logger.py `log_trainer_metrics_payload`:767（:788-794 拼键） |
| checkpoint→HF | utils/checkpoint_converter.py `convert_single`:374 |

## 6. 配置文件

| 文件 | 用途 |
|------|------|
| `configs/defaults/default_train.yaml` | 全参数参考（338 行，头注释即文档） |
| `configs/defaults/default_eval.yaml` | 评测默认 |
| `configs/examples/example_countdown.yaml` | 最小可跑示例（Qwen2.5-3B GRPO） |
| `configs/examples/example_hendrycks_math*.yaml` | 数学训练/带 eval |
| `configs/examples/example_wordle.yaml` | 多轮环境示例 |
| `configs/examples/example_i3_code.yaml` | 代码 + 沙箱 |
| `configs/examples/example_dapo_math.yaml` / `example_deepscaler.yaml` | DAPO / DeepScaleR |
| `configs/run.yaml` | 用户运行配置样板 |

## 7. 调试常用（符号 → 打点）

| 观察点 | 方法 |
|--------|------|
| 权重更新是否阻塞 | `timeline.py` weight_sync InfraEvent（UI Timeline） |
| 是否打了老模型 | `inflight_rollout_info[*].sample_off_policy_steps` 计数（:2555-2566） |
| 沙箱池预热 | `eager_prepare_resources`(default_train.yaml:49) |
| 单个样本为何卡住 | `individual_sample_queues[server]` deque（:1125） |
| vLLM 显存 | `/torch_memory` 控制路由（server.py:95） |