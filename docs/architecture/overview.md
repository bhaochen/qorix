# 系统架构概览

> Qorix 用 **编排 (orchestrator) × 推理 (inference) × 训练 (trainer)** 三组件在 Ray 集群上协同完成 LLM RL 后训练：编排器负载均衡地向 vLLM 推理服务器发 prompt，收生成后由环境奖励打分、组批，送到 trainer（FSDP/Megatron）做梯度步，再把权重经 NCCL 广播回推理端——两端互不等候。

## 1. 代码组织（34K LOC / 82 文件）

```
train.py / eval.py                     # 根入口脚本（pyproject scripts: train= "train:main", eval= "eval:main"）
src/qorix/
  orchestrator/                        # 编排器：调度、容量、优势、协调（~14K LOC）
    orchestrator.py                    #   主循环 run()（3373 行）
    generate.py                        #   推理调用与 rollout 组装（1819 行）
    batch_processor.py / scheduler.py  #   批次处理、服务器调度
    env_worker_pool.py                 #   奖励函数子进程池
    eval_runner.py                     #   训练中定期评测
    loggers/                           #   事件/W&B/系统监控     （~8K LOC）
  inference/                           # vLLM 服务器 + NCCL worker
    server.py                          #   vLLM 启动与控制路由（173 行）
    worker.py                          #   NCCLWeightUpdateWorker（311 行）
  trainer/                             # 训练后端（~8K LOC）
    backends/fsdp.py                   #   FSDP2 后端（743 行）
    backends/megatron.py               #   Megatron 后端（3034 行）
    weight_sync.py                     #   权重广播发送侧（439 行）
    loss.py / micro_batch.py           #   损失函数、微批装箱
  environments/                        # 训练环境（自动发现）
    registry.py / base.py / rewards.py
    countdown/ wordle/ hendrycks_math/ deepdive/ deepscaler/ dapo_math/ i3_code/
    _sandbox/                          #   Prime/Modal/Daytona/E2B 沙箱抽象
    tool_env.py / parsers.py
  evals/                               # 评测目标（自动发现）
    base.py / registry.py / aime_2024/ aime_2025/ gpqa_diamond/ math500/
  eval_standalone/                     # 离线评测 driver（830 行）
  utils/
    config*.py / config_loader.py      # 三层配置系统
    ray_runtime/runtime.py             # Ray 集群与 actor 编排（1302 行）
    tlog/                              # 终端日志系统
    checkpoint_converter.py            # FSDP/Megatron → HF 转换
configs/defaults/default_train.yaml    # 默认配置（全参数参考）
configs/defaults/default_eval.yaml     # 评测默认配置
configs/examples/*.yaml                # 示例训练配置
docker/Dockerfile / Dockerfile.base    # 构建镜像（base 复用上游 vLLM 镜像）
```

## 2. 三组件与 Ray

| 组件 | 位置 | 职责 |
|------|------|------|
| **Orchestrator** | 单进程，训练机 CPU 节点 | 加载环境数据集 → 采样 prompt → 调度推理 → 打奖励 → 计算优势 → 组训练批 → 协调权重广播 → 上报全链路事件/指标 |
| **Inference** | N 个 vLLM server × TP size（GPU actor） | 提供 OpenAI `/v1/completions`；以 `qorix.inference.worker.NCCLWeightUpdateWorker` 扩展 worker 接收权重广播（`server.py:42`） |
| **Trainer** | trainer_num_workers 个 worker | FSDP2 或 Megatron 后端：批次预处理、微批/打包、梯度步、保存 checkpoint、gather 权重待广播 |

**Ray 资源模型**（`utils/ray_runtime/runtime.py`）：
- 推理端每个 `InferenceServerActor`（`:1271`）持有一个 vLLM 服务器进程（TP 维度），TP size 内共享 GPU。
- 训练端 count = FSDP/Megatron 的 world size。
- 放置策略可选 `PACK / SPREAD / STRICT_PACK / STRICT_SPREAD`（`runtime.py:38`）。
- 环境变量透传（`_SETUP_ENV_KEYS`，`:39-53`）：`CUDA_VISIBLE_DEVICES`、`MASTER_ADDR/PORT`、`RANK/LOCAL_RANK/WORLD_SIZE`、`NCCL_SOCKET_IFNAME`、`QORIX_RUN_DIR`、`QORIX_CHECKPOINT_DIR` 等。

## 3. 启动与关停（orchestrator.py）

```
train.py main() → config → orchestrator.main() (:3279) → run() (:613)
```

- `start_processes`(:281)：`init_ray_cluster` → 启动 **OTLP 接收器**（先于推理服务器，采集 vLLM 指标）→ 部署推理服务器（含 eval 服务器与 baseline eval 逻辑 :462/:492）→ 部署 trainer worker。
- `run()`(:613) 并行三个任务：
  1. **权重更新 watcher**：`ray.get(weight_sync_load_refs)` 推进 `inference_model_step` / `eval_server_model_step`（:2522-2553）；
  2. **rollout 事件循环** `_rollout_loop`：采样/调度 dispatch；
  3. **指标上报循环** `wandb_logger.start_event_upload_loop`(:637)。
- 容量 = `max_concurrent_prompts_per_server × len(server_urls)`。
- HTTP：编排器侧 `httpx.AsyncClient(max_connections=8192, max_keepalive_connections=0, timeout=1200)`(:642)；生成侧独立线程池（见 data-flow）。
- 训练开始前执行 baseline eval（resume 时跳过，:645-646）。
- `stop_processes`(:515)：终止全部服务器与 trainer worker，恢复本地 torch。

## 4. 模块职责速查表

| 文件 | 关键符号 | 职责 |
|------|---------|------|
| `orchestrator/orchestrator.py` | `run` :613 / `_rollout_loop` / `_update_inference_weights` / `_assemble_single_turn_group` :1651 / `_assemble_multiturn_group` :1752 | 主循环、调度、组批、优势、协调 |
| `orchestrator/generate.py` | `generate_completion` :547 / `_sync_http_post` :42（stdlib http.client，2048 线程） / `run_multiturn_rollout` :698 | 推理调用、错误归一、多轮 interleave |
| `orchestrator/env_worker_pool.py` | — | 奖励函数放**子进程**执行（防 GIL / 崩溃隔离） |
| `orchestrator/scheduler.py` | — | 服务器调度策略 |
| `inference/server.py` | `build_base_args` :30 / `attach_control_routes` :63 / `run_server` :155 | vLLM 启动、控制路由 |
| `inference/worker.py` | `NCCLWeightUpdateWorker` :31 / `receive_state_dict` :108 / `load_weights` :207 | 接收权重广播 |
| `trainer/backends/fsdp.py` | `FSDPBackend` :59 / `train_step` :294 / `gather_weights_for_inference` :520 / `save_checkpoint` :594 | FSDP2 训练 |
| `trainer/backends/megatron.py` | `MegatronBackend` :1590 / `gather_weights_for_inference` :2458 / `_convert_qwen_to_hf` :647 | 张量/流水/专家并行训练 |
| `trainer/loss.py` | 头部注释 :5-12 | 全部 RL 损失 + 修饰 |
| `trainer/weight_sync.py` | `broadcast_weights_to_inference` :348 / `_broadcast_flattened_buckets` :157 | 权重广播发送侧 |
| `environments/registry.py` | `get_environment` :158 / `check_environments` :187 | 自动发现 + 依赖预检 |
| `evals/registry.py` | `_load_eval_class` :55 | 评测自动发现 |
| `eval_standalone/driver.py` | `run_eval` :827 | 离线评测全流程 |
| `utils/ray_runtime/runtime.py` | `InferenceServerActor` :1271 | Ray actor 编排 |
| `utils/config_loader.py` | `parse_args_and_load` / `_deep_merge` :40 | 三层配置合并 |
| `utils/checkpoint_converter.py` | `convert_single` :374 | 检查点转 HF |

## 5. 设计要点

- **无阻塞流水**：推理端只要编排器喂 prompt 就一直生成；训练端只要有 batch 就训练——重叠消除 GPU 空闲。
- **权重广播双组**：`full`（trainer + 全部推理含 eval）与 `training_only`（trainer + 仅训练服务器）分离，eval 服务器只进 `full` 组，避免拖慢训练链路。
- **幂等增量**：非求，整个评测与事件上报以 `tail_idx` / `step` 为幂等键，断点续传不丢不重。
- **单一事实源**：配置从 defaults → run config → CLI 覆盖逐层收敛；所有运行参数在 `configs/defaults/default_train.yaml`。