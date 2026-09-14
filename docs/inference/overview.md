# 推理服务器

> 推理 = vLLM OpenAI 兼容服务 + 自定义扩展 worker，N 台服务器 × TP size 部署在 Ray GPU actor 内。控制面通过附加路由（health/collective/load_weights/torch_memory），权重面通过 NCCL 广播（见 weight-broadcast.md）。

## 1. 服务器组装（`inference/server.py`，173 行）

- `build_base_args`(:30)：构造 vLLM 参数。关键：
  - `worker_extension_cls = "qorix.inference.worker.NCCLWeightUpdateWorker"`(:42)；
  - `enable_prefix_caching=False`(:47)。
- `attach_control_routes`(:63) 挂载自定义路由（FastAPI app）：

| 路由 | 行号 | 用途 |
|------|------|------|
| `GET /testing` | :66 | 健康检查 |
| `GET /collective_test` | :70 | 集体通信（NCCL）探测 |
| `POST /init_broadcast` | :75 | body `{host, port, world_size, rank, group}` → collective RPC 初始化 NCCL broadcast |
| `GET /load_weights?group=full|training_only` | :88 | 触发 worker `load_weights` |
| `GET /torch_memory` | :95 | 经 collective RPC 从各 worker 进程取 torch allocator 指标（**强调来自真正持张量的 vLLM worker 进程而非 API 进程**） |

- 启动封装：`custom_build_async_engine_client`(:120)、`custom_run_server_worker`(:128，`build_app` + `attach_control_routes` + `serve_http`)、`custom_run_server`(:149)、`run_server`(:155，argparse 的 host/port/model 默认取 `cfg.inference_host`/`cfg.inference_base_port`；`uvloop.run`)。

## 2. 采样请求契约（generate.py 视角）

生成请求体：`return_token_ids=True`、`n=1`、`logprobs=1`（为 TIS/PPO clip 用）、`skip_special_tokens=False`（保证 token_ids 与文本对齐）、`include_stop_str_in_output=False`、`priority`（vLLM priority 调度）；`max_tokens` 按 `max_model_len - prompt_len` 动态收缩（generate.py:466-483）。

错误归一（generate.py:527-542）：
- `_parse_context_length_error`(:391) → `PromptTooLongError`
- `_parse_max_tokens_error`(:410) → `ContextExhaustedError`
- 其余 → `RolloutError`

## 3. 部署拓扑（`utils/ray_runtime/runtime.py`）

- `InferenceServerActor`(:1271) `@ray.remote(max_restarts=0)`：每个 actor 持有 1 个 vLLM 服务器进程（TP 维度）。
  - `__init__`(:1274)：install_config、`ray.util.get_node_ip_address`/`gethostname`/node_id、`_resolve_cluster_node_index`、`_pick_free_port`、`tp_group_id=server_idx`；
  - **`_collect_visible_gpu_hardware` 在 vLLM 子进程占用设备前收集**；
  - `public_url`(:1294) = `http://{node_ip}:{port}`；`_server_info_payload`(:1298)。
- eval 服务器单独部署（`eval_num_servers`），不参与训练采样；只加入 `"full"` 权重组。

## 4. 推理规模配置

| 配置 | 说明 |
|------|------|
| `inference_num_workers` | 训练用服务器数量 |
| `inference_tensor_parallel_size` | 每服务器 TP 维度（占用同节点多 GPU） |
| `gpu_memory_utilization` | vLLM 显存上限 |
| `max_model_len` / `max_num_seqs` / `max_num_batched_tokens` | 序列长度与批容量 |
| `vllm_scheduling_policy` | priority / fcfs |
| `enable_thinking` / `chat_template` | 推理格式控制 |

## 5. 与编排器的连接

- 容量调度：`_pick_server_with_capacity`(orchestrator.py:949) 跳过 eval 服务器。
- 权重更新生命周期完全由编排器驱动（`_update_inference_weights`，见 orchestrator/data-flow.md §5）。
- OTLP 接收器 `orchestrator/loggers/otlp_receiver.py`(246 行) 先于推理服务器启动，承接 vLLM metrics（torch.profiler/telegraf 来源），采样上 vllm_metrics 表。