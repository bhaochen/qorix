# 评测系统

> 两条评测路径：**训练中定期评测**（`orchestrator/eval_runner.py`，独立 eval 服务器，`eval_every` 触发）与**训练后离线评测**（`eval_standalone/driver.py`，对保存的 checkpoint 逐阶段转换+推理+上传）。评测目标同样自动发现。

## 1. 注册与发现（`evals/registry.py`）

- 解析顺序：`evals/<name>/eval.py` → 回退 `environments/<name>/environment.py`（`_load_eval_class` :55；`_find_eval_class` :39；`_CLASS_CACHE` :33）。
- `Eval` 基类（`evals/base.py` :42）两种形态：
  1. **包裹已有环境**：`environment_name` 指向环境，重写 prompt/采样/打分逻辑，`__getattr__` 透传环境属性；
  2. **完全独立**：`EvalMetricsResult` 自产指标。

## 2. 内置评测目标

| Eval | 文件 | 说明 |
|------|------|------|
| AIME 2024 | `evals/aime_2024/eval.py` (:28) | 美国数学邀请赛 2024 |
| AIME 2025 | `evals/aime_2025/eval.py` (:28) | 美国数学邀请赛 2025 |
| MATH-500 | `evals/math500/eval.py` (:28) | 500 题数学基准 |
| GPQA Diamond | `evals/gpqa_diamond/eval.py` (:43, `extract_multiple_choice` :30) | 研究生级问答 |

## 3. 训练中评测（`orchestrator/eval_runner.py`）

- 配置：`eval_before_training` / `eval_after_training` / `eval_num_servers` / `eval_start_end_use_all_servers`；环境级 `eval_every`（>=1）按步触发。
- eval 服务器不参与训练采样，只加入 `"full"` 权重组，模型版本由 `eval_server_model_step` 管理（orchestrator/data-flow.md §5）。

## 4. 离线评测驱动（`eval_standalone/driver.py`，830 行）

- `run_eval`(:827) = `setup_logging()` + `asyncio.run(_run_eval_async)`。
- **checkpoint 发现**：`_discover_checkpoints`(:85，按 `meta.json`（原生）或 `config.json`（HF）标记 + `_STEP_RE`)、`_resolve_checkpoints`(:111，显式列表优先、同 step 去重)、`_convert_checkpoint`(:140，调 `utils.checkpoint_converter`)、`_cleanup_converted`(:164)、`_install_compat_config`(:175)。
- **配置**：`_parse_eval_configs`(:231) 继承 `configs/defaults/default_eval.yaml`：`wandb_run_path`/`evals` 必填；`hf_weights:false` 时逐 checkpoint on-the-fly 转 HF；推理 `num_workers 1`、tp 1、`gpu_memory_utilization 0.9`、`max_model_len 4000`、`max_concurrent_samples_per_server 256`、`enable_thinking false`、`temperature 1.0`、`max_tokens 3700`；`ray_namespace qorix-eval`。
- **两阶段收集**：`_collect_eval_objects`(:280) 先全部 prompt 预取 → 各服务器 rollouts；每样本 `log_eval_prompt` / `log_eval_rollout`。

## 5. Arrow 表与上传

所有 eval 结果转 **Arrow 表**（`_zstd_compress` :276 压缩）：

| 函数 | 表内容 |
|------|--------|
| `_eval_prompts_to_table` :392 | prompt |
| `_eval_generations_to_table` :411 | 生成文本 |
| `_eval_env_responses_to_table` :443 | 环境响应 |
| `_eval_tool_calls_to_table` :472 | 工具调用 |
| `_samples_data_eval_to_table` :516 | 样本数据 |
| `_rollouts_metrics_eval_to_table` :535 | rollout 指标 |
| `_golden_answers_eval_to_table` :560 | 参考答案 |
| `_info_turns_eval_to_table` :584 | 信息轮次 |
| `_sample_tags_eval_to_table` :616 | 样本标签 |

`_upload_eval_zip`(:645) 压缩后上传 W&B（`$WANDB and eval` 相关 artifact）——UI 端 `generations_eval` 等表按 `COUNT(DISTINCT sample_idx)` 聚合（qorix-ui main.py）。

## 6. 评测配置入口（`eval_standalone/config_loader.py` + `config_schema.py`）

- `eval.py` → `eval_standalone.config_loader.parse_args_and_load()`（127 行）+ 扁平 Schema（108 行，`extra="forbid"`）。
- 字段：`wandb_run_path`、`evals`、`checkpoints`/`checkpoint_dir`、`hf_weights`、推理规模与 Ray 参数。

## 7. 查看结果

- 训练中 eval：`eval_runner` 会把指标写入事件日志/W&B（UI Evals 页 + Metrics 页可看）。
- 离线 eval：生成的 eval 表直接在 qorix-ui 的 **Evals** 页面与 rollouts 页面浏览（需 sync 对应 run）。
- 参照 `configs/defaults/default_eval.yaml` 与 `configs/evals/eval.yaml`。