# 配置系统

> 三层配置：**defaults（`configs/defaults/default_train.yaml`）→ run config（`--config`，缺省回退 `configs/run.yaml`）→ CLI 覆盖**。所有字段经 pydantic `QorixConfig`（`extra="forbid"`）校验，扁平结构。

## 1. 三层合并（`utils/config_loader.py`）

- `parse_args_and_load()`：`_deep_merge`(:40) 递归合并字典，**列表整体替换**（非按索引覆盖）；`_coerce_value`(:54) 尽力按目标类型转换 CLI 字符串。
- 全局单例（`utils/config.py`）：`_cfg`(:20)、`get_config()`(:23-29) 惰性加载默认运行配置、`install_config`(:32-39) 由 load 阶段注入、`cfg` 代理(:42-51) 提供属性访问。
- 训练入口：`train.py:8` → `orchestrator.main()`；评测入口：`eval.py:13` → `eval_standalone.config_loader`（独立 Schema，见 `dst`）。

## 2. Schema（`utils/config_schema.py`）

扁平结构，`extra="forbid"`。字段分组：

| 分组 | 代表字段 | 说明 |
|------|---------|------|
| 基本 | `model`、`model_dtype`、`mixed_precision_dtype` | float32/float16/bfloat16 |
| 环境 | `environments: list[EnvironmentEntry]` | name/weight/kwargs/reward_min/max |
| Ray | `ray_address`、`ray_auto_start_local`、`ray_namespace`、placement strategy、`ray_broadcast_*` | 集群/广播通道（:91-107） |
| 规模 | `inference_num_workers`、`inference_tensor_parallel_size`、`trainer_num_workers` | 服务器与 worker 数（:110-112） |
| 流量 | `max_concurrent_prompts_per_server`、`prompts_batch_size_for_trainer`、`number_of_steps`、`max_async_rollout`、`enable_prompt_prefetch`、`enable_individual_sample_lanes`、`max_off_policy_steps`、`discard_group_zero_advantage` | 异步管线旋钮（:115-125） |
| 优化器 | `learning_rate`、`weight_decay`、`grad_clip`、`lr_scheduler`（none/constant/linear/cosine）、`warmup_steps`、`min_lr_ratio` | （:128-134） |
| 后端 | `train_backend`（fsdp/megatron）、`fsdp_context_parallel_size`、`megatron_*`（TP/PP/CP/EP、distributed_optimizer、transformer_engine、sequence_parallel、gradient_checkpointing、optimizer_cpu_offload…） | （:135-159） |
| RL 算法 | `algorithm`、`number_of_minibatches`、`use_ppo_clip`、`clip_low/high`、`ppo_clip_ref_logprobs`、`sapo_tau_pos/neg`、`dr_grpo_loss_agg_mode`、`advantage_norm`、`use_tis`/`tis_cap`/`tis_mode`/`tis_floor`、`kl_penalty_tau`、`kl_estimator`(k2/k3)、`entropy_coef` | （:163-187） |
| 稳定性过滤 | `filter_overlong`、`filter_gibberish`、`filter_repetition` 及各自阈值 | （:190-199） |
| 权重广播 | `weight_broadcast_mode`（flattened_bucket/per_tensor）、`weight_broadcast_bucket_mb`、`weight_broadcast_cpu_staging`、`weight_broadcast_pin_memory`、`weight_broadcast_free_grad_buffers` | （:202-206） |
| 推理 | `seq_len`、`pad_to_multiple_of`、`inference_host`、`inference_base_port`、`gpu_memory_utilization`、`max_model_len`、`max_num_seqs`、`vllm_scheduling_policy`（priority/fcfs）、`enable_thinking`、`chat_template` | （:209-221） |
| 采样 | `group_size`、`temperature`、`top_p`、`max_tokens`、`interleaved_rollouts`；`get_sampling_params()`(:232) | group 语义 + 采样 |
| 检查点 | `checkpoint_every`、`checkpoint_save_training_state`、`resume_from_checkpoint`、`checkpoint_dir`、`checkpoint_keep_last/every` | （:238-243） |
| W&B/指标 | `use_wandb`、`wandb_project`、`wandb_run_name`、`wandb_tags`、`wandb_upload_code/logs/…`、各采样间隔、`track_gpu_events` | （:246-264） |
| 评测 | `eval_before_training`、`eval_after_training`、`eval_num_servers`、`eval_start_end_use_all_servers` | （:267-270） |

环境入口 `EnvironmentEntry`（:20-25）：`{name, weight, kwargs, reward_min, reward_max}`。weight 默认 1.0。
评测入口 `EvalEntry`（:28-34）：`{name, eval_every, pass_k, num_samples, separate_eval_samples, kwargs}` + `get_sampling_overrides()`(:40-45)。

## 3. 典型的 `configs/run.yaml`

```yaml
model: Qwen/Qwen2.5-3B
train_backend: fsdp
algorithm: grpo
learning_rate: 1e-6
inference_num_workers: 2
trainer_num_workers: 2
max_concurrent_prompts_per_server: 32
prompts_batch_size_for_trainer: 16
number_of_steps: 300
max_async_rollout: 2
enable_prompt_prefetch: true
enable_individual_sample_lanes: true
environments:
  - name: countdown
    weight: 1.0
    reward_min: 0.0
    reward_max: 2.0
wandb_project: qorix
```

## 4. 覆盖与验证

- CLI 覆盖任何参数：`uv run train.py --config ... --learning_rate 5e-7 --number_of_steps 500`。
- `use_wandb` 下运行自动打 tag `qorix`（`wandb_tags`），供 UI 自动发现。

## 5. 评测配置（`eval_standalone/config_schema.py`）

独立扁平 Schema（`extra="forbid"`）：`wandb_run_path`、`evals`、`checkpoints`/`checkpoint_dir`、`hf_weights`、推理规模与 Ray 参数。默认值见 `configs/defaults/default_eval.yaml`（tp=1、`max_model_len 4000`、`max_concurrent_samples_per_server 256`、`ray_namespace qorix-eval`）。