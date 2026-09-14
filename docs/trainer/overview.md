# 训练器深度解析

> 训练器在 Ray worker 中运行，提供 **FSDP2**（研究/原型，PyTorch 原生路径）与 **Megatron**（大规模扩展，TP/PP/CP/EP）两套后端。`TrainingBackend` ABC（`trainer/backends/__init__.py:23`）统一接口，`create_backend` 由 `runtime.py` 按 `train_backend` 选择。

## 1. 抽象（`backends/__init__.py`）

- `TrainingBackend`(:23)，`init` 抽象(:39)。docstring(:5-6)：FSDP 走 PyTorch 原生路径；Megatron 走张量/流水并行。
- worker 由 Ray actor 拉起，环境变量 `WORLD_SIZE/RANK/LOCAL_RANK/MASTER_ADDR` 由 `runtime.py` 注入。

## 2. FSDP2 后端（`backends/fsdp.py`，743 行）

**初始化**（`init` :84）：
- nccl 初始化；`init_device_mesh` 建 2D mesh（dims `("dp","cp")`），`_flatten("dp_shard_cp")` 把 CP 折进 FSDP shard；CP 组/rank 另存。

**模型构建**（`_build_model` :165）：
- `AutoModelForCausalLM` + `flash_attention_2`（缺失回退 sdpa，CP>1 强制报错）；gradient_checkpointing。
- `MixedPrecisionPolicy(param_dtype=mp, reduce_dtype=f32)`。
- 利用 HF `_no_split_modules` 逐模块 `fully_shard`（embeddings 在非 tie 时单独 shard），最后整体 `fully_shard`。
- `_build_optimizer` = AdamW(lr, weight_decay)。

**训练步**（`train_step` :294）：见 `rl-algorithms.md` §4 的顺序清单。

**权重 gather**（`gather_weights_for_inference` :520）：FSDP 下 rank 0 作为广播 root（`is_weight_broadcast_rank` :559）。

**检查点**（`save_checkpoint` :594 / `load_checkpoint` :699）：
- DCP（Distributed Checkpointing）格式；可冻结 checkpoint dir 约定 `checkpoint_dir/step_N/`（meta.json 原生标记）。
- RNG 状态单独存取（`_get_rng_state`/`_load_rng_state` :578/:587）。

## 3. Megatron 后端（`backends/megatron.py`，3034 行）

**符号速查**：
| 符号 | 行号 | 职责 |
|------|------|------|
| `_pad_vocab_size` | :62 | vocab 对齐 |
| `_build_packed_segment_attention_mask_from_position_ids` | :145 | 打包注意力掩码 |
| `_compute_rl_loss_vocab_parallel` | :244 | vocab 并行 RL 损失 |
| `_hf_to_transformer_config` | :503 | HF → Megatron TransformerConfig |
| `_convert_qwen_to_hf` | :647 | Qwen 专用权重名映射 |
| `_gather_ep_expert_params` | :838 | EP→HF 全量专家参数 |
| `_gather_pp_state_dicts` | :915 | 流水并行聚合 |
| `_all_gather_tp_param` | :986 | TP 聚合 |
| `_free_grad_buffers`/`_restore_grad_buffers` | :1025/:1051 | 广播期间释放/恢复梯度 buffer |
| `_build_gpt_model` | :1067 | GPTModel + DDP optimizer |
| `_create_bridge` | :1180 | bridge 导出 |
| `_load_hf_checkpoint(_manual)` | :1196/:1278 | HF 检查点加载 |
| `init` | :1620 | 后端初始化 |
| `_build_model_and_optimizer` | :1770 | 模型+分布式优化器 |
| `_compute_batch_logprobs` | :1871 | batch 级 ref logprobs |
| `gather_weights_for_inference` | :2458 | 为推理 gather（见 weight-broadcast.md §4） |

**并行维度**：TP/PP/CP/EP 尺寸来自 cfg；sequence parallel 仅 TP>1 启用；可选 `optimizer_cpu_offload`、`gradient_checkpointing`、`distributed_optimizer`、`overlap_grad_reduce`、TransformerEngine、unified memory JIT（可禁用以跳过启动 JIT）。

## 4. 微批与序列打包（`micro_batch.py`）

- `MicroBatch`(:15)：`{input_ids, loss_mask, advantages, vllm_logprobs, position_ids}`；`can_fit`(:31)；`add_sample`(:35，每样本 position_ids 从 0 重排)。
- 文档(:5-12)：序列打包 + 动态微批 + **First Fit Decreasing 装箱**。

## 5. 上下文并行（`context_parallel.py`）

- `shard_for_cp`(:13)：沿 dim1 `torch.chunk`。
- `_get_cu_seqlens_from_position_ids`(:31)：为 `ring_flash_attn` 布局 cuselens。
- `setup_cp_params`(:47)。

## 6. 训练侧指标（`trainer/metrics/`）

- `timeline.py`：`GPUTimelineLogger`(:54，`start_step` :78 / `track` :92 / `finalize_step` :144 / `get_serializable_events` :195)；`GPUEvent`(:38, `duration_ms` :50)；`_NullTracker`(:212) 关闭开关。
- `torch_memory_logger.py`：`TorchMemorySample`(:27, `to_dict` :38)、`TorchMemoryLogger`(:51)。

## 7. 调参要点

- `trainer_num_workers` = 并行 world size；FSDP 下 CP 通过 `fsdp_context_parallel_size` 折进 dp_shard。
- `number_of_minibatches` 更大 → 更细粒度梯度步，但 ref logprobs（`ppo_clip_ref_logprobs=="batch"`）开销在 batch 级。
- 打包 + 微批 + FFD 装箱下，`pad_to_multiple_of` 与 `seq_len` 决定 batching 效率。