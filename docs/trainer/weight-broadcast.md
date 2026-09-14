# 权重广播协议

> 训练端把更新后的权重经 **NCCL 广播** 推给全部推理服务器。发送侧在 `trainer/weight_sync.py`，接收侧在 `inference/worker.py`。协议默认 **flattened_bucket_v1**：按 dtype 分桶 → 归并扁平张量 → 分块广播；metadata 首步全量 pickle 广播，后续仅发 `size=0` 由接收端复用缓存，避免每步重发张量名/shape。

## 1. 双 NCCL 通信组

| 组 | 成员 | 用途 |
|----|------|------|
| `"full"` | trainer + 全部推理服务器（含 eval 服务器） | 全量同步 |
| `"training_only"` | trainer + 仅非 eval 推理服务器 | 训练链路同步 |

- eval-only 服务器仅加入 `"full"` 组，避免拖慢训练权重更新。
- 通信组经 `setup_inference_communicator`(:211) 与 `setup_inference_communicator_for_group`(:260) 建立（按加入服务器的 IP:port 列表）。
- 接收侧惰性初始化：`NCCLWeightUpdateWorker._ensure_communicators_dict`(:43) 首次用时创建 `PyNcclCommunicator`（基于 `StatelessProcessGroup`）。

## 2. 发送侧（`trainer/weight_sync.py`）

- 模块级缓存（:24-25）：`_metadata_sent_comms` / `_flattened_chunks_cache`——同架构下 metadata 与 chunk 布局只算一次。
- `_stage_state_dict_to_cpu`(:38)：逐 key `detach().cpu()`，**最多同时存在一份参数的 GPU+CPU 副本**，随后 `torch.cuda.empty_cache()`（:48-67）。
- `_build_flattened_bucket_chunks`(:70)：按 `max_bucket_bytes`（=`weight_broadcast_bucket_mb` MiB）切桶，**桶内按 dtype 分组**，产出 `{dtype, keys, shapes, numels}`。
- `_broadcast_metadata`(:123)：首次对该 communicator 发送 pickle 后的 metadata（大小 + 字节）；后续只发 `metadata_size=0`(:131-154)——与接收侧缓存闭环。
- `_broadcast_state_dict_flattened_buckets`(:157)：按 chunk：先发各 chunk metadata，逐 chunk `flat = cat(…)` → `comm.broadcast(flat, src=0)`。
- 公开流程：
  1. `prepare_weights_for_broadcast`(:297)：gather + 可选 CPU staging；
  2. `broadcast_weights_to_inference`(:348)：并行 `send_weights_to_inference`(:391，flattened_bucket 或 per_tensor 模式 `_broadcast_state_dict` :407)；
  3. 返回各服务器起止 `weight_sync_load_refs` 供编排器 `ray.get`。

## 3. 接收侧（`inference/worker.py`）

- `receive_state_dict`(:108)：若非 per-tensor，则：

```
metadata → 若 size==0 复用缓存（架构未变，:162-171）
          → 否则按 chunks 逐个：
             torch.empty(total_numel, dtype, device)
             comm.broadcast(flat, src=0)
             按 keys/shapes/numels 切片 view → yield (key, view)
```

- `load_weights`(:207)：用 **CUDA event 计时真实 GPU 传输**（非 enqueue 时间，:215-225），把 `receive_state_dict` 迭代喂给 `model.load_weights`。
- `collect_torch_memory_metrics`(:230)：采集 `torch_allocated_gb / torch_reserved_gb / torch_max_allocated_gb`，并把 `CUDA_VISIBLE_DEVICES` 映射为物理 GPU 索引、附 `tp_rank`。

## 4. 显存优化开关（Megatron 侧重点）

| 开关 | 效果 |
|------|------|
| `weight_broadcast_cpu_staging`（默认注意） | 14B 级模型整份 GPU state_dict ≈28GB 会超显存——先全量移 CPU 再广播（`_stage_state_dict_to_cpu`） |
| `weight_broadcast_pin_memory` | 钉页内存，NCCL 传输快 2-3x |
| `weight_broadcast_free_grad_buffers` | 广播期间释放 DDP 梯度 buffer ≈14GB（`_free_grad_buffers`/`_restore_grad_buffers` 配对，用后恢复） |

- Megatron 侧 `gather_weights_for_inference`(:2458) 需全部 rank 同步调用（内部 barrier），广播 root（rank0）实际发送。
- Megatron gather 优先 "bridge" 导出（模型无关）；失败回退手工 all-gather + `_convert_qwen_to_hf` 转换（Qwen 专用）。

## 5. 编排器协调

1. `_init_weight_broadcast`(orchestrator.py:2629)：`world_size = 1 + num_inference × tp_size`；单节点 loopback 优先。
2. 训练步完成 → `ray.get(weight_sync_load_refs)`（`asyncio.to_thread`）。
3. 起止时间写为 `weight_sync` InfraEvent（:2527-2541）。
4. 广播成功 → 推进 `inference_model_step`；group==`"full"` 同步推进 `eval_server_model_step`（:2543-2553）。

## 6. 健康探测

- `GET /collective_test`（server.py:70）：集体通信探测。
- `GET /load_weights?group=full|training_only`（server.py:88）：手动触发 worker `load_weights`（调试用）。