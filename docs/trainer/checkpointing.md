# 检查点 — 保存、转换与恢复

> 训练检查点是 DCP 分片;评测/部署检查点是 HF safetensors。两者之间有一条"随时可补"的转换桥,详见本篇。

## 1. 训练侧保存(FSDP 后端)

`save_checkpoint`(trainer/backends/fsdp.py:594),三种可组合内容:

| 内容 | 开关 | 说明 |
|------|------|------|
| model + optimizer + scheduler | `checkpoint_save_training_state` == True | `get_state_dict(cpu_offload=True)`,DCP 分片到各 rank,带 optimizer/RNG 以支持**无缝 resume** |
| 仅 model | 开关关闭 | 只存模型权重(文件更小,不能精确实resume optimizer) |
| HF 元数据 | 恒定 | `hf_meta/` 里 `config.json` + `tokenizer`,供离线转换;根目录 `meta.json` 记 `{base_model, step, backend}` |

可靠性设计,值得照抄:
- **分步写 `step_N_tmp` + 伪原子 rename**:所有 rank 写完 → `dist.barrier()` → rank0 `rmtmp rename tmp→final`(fsdp.py:686-693);中途崩溃留下的是干净目录,不会读到半个检查点;
- **dcp_save 重试 3 次**:失败则清掉 tmp 重来(fsdp.py:640-660);
- **per-rank RNG state**:`rng_state_rank_{RANK}.pt`,保证 resume 后数据序可复现(:681-684)。

作业排布:何时存、存哪一步,由 `checkpoint_pending` 状态与 orchestrator 的 `_save_batch_async`:2337 下游决定——存储是"背靠背"在权重更新之后。

## 2. 目录规约

```
ckpt_root/
  step_0500/              # 训练检查点(DCP)
    meta.json
    hf_meta/{config.json,tokenizer.json,...}
    *.distcp             # DCP shards
    optimizer/  scheduler/ 或 model/     # 视 save_training_state 而定
    rng_state_rank_{RANK}.pt            # 仅 save_training_state
```

`step_%(...04)d` 零填充格式(正则 `_STEP_RE`,driver.py:80),被 eval_standalone 用 `step_N` 模式扫描。

## 3. 评测/部署前的转换(不做也不行)

转换器 `utils/checkpoint_converter.py` `convert_single`:374,**输入**训练 DCP,**输出** HF safetensors。处理的三个"坑":

| 坑 | 解法 | 位置 |
|----|------|------|
| FSDP 的 `model.` 前缀 | strip 前缀,逐个 load shard | `_convert_fsdp`:62 |
| Megatron `gate_up` 交错 | `_deinterleave_tp_gated_mlp`:156;TP>1 时 DCP `no_dist` 重建出 `[g0;u0;g1;u1...]` → chunk 重排 | :156 |
| TP 尺寸探测 | `_detect_tp_size`:122:`meta.tp_size` 优先,否则数 column-parallel 张量(word_embeddings/linear_qkv)的 chunk 数 | :122 |

真实使用点:eval_standalone 在 `_convert_checkpoint`(driver.py:140)里**按需转换并缓存**:如果 `output/config.json` 已存在则直接复用(断点续转),转换失败则 `rmtree` 清理防止脏产出( :156-159)。

## 4. 恢复(load_checkpoint)

`load_checkpoint`(fsdp.py:699)是 `save_checkpoint` 的逆操作:
1. 用当前模型结构构造空 state_dict(optimizer 也构造);
2. `dcp_load` 全部分片;
3. `set_state_dict` 写回 model/optimizer(`cpu_offload=True`,安全);
4. 恢复 scheduler(若有);RNG 按 rank 读回。

注意:**Meta 结构必须与保存时一致**才能正确 load——`save_training_state` 开关改了,老检查点可能无法原样恢复 optimizer。

## 5. 何时用 HF 转换后的检查点

- 独立评测(eval_standalone):`hf_weights=true` 时直接从 `step_N/config.json` 扫描;
- 发布/微调起点:转换器跑完即可用 `transformers` 加载;
- CI 镜像构建只需要 base 镜像,不需要检查点进镜像。

## 6. 检查点健康检查清单

- [ ] `step_N/meta.json` 的 `backend` 与本次训练后端一致;
- [ ] `mkdir_barrier` -> rename 是无损的(观察 target 目录无 `.tmp` 残留);
- [ ] `save_training_state` 与 resume 场景匹配;
- [ ] 恢复后对比 `train_loss` 曲线无明显跳变(resume 复现性);