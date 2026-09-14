# 独立评测 — eval_standalone 全流程

> 训练内 eval(运行中唤醒)见 `docs/evals/overview.md`。本篇讲**独立跑评测**:对一个已有 checkpoint 目录离线跑 pass@k,不搭训练集群。

## 1. 入口与数据流

```
python eval.py --config configs/examples/example_hendrycks_math_eval.yaml
   └─ eval_standalone/driver.py run_eval:827
        ├─ 解析 EvalStandaloneConfig(独立 config schema)
        ├─ _resolve_checkpoints:111 合并 显式 checkpoints + 自动发现
        ├─ _convert_checkpoint:140   (native→HF,按需)
        ├─ _install_compat_config:175 (造一个 QorixConfig 兼容副本)
        ├─ 起 vLLM server(独立实例 worker)
        ├─ 每步:整个数据集 × pass@k completions
        └─ 汇总 → pa.Table → W&B / 本地 dor 帧
```

## 2. checkpoints 解析:两种来源一个 merge

| 标识 | 类型 | 判定 |
|------|------|------|
| HF 可加载 | `step_N/config.json` | `hf_weights=true` 时 `_discover_checkpoints`:85 |
| 训练原生 | `step_N/meta.json` | `hf_weights=false`(native DCP) |

merge 语义(_resolve_checkpoints:111):**显式 `checkpoints` 覆盖自动发现的同 step**(`by_step[step] = entry`,显式优先)。文件顺序 = step 升序 → 一个 eval 文件可以覆盖多步权重(曲线)。

## 3. 转换桥复用能力

`_convert_checkpoint`:140 复用 `checkpoint_converter.convert_single`:
- **缓存**:输出目录已含 `config.json` → 直接复用,不重转(断点续跑友好);
- **失败清理**:转换抛异常则 `rmtree` 删除半成品,避免下次误复用(driver.py:156-159);
- eval 结束时 `_cleanup_converted`(:164)清掉临时转换结果。

## 4. pass@k 与世界真相

driver.py 里 `pass_k` 的两种配置形态(`at_k` 整数列表 / `pow_k` 幂次):
- 每个 sample × 每个 eval checkpoint × `max(k)` 次 completion(completion_idx 0..max(pass_k)-1,见 `EvalGenerationRollout`:866);
- 同环境、同奖励函数(`metrics`),只是 completion 数不同;
- 聚合:`EvalGenerationRollout.eval_metrics` + `golden_answers` 对齐 → 每 k 算 pass@k(`correct/总 completions`,样本级 OR)。

## 5. 兼容配置「借壳」带来的必然性

`_install_compat_config`:175 是理解 standalone eval 的钥匙:EvalRunner/generate.py/server.py 都读全局 `config.cfg`。它拼一个 QorixConfig,字段白名单(推理参数/sampling/ray),把 eval 需要的字段**填进既有结构,而不是让 eval 引入第二套推理路径**。推论:
- standalone eval 与训练时 eval **共用同一个推理 server 代码**(inference/server.py),bug 一处修两处好;
- eval 配置字段 ≈ 训练共享字段的子集,别期待 eval 独有 set 更大。

## 6. 度量输出

- W&B 表:`eval/.../step/N` 下,`correct` / `pass@k` / `sample_metrics` 全量 rows;
- 本地:events 帧(`tail_idx`)落 dor 文件(同训练链路);
- golden answers:从 eval 自身逻辑(如 math500 的 `golden_answers` 提取)得出,不依赖训练时的 `golden_answers` 字段——见 evals/base.py `metrics` 契约。

## 7. 什么时候用独立评测,什么时候用训练内 eval

| 场景 | 选型 |
|------|------|
| 训练中途看"这步权重量级如何" | 训练内 eval server(同步,step 对齐) |
| 已训完,测多个 step 的曲线 | standalone(多 checkpoint 一次跑完) |
| HF 权重(已有 config.json)发布前验证 | standalone + `hf_weights=true` |
| 只测一个 checkpoint 且懒 | standalone(自动发现 step_N) |