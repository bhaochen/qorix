# 配置系统 — 三层合并与参数族

> 完整参数见 `configs/defaults/default_train.yaml:1`(338 行全参数即文档);本篇讲**合并规则**与**参数族群**的取舍。启动→参数 → 每一条都落在一个参数族,别在 YAML 里乱塞。

## 1. 三层合并(utils/config_loader.py)

```
defaults/default_train.yaml ──┐
defaults/default_eval.yaml ───┼─ _deep_merge:40(后者覆盖前者)
run.yaml(用户)─────────────┘  └─ _coerce_value:54 schema 强转
                                    ↓
                          config.cfg(全局单例,冻结于 start_processes)
```

- **存活对象是 dataclass `QorixConfig`**(config_schema.py),YAML 只做载体;
- 合并是「写语义」:后出现的键覆盖前面的(数组整体替换,非按索引 merge);
- schema 强转失败会直接 raise——这是"默默漏参数"的守门员。

## 2. 参数族速查表

| 族 | 关键字段 | 一句话理解 |
|----|---------|-----------|
| 全局 | `project_name` / `run_id` | 决定 W&B 名称与目录 |
| 训练 | `train_steps` / `micro_batch_size` / `num_micro_batches` / `gradient_accumulation` | FSDP 微批循环的步进 |
| rollout | `max_rollout_length` / `group_size` / `num_rollout` / `max_tokens` | 一次组播采样吃多深 |
| 异步 | `max_async_rollout` / `max_off_policy_steps` | rollout 与训练的最大距离 |
| 权重 | `weight_sync_group`(`full`/`training_only`) / `stale_rollout_discard` | 谁能收到权重 / 老 rollout 处置 |
| 环境 | `env_name` / `env_config` / `multiturn` / `system_prompt` | 选哪种玩法 |
| 沙箱 | `sandbox` 段(`eager_prepare_resources`:49) | 训练前预热沙箱池 |
| 评测 | `eval` 段 / `eval_name` / `quantization` | eval server 何时醒来 |

## 3. 三个专项参数(训练效能的钥匙)

| 参数 | 位置 | 机制 |
|------|------|------|
| `eager_prepare_resources` | default_train.yaml:49 | 训练前预创建沙箱池,避免替代首轮 strawberry |
| `max_concurrent_prompts_per_server` / `prompts_batch_size_for_trainer` | default_train.yaml:39-40 | 单 server 并发上限与喂给 trainer 的批大小 |
| `discard_group_zero_advantage` | default_train.yaml:43 | 整组无学习信号(advantage 全同)→ 丢弃 |

## 4. 环境/评测配置形态(bridge 模式)

Eval 体系不读自己的一套 schema,而是「借壳」:
- eval 配置用 **EvalStandaloneConfig**(eval_standalone/config.py:22),但真正运行时通过 `_install_compat_config`(driver.py:175)拼一个 QorixConfig 的**兼容副本**,喂给既有 `EvalRunner / generate.py / inference/server.py`——这就解释了为什么 standalone eval 能复刻训练时的推理链路。
- 因此配置修改优先看 `utils/config_schema.py` 的字段,再谈加 eval。

## 5. 读配置的正确姿势

```bash
grep -n "" configs/defaults/default_train.yaml | sed -n '1,338p'   # 全参数手册
```

按「族 → me 」读:`graph_level` → `micro_batch` → `num_micro_batches`,`max_async_rollout` → `max_off_policy_steps` → `weight_sync_group` 是一个不可拆的关切模块。