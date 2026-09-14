# 奖励设计 — Rubric 组合引擎

> 不是"调一个奖励函数",而是"组合多个奖励函数并按权重合出 `total_reward`"。两套 FAQ:同步/异步混用、异常兜底、golden answer 传播、metrics vs rewards。

## 1. 数据结构(rubrics 的世界)

| 类型 | 说明 |
|------|------|
| `RewardResult` | 解析一条 sample 的最终产物:`total_reward` + `sample_metrics` + `golden_answers` + `info_turns` + `sample_tags`(environments/base.py:35) |
| `_RewardEntry` | 注册时的一次快照:`{func, weight, name, golden_answer, range_min, range_max, invert}`(rewards.py:74) |
| `Rubric` | 组合器,`add_reward(func, weight=..., ...)` 链式注册(rewards.py:89) |

## 2. 调用与拆包

`score()`(rewards.py:174)的全部逻辑就三小步:

```python
for entry in self._entries:
    call_kwargs = _build_call_kwargs(entry.func, available)   # 按签名只给它们声明要的参数
    raw = await maybe_await(entry.func, **call_kwargs)        # 同步/异步透明
    # 拆包: (score, golden_answer) 元组 或 裸 float
    total_reward += score_val * entry.weight
    sample_metrics[entry.name] = score_val
```

三大设计点:

1. **签名内省 `_build_call_kwargs`:48**——奖励函数声明什么参数就给什么(`completion`/`sample`/`state`/`eos_token`),不声明就不塞。天然支持"有的函数只看 completion,有的要看 state"。`**kwargs` 的函数拿全部非 None 值。
2. **同步/异步透明 `maybe_await`:36**——函数返回 coroutine 就 await,否则直接取结果,奖励作者不用关心框架在 async 环境里。
3. **异常兜底**——奖励函数抛异常,log warning 并记 `0.0`(:205-211)。**这是 feature 不是 bug**:单一奖励函数炸了不拖死整条 rollout。

## 3. 权重与 golden answer 真相

- `weight` 乘在 score 上(`score_val * entry.weight`),**不是**缩放进入回调:负权重即「惩罚项」;
- golden answer 三来源优先级(rewards.py:224-228):
  1. 函数返回值里的元组(`(score, golden_answer)`)最高;
  2. 注册时显式 `golden_answer=` 其次;
  3. 否则不记录。
- `metrics_ranges` 属性(:156)自动从 `range_min/range_max/invert` 生成——这是给 W&B `metrics_ranges` 表用的,「metric 是 weight=0 的 reward,追踪不改总分」(`add_metric`:133)。

## 4. 常见玩法对照

| 想要 | 用什么 |
|------|--------|
| 纯精确匹配 | 一个 `exact_match` 函数 + weight=1 |
| 主干奖励 + 形状奖励 | 两个函数,主 weight 大于从 |
| 只观测,不改梯度 | `add_metric`(weight=0) |
| 反向(越短越好/越少工具越优) | 返回负 score,或 `invert=True` |
| answer 进 meta 表 | 函数返回 `(score, answer_text)` 元组 |

## 5. 坑位提醒

- 奖励函数的 `completion` 是**纯文本**(或 tokens),别指望拿到 logits——那是 loss 层的职权;
- `score` 被调用多次(多轮 interleaved 每轮一 score),`_RewardEntry` 是**注册时冻结**,数据结构里改 weight 不会在每个轮次间生效;
- `golden_answers` 是给 eval/eval_standalone 比对的,训练侧不打它即可(或打了也无妨)。