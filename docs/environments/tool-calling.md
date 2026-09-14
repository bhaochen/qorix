# 工具调用 — ToolCall / ToolResult / 执行循环

> 想让模型学会「观察→调用工具→结果回喂」,核心就是 `ToolEnvironment`。它继承 `MultiTurnEnvironment`,把"回合"改写成「生成 → parse → execute → format → 回喂」的循环。

## 1. 两个核心数据结构

`ToolCall`(tool_env.py:158)与 `ToolResult`(:165):前者是「模型想干的活」,后者是「干完的结果」。

| 字段(节选) | ToolCall | ToolResult |
|------------|----------|-----------|
| 身份 | `tool_name` / `arguments`(原始 str 或 JSON) / `raw_text` | `tool_name` / `content` |
| 状态 | — | `success: bool` / `exit_code` / `error` / `truncated` |
| 追踪 | `tool_call_idx` | `trace_id` / `duration_ms` |

一个 `ToolResult` 记录被 `format_tool_result`:624 染成文本段(带 `<tool_response>` 边界),回喂进对话历史——这就是 `ToolCallRecord`(event_logger.py:711)里的 `env_response_generation_idx` 语义:生成 N 的调用,其结果在约 N+1 轮出现在对话里。

## 2. 环境要实现的五件事

| 方法 | 职责 | 位置 |
|------|------|------|
| `parse_tool_calls(completion)` | 从模型文本里抠出结构化 `list[ToolCall]` | :529 |
| `is_final_answer(completion, state)` | 判断这轮是不是终结回答(结束循环) | :551 |
| `execute_tool(tool_call)` | 真正的工具执行(沙箱 / HTTP / 内存 mock) | :572 |
| `env_response(...)` | 生成工具结果 → 组装进 messages | :673 |
| `is_done(state)` | rollout 阶段终结判定:`(done, reason)` | :756 |

另外两个挂钩脚:
- `get_tool_metrics(state)`:把工具使用统计(次数/成功率)带进 `sample_metrics`(:768);
- `_build_tools_prompt_section`:474 生成工具 schema 描述块,挂在 system prompt 上;
- `_default_error_formatter`:470 兜底错误文案(超时/OOM 的宽容呈现,给模型 next attempt)。

## 3. 回合循环(single-turn 内部复用)

```
ToolEnvironment.env_response
  └─ 完成文本中 parse_tool_calls
       └─ 若 is_final_answer → 返回 (final_answer, done)
       └─ 否则 execute_tool 逐条(异步并发? 可串行)
            └─ format_tool_result → messages 追加 tool_response
            └─ 返回 (prompt_tokens, new_messages, ...)
```

注意「模型要不要再走一轮」由 `is_done` / `is_final_answer` 决定,不是「还有没有工具结果」。这也是别名的来源:**早停**发生在 is_final_answer 而非 execute。

## 4. 工具列表三态

工具环境里 `self.tools` 决定「模型能调用什么」:

- **mock(内存函数)**:单测 / 无沙箱时(sample 培训);
- **真实执行**:通过 `execute_tool` 里对 `SandboxProvider` 的调用执行任意命令(需 `sandbox` 配置段);
- **禁用**:`ToolEnvironment` 默认不注入工具,或者工具的 schema 为空 → 模型只能输出 final_answer。

## 5. 时间线归属

一次工具执行在 UI 上是一个 `RolloutEvent(event_type="tool_execution", phase=start/end)`(event_logger.py:826),lane 槽位串在一起;**权重更新不打断**——execute 跑在 sandbox,`sample_off_policy_steps` 照样累计,tool 执行受 `max_off_policy_steps` 管辖。

## 6. 代码库中的示例

写一个新工具环境最少拖拽三块:ToolCall/ToolResult 子类、`execute_tool` 的实现、`get_tool_metrics`。
参照 `test_*env` / `example_i3_code.yaml` 的 `env_config` 写法即可落地。