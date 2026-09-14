# 环境系统

> 环境 = 训练任务的具体定义：数据集、prompt 编译、奖励函数。**放入 `src/qorix/environments/<name>/` 即自动发现**，无需注册。内置 7 个环境 + 可插拔沙箱（代码执行）+ ToolEnvironment 基类。

## 1. 自动发现（`environments/registry.py`）

- `_find_env_class`(:41)：遍历 `environments/<name>/environment.py` 找**唯一非抽象具体 `Environment` 子类**；`_CLASS_CACHE`(:158) 缓存类。
- `_read_packages`(:57)：**AST 解析** `REQUIRED_PACKAGES` / `OPTIONAL_PACKAGES`（不 import，缺依赖也能诊断）。
- `_check_required_packages`(:83)：import 前预检，缺依赖打印 `uv add …` 提示（:102、:107-111、:134-144）。
- API：`get_environment(name, **kwargs)`(:158)、`list_environments`(:175)、`check_environments`(:187，打印 OK/MISSING 表)。
- `__init__.py` 导出全家桶：`Environment`/`SingleTurnEnvironment`/`MultiTurnEnvironment`/`ToolEnvironment`、`Sample`、`RewardResult`、`EvalMetricsResult`、`TrajectoryStep`、`RolloutState`、`ChatMessage`/`Messages`、`ToolCall`/`ToolResult`、`func_to_tool_schema`、`parse_xml_tool_calls`、`parse_function_call_syntax`、答案解析器（extract_boxed_answer/…/strip_think_tags）、`Rubric`、`get_environment` 等。

## 2. 基类与生命周期（`environments/base.py`）

数据契约：
- `ChatMessage` / `Messages`(:22-23)；`Sample`(:27)：`{prompt: str|list[ChatMessage], answer, metadata}`。
- `RewardResult`(:35)：`{total_reward, sample_metrics, golden_answers, info_turns(list of {turn_order, info_key, info_value, info_type}), sample_tags}`。

生命周期：`load_dataset` → `compile_prompt` → `compute_reward(sample, completion, …)` → `RewardResult`；多轮含 `next_turn(history)`。

## 3. 奖励组合（`environments/rewards.py`）

- `maybe_await`(:36)：统一同步/异步奖励函数。
- `_build_call_kwargs`(:48)：签名内省，按参名传递；函数接受 `**kwargs` 则透传所有非 None 参数。
- `Rubric`：组合多路 sync/async/加权/metric-only 奖励并合并为 `RewardResult`。
- 归一化：环境入口 `reward_min`/`reward_max`（config_schema）逐环境归一（详见 config/overview.md）。

## 4. 内置环境

| 环境 | 目录 | 类型 | 要点 |
|------|------|------|------|
| Countdown | `countdown/` | 单轮 | `SYSTEM_PROMPT`/`INSTRUCTION_PROMPT`(:21/:26)，数据集 `Jiayi-Pan/Countdown-Tasks-3to4`，`metrics_ranges`(:62)；`reward_format`(:124)/`reward_equation`(:128) 经 Rubric 组合 |
| Wordle | `wordle/` | **多轮** | 交互式猜词游戏（多轮 rollout） |
| Hendrycks Math | `hendrycks_math/` | 单轮 | 竞赛级数学（需 `math-verify`） |
| DAPO Math | `dapo_math/` | 单轮 | DAPO 风格数学 |
| DeepScaleR | `deepscaler/` | 单轮 | DeepSeek 风格长推理 |
| DeepDive | `deepdive/` | agentic | `web_tools.py` 网页检索、`open_one.py`、`rate_limit.py`、`formatting.py`、`config.py` |
| I3 Code | `i3_code/` | 单轮+沙箱 | 代码生成 + 沙箱测试；`verification_utils.py`、`deepcoder_utils.py`、`sandbox_utils.py` |

## 5. 沙箱抽象（`environments/_sandbox/`）

代码执行环境多提供商：
| 文件 | 提供方 |
|------|--------|
| `prime.py`(:306) | Prime Sandbox |
| `modal_provider.py`(:254) | Modal |
| `daytona.py`(:205) | Daytona |
| `e2b_provider.py`(:251) | E2B |
| `base.py`(:152) | 抽象基类 |
| `pool.py`(:388) | 池化：惰性初始化/复用 |

## 6. Tool 环境（`environments/tool_env.py`，829 行）

- `ToolEnvironment` / `ToolCall` / `ToolResult`。
- `func_to_tool_schema`：Python 函数 → JSON schema（供模型声明工具）。
- 解析：XML 工具调用（`parse_xml_tool_calls`）与函数式调用（`parse_function_call_syntax`）双格式。
- `parsers.py`(169 行)：纯函数答案提取（boxed/answer 标签/代码块/XML tag/think 剥离）。

## 7. 新增环境的 4 步

1. 建目录 `src/qorix/environments/<name>/`，写 `environment.py`（继承 `SingleTurnEnvironment`/`MultiTurnEnvironment`）。
2. 定义 `REQUIRED_PACKAGES` / `OPTIONAL_PACKAGES`（AST 预检自动生效）。
3. `configs/examples/*.yaml`（或 CLI）配置 `environments: [{name, weight, reward_min, reward_max}]`。
4. `check_environments()` 或以 `example_i3_code.yaml` 为样板跑通。

> **注意**：奖励函数以 `Rubric` 组合、同步/异步皆可；多轮环境走 interleaved rollout 路径（`generate.py:run_multiturn_rollout`），本地 tokenize 保证前缀精确匹配。