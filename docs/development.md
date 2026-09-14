# 开发指南

> 面向新贡献者的最小可运行知识集：本地跑通训练/评测、新增环境与评测、Docker 构建、调试。

## 快速开始

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv sync                        # 安装依赖（含 flash-attn/torch/vllm）
wandb login                    # W&B（metrics + UI 数据源）
uv run train.py --config configs/examples/example_countdown.yaml
```

- 需要 NVIDIA GPU；Python 3.11+；uv。Docker 用户直接 `docker pull ghcr.io/bhaochen/qorix:latest`。
- 参数覆盖：`uv run train.py --config ... --learning_rate 5e-7`。
- 评测：`uv run eval.py --config configs/evals/eval.yaml`（需 `wandb_run_path` + `evals` 列表）。
- 依赖提示：math 用 `uv add math-verify`；M-wordle 用 `uv add textarena`；code 沙箱用 `uv add daytona-sdk`。

## 三层配置速查

- 默认值全参数参考：`configs/defaults/default_train.yaml`。
- 覆盖顺序：defaults → `--config`（缺省 `configs/run.yaml`）→ CLI。列表整体替换（`utils/config_loader.py:40`）。
- CLI 值尽力按目标类型转换（`_coerce_value` :54）。

## 新增环境（4 步，见 environments/overview.md）

```bash
mkdir src/qorix/environments/<name>
# environment.py: from qorix.environments import *
# class MyEnv(SingleTurnEnvironment): load_dataset / compile_prompt / compute_reward
```

1. 继承对应基类，`REQUIRED_PACKAGES` 声明依赖（AST 预检）。
2. 奖励用 `Rubric` 组合（同步/异步皆可）。
3. 配置 `environments: [{name, weight, reward_min, reward_max}]`。
4. `check_environments()` 验证；多轮环境按 interleaved rollout 约束（本地 tokenize）。

## 新增评测目标（2 步）

1. `src/qorix/evals/<name>/eval.py`：继承 `Eval`（包裹环境）或完全独立产 `EvalMetricsResult`。
2. 在 `configs/defaults/default_eval.yaml` 或 eval.yaml 的 `evals:` 列表引用。

## Docker

- 应用镜像：`docker/Dockerfile`（base 选用 `ghcr.io/eduardoslonski/telescope-base:latest`，复用 vLLM/torch 环境；`uv sync --frozen --no-dev`）。
- base 镜像构建：`docker/Dockerfile.base`（`vllm/vllm-openai:v0.17.0`；Apex/TE 源码构建；`setuptools>=77.0.3,<80` 锁版）。
- 在云 GPU 平台（Vast.ai/RunPod）用镜像 `ghcr.io/bhaochen/qorix:latest` 作模板。

## 调试

- **终端日志**：`logs/{orchestrator,trainer,inference}/{comp}.log` + `.detailed.log`（tlog 自动建立）。
- **W&B/UI**：训练自动打 `qorix` tag；`pip install qorix-ui && qorix` 开 dashboard（localhost:8005）。
- **权重网络**：`GET /collective_test` / `GET /load_weights`（inference server 控制路由）。
- **checkpoint→HF**：`uv run python tools/convert_checkpoint_to_hf.py ...`（FSDP/Megatron → safetensors）。

## 代码结构速查（34K LOC / 82 文件）

```
train.py/eval.py                   入口
src/qorix/
  orchestrator/   主循环/调度/组批/日志   （orchestrator.py 3373 行自证为最大单文件）
  inference/      vLLM server + NCCL worker
  trainer/        FSDP2/Megatron + weight_sync + loss
  environments/   环境（自动发现）
  evals/          评测目标（自动发现）
  eval_standalone/ 离线评测 driver
  utils/          config / ray_runtime / tlog / checkpoint_converter
configs/          defaults + examples + run.yaml
docker/           Dockerfile(.base)
tools/            转换脚本
```

## 检查点与恢复

- `checkpoint_dir/step_N/`：原生 `meta.json`；HF 转后含 `config.json`。
- `resume_from_checkpoint` 可从任意 checkpoint 恢复；`checkpoint_save_training_state` 决定是否保存优化器/RNG 状态。
- 配置 `checkpoint_keep_last/every` 控制保留策略。