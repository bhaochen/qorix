# Qorix 文档

> Qorix — 面向推理与 Agent 的 LLM 后训练的强化学习(RL)框架。编排 (orchestrator) × 推理 (inference) × 训练 (trainer) 三组件在 Ray 集群上异步协同。

## 快速开始

```bash
uv run train.py --config configs/examples/example_countdown.yaml   # 训练
uv run eval.py --config configs/evals/eval.yaml                    # 独立评测
uv run python tools/convert_checkpoint_to_hf.py ...                # 转 HF 格式
```

## 目录

### 架构
| 文档 | 说明 |
|------|------|
| [系统架构概览](architecture/overview.md) | 三组件职责、Ray 资源模型、启动/关停流程、模块职责表（34K LOC / 82 文件） |
| [模块地图与符号索引](architecture/module-map.md) | 功能 → 源码符号:行号直达表;缩略语速查(TP/FSDP/TIS/ICE-POP…) |
| [设计哲学](architecture/design-philosophy.md) | 七条不成文约定:单样本优先、组槽位、同步线程、全局单例、off-policy、事件帧、先兼容 |

### Orchestrator
| 文档 | 说明 |
|------|------|
| [异步训练数据流](orchestrator/data-flow.md) | Prompt 预取 → 推理 → 奖励 → 优势 → 训练 → 权重广播 完整链路、采样/异步/车道、off-policy 取消 |
| [主循环与派单](orchestrator/main-loop.md) | run() 三协程、组/个体派单路径、权重追权重的 commit 顺序、故障重试哲学 |

### 配置
| 文档 | 说明 |
|------|------|
| [配置系统](config/overview.md) | 三层配置（defaults → run → CLI）、QorixConfig Schema 全字段分组 |
| [参数族参考](config/parameter-reference.md) | 三层合并规则、参数族群、专项参数(eager_prepare / local_micro_batch / stale_rollout) |

### Trainer
| 文档 | 说明 |
|------|------|
| [训练器深度解析](trainer/overview.md) | FSDP2 / Megatron 后端、微批打包、gradient sync 优化、上下文并行 |
| [RL 算法与损失](trainer/rl-algorithms.md) | 7 算法（GRPO/RLOO/REINFORCE++/DR-GRPO/CISPO/GSPO/SAPO）+ PPO Clip/TIS/KL/熵修饰 |
| [权重广播协议](trainer/weight-broadcast.md) | NCCL 双通道（full/training_only）、flattened_bucket_v1、CPU staging、显存优化 |
| [检查点](trainer/checkpointing.md) | 保存(DCP 分片+伪原子 rename)、目录规约、转 HF 三坑、恢复 |

### Inference
| 文档 | 说明 |
|------|------|
| [推理服务器](inference/overview.md) | vLLM server、NCCLWeightUpdateWorker、控制路由、torch_memory、部署拓扑 |
| [请求契约](inference/request-contract.md) | 线程池 POST、请求体、动态 token 预算、错误归一、InterleavedTokenizer、控制路由 |

### Environments
| 文档 | 说明 |
|------|------|
| [环境系统](environments/overview.md) | 自动发现、基类生命周期、Reward Rubric、ToolEnvironment、沙箱 Providers |
| [奖励设计](environments/reward-design.md) | Rubric 组合引擎、签名内省、同步/异步透明、golden answer 优先级、metrics vs reward |
| [工具调用](environments/tool-calling.md) | ToolCall/ToolResult、五个必须实现的方法、回合循环、工具三态 |
| [沙箱](environments/sandbox.md) | SandboxProvider ABC、生命周期五个阶段、池化与预热、自建 provider 契约 |

### Evals
| 文档 | 说明 |
|------|------|
| [评测系统](evals/overview.md) | Eval 注册、内置基准、独立 eval driver、Arrow 表上传 |
| [独立评测](evals/eval-standalone.md) | checkpoint 解析(hf/native)、转换桥、pass@k、compat config 借壳、适用场景 |

### 可观测性
| 文档 | 说明 |
|------|------|
| [事件日志与 W&B](observability/overview.md) | EventLogger、section/group/metric 三层命名、tail 窗口、Arrow+zstd 上传 |
| [事件体系](observability/event-schema.md) | 四层事件金字塔、Event/GenerationRollout/RolloutEvent 字段表、tail 帧与 block、查询指南 |
| [监控与可视化](observability/monitoring.md) | 四个观测出口、W&B 三层命名、三类暂停信号、铁手诊断三问 |

### 开发
| 文档 | 说明 |
|------|------|
| [开发指南](development.md) | 本地运行、新增环境/评测、Docker 构建、调试 |

### 面试
| 文档 | 说明 |
|------|------|
| [面试准备](interview-prep.md) | 15 道深度问答:并发模型、off-policy 代价、错误处理、token 一致性、检查点转换 |

> **学习路径 (45min)**：`architecture/overview.md` → `orchestrator/data-flow.md` → `orchestrator/main-loop.md` → `trainer/weight-broadcast.md` → `config/parameter-reference.md` → `observability/monitoring.md` → 用 `example_countdown.yaml` 跑通首个训练 → 时间充裕再读 `interview-prep.md`。