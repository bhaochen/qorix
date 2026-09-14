# 设计哲学

理解这套代码,先接受它的七个不成文约定。文档里的每一个实现细节,几乎都能从这七条推出。

## 1. 单样本优先,环境决定多轮

`base.py Sample:27` 就是整个框架对"一条数据"的定义:**一个 system prompt + 一个用户 prompt + 一个 `Response` 槽**。翻成白话:

- 训练单元永远是一条 `Sample`,它可以在环境里被多轮地调用模型(interleaved rollout),也可以被环境"改写后再掷回"(`detour` / feedback),但它的**身份**不会变——`sample_id` 全程不变。
- 环境换多轮(如 wordle:表驱动角色),只是换 `rollout_loop` 的实现 + 一组专门的 `agent` API,框架的组调度/重放/评测一点不用改。

推论:要给框架加一种"新玩法",第一问是"它还是一份 Sample 吗";是 → 改环境;不是 → 改核心。

## 2. 组(group)是铁饭碗,不存在的样本填 `None`

调度层面没有任何"单样本"概念,**连个体模式 `_start_next_individual`:1064 也是以 `group_size` 为一组**。组的作用:

- 一次 HTTP 请求拍一组 `k` 个 prompt(vLLM 合并),对同一权重版本采样——天然保证**组内同权重**;
- 组内的 lane 槽位预分配(`lane_slots`),时间线上可以平铺对齐;
- 掉队的样本(HTTP 超时/梦游 token)不会拖垮整组——它的槽位留空,组照常聚合:见 `_assemble_single_turn_group`:1651 里对 `None` 的防御式处理。

推论:看到 `for i in range(group_size)` 一律先想"这里是槽位,不是样本"。

## 3. 网络库的异步是骗人的,同步线程才是真理

代码反复出现同一种注释「Performs synchronous HTTP request off an async context loop」(generate.py:42):`asyncio` 主循环 + 线程池(`executor=ThreadPoolExecutor(max_workers=2048)`)真同步阻塞。原因很朴素:

- 框架的主干是一条**事件循环**,不能被任何 `await aiohttp.post()` 的响应式 IO 卡住;
- 真正吃时间的是 rollout 循环与训练循环两个协程,它们天然该跑线程之外;
- 2048 这个数字就是"并发上限约等于没人会怀疑代数的安全垫"。

推论:别在 `_rollout_loop` 里自己写异步请求,同步线程池是唯一被支持的心跳。

## 4. 全局单例配置,读多写少,写必有卫兵

`config.cfg`(utils/config.py)是全局单例,**启动后基本只读**。改动只允许集中在启动前的三层合并(config_loader):
`defaults/default_train.yaml` → `defaults/default_eval.yaml` → 用户 `run.yaml`(后者覆盖前者,`_deep_merge`:40)。

卫兵写在两个地方:
- `start_processes`(orchestrator.py:281)启动后再也不准改配置:Batch 组成(`prompts_batch_size_for_trainer` × `num_micro_batches`)、采样参数、组大小全部在此刻结算;
- `_update_inference_weights`:2479 之后的回路里,配置只读、唯一可变的是 `_state(_deferred_dispatch)/_dispatch_task` 这类运行时水位。

推论:报 bug 先看 config_loader,大概率是某项参数在 `_coerce_value`:54 被强转了类型。

## 5. off-policy 是不可回避的日常,不是异常路径

任何分布式 RL 训练,权重更新与 rollout 严格同步都太贵。框架的默认答案:**接受策略滞后,并把它量化**。

- 每个 inflight 组记录 `sample_off_policy_steps`(orchestrator.py:2555):权重更新一次,未结束的样本计数 +1;
- 超过 `max_off_policy_steps` 的组被**取消**(权衡:等它结束不如新开一组);
- 掩码技术(`dapo_mask` 等)让奖励信号的 IS 权重不炸。

推论:看见 `max_async_rollout` / `max_off_policy_steps`,翻译成「允许模型落后几版权重」。

## 6. 事件与指标封成周期帧(tail window),不是每行一条

真正的观测面是一个`帧`队列:`tail_idx` 每上传周期 +1,trainer/orchestrator/eval 各自事件全部归档到同一 `EventLogger`(只用 `pub/sub`)。模型端(训练侧)与推理侧(wandb)不在同一条链路上:它们只共享事件 schema,不同上传周期。

推论:读 `event_logger.py` 的 `Event:656` 时,`tail_idx` 不是"时间戳序号",而是「这份数据属于第几次上传帧」。

## 7. 先跑通最小路径,再加密特殊性

框架量子态:注册系统(auto-discover)在处理"用户带了 eval 段、沙箱池、Megatron"前,先处理"只给一个 model + vLLM worker"的最小路径。这一点在 Eval 的兼容配置 `_install_compat_config`:175 里写得最明白:

> 「Instead of refactoring them, we build a compat config that exposes the fields they need.」

也就是:**优先兼容,而非重构**。给现有模块接新能力时,先在配置层补齐字段,再谈动数据结构。

## 下一步阅读

- 想看调度闭环:主循环 `run()` orchestrator.py:613 → 派单 `_start_next_prompt`:996 → 组聚合 `_assemble_single_turn_group`:1651。
- 想看权重如何追上来:watcher `_weight_update_watcher`:776 → CPU staging `weight_sync.py:38` → 接收 `worker.py:108`。
- 想看什么是"可评测的最小框架":eval_standalone 的兼容配置 `driver.py:175`。