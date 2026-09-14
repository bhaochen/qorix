# 沙箱 — SandboxProvider 抽象与池化

> 沙箱解决:「模型的任意命令执行」不能发生在训练进程里。三个 cloud provider(Prime / Modal / Daytona)共享同一 ABC,环境代码**只依赖这份接口**。

## 1. 接口三件套(_sandbox/base.py)

| 类型 | 字段 | 角色 |
|------|------|------|
| `SandboxHandle:19` | `id` + `provider_data` | `create` 返回的不透明句柄,`provider_data` 存原生对象(modal.Sandbox 等)供 provider 专用操作 |
| `SandboxConfig:32` | `image / cpu / memory_mb / disk_size_gb / gpu_count / timeout_seconds / environment_vars / extra` | provider 无关的创建配置;`extra` 透传 provider 特有参数(team_id、labels、start_command…) |
| `ExecResult:48` | `exit_code / stdout / stderr` | 命令结果 |

错误层次(`SandboxError` 基类 + 三个子类):
```
SandboxError
 ├─ SandboxCommandTimeoutError  :62  命令超时(带 command/timeout/sandbox_id)
 ├─ SandboxOOMError             :75  内存耗尽
 └─ SandboxNotRunningError      :79  沙箱已停(终止/过期)
```

## 2. 生命周期(from base.py 文档字符串 diff)

```
create      → SandboxHandle
setup       → 注入 image 与配置,冷启动依赖(缓存/软件装好)
ready       → 完成后进 ready 队列(可执行命令)
execute     → ExecResult(exec 单条命令)
destroy     → 回收(SandboxNotRunningError 后强制销毁)
```

`InfraEvent(event_type="sandbox", phase=create/setup/ready/execute/destroy)`(event_logger.py:838)完整记录这五个阶段——这是「沙箱为何慢」的定位工具。

## 3. 池化与预热

`_sandbox/pool.py:388` 的池负责容量:
- `eager_prepare_resources`(default_train.yaml:49)=\"训练前就建好池\",避免 rollout 第一波全卡在 create (巨大 ttft);
- 池按 `SandboxConfig` 的一等键哈希(等幂),同配置复用;
- 释放/驱逐交给 LRU / destroy 策略,与 `timeout_seconds` 联动(超时即摧毁防泄漏)。

## 4. 错误如何变成「给模型的反馈」

`SandboxCommandTimeoutError` 等异常在工具层被 catch → `_default_error_formatter`:470 归一成一句话(如「命令超时(120s),请重试或简化」),作为 `tool_result` 文本回喂 —— 模型据此决定 next attempt。**框架不负责告诉模型"别乱跑"**——那是 format_tool_result 的呈现层职责。

## 5. 自建 provider 的最小契约

```python
class MyProvider(SandboxProvider):
    async def create(self, config: SandboxConfig, name: str = "") -> SandboxHandle: ...
    async def execute(self, handle, command, timeout_s) -> ExecResult: ...
    async def destroy(self, handle): ...
    # (setup / ready 可选,默认空实现)
```

注册后:env 只需 `SandboxProvider.create(SandboxConfig(image=..., cpu=..., memory_mb=...))` 即可,'环境代码永不 touch provider 私有字段'。

## 6. 常见故障速查

| 症状 | 检查 |
|------|------|
| 首轮全 404/503 | eager_prepare_resources 未开,或池 get 到错 image |
| 命令行为诡异 | `environment_vars` 未透传(provider `extra` 里带错 team_id) |
| 沙箱泄漏 | `timeout_seconds` 设成 0(禁用)或 destroy 未实现 |
| 不透明超时 | `SandboxCommandTimeoutError` 没进 `format_multiple_tool_results` 的归一分支 |