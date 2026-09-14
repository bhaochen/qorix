# 推理请求契约 — orchestrator → vLLM 的线上线下协议

> 单轮 vs 多轮;静态 vs 动态 token 预算;错误归一;tokenizer 一致性。这一步是所有「采样坏掉」问题的根源,也是理解 vLLM worker 端控制路由的钥匙。

## 1. 走的不是普通 HTTP,是「线程池同步 POST」

orchestrator 不直接 `await httpx.post`,而是把 `body` 抛进 `_http_pool`(2048 线程的 `ThreadPoolExecutor`),`loop.run_in_executor` 拿结果(generate.py:511-515)。源码注释说得很直白(该文件头部):

> With 2000+ concurrent requests, async httpx creates massive event loop contention (~3s overhead per request from JSON parsing + protocol handling).

配套 `_retry_request`(generate.py:326)指数退避重试(连接类错误),vLLM 的错误响应(`choices` 缺失)不重试,直接抛 `RolloutError`。

## 2. 请求体(`generate_completion_with_tokens`:489)

```jsonc
POST {server}/v1/completions
{
  "model": "<cfg.model>",
  "prompt": [[token_ids...]],        // list of lists,vLLM 免二次 tokenize
  "return_token_ids": true,
  "n": 1,
  "logprobs": 1,
  "skip_special_tokens": false,      // text 与 token_ids 严格对齐
  "include_stop_str_in_output": false,
  ...cfg.get_sampling_params(),      // temperature / top_p / repetition_penalty ...
  "max_tokens": <动态预算>,
  "priority": <request priority>
}
```

四个容易踩的开关:
- `skip_special_tokens: false` + `include_stop_str_in_output: false`:保证回传 text 能逐 token 反解(训练侧要用 token 级别的 logprobs 算 masked loss);
- `return_token_ids`:避免 orchestrator 再 tokenize 一遍，彻底闭环 "vLLM 的 token" ⟷ "tokenizer 的 id";
- `logprobs: 1` + `n: 1`:逐位置 logprob 是 GRPO/importance sampling 的真数据源;
- `priority`:调度器按此插队(early-break 之类)。

## 3. token 预算:三种错误一个归一处

`max_tokens` 不是配置里的死数——它被动态裁剪:

```
available = cfg.max_model_len - len(prompt_token_ids)
actual   = min(requested, available)     # generate.py:468-483
```

因此 vLLM 侧几乎不会真正打爆 context(除非 token 越界),但兜底仍然在:响应里没有 `choices` 时,grep 两类 vLLM 报错文本:

| 错误 | 识别 | 归一异常 |
|------|------|---------|
| context 太长 | `maximum context length is X tokens` + `request has Y input tokens` | `PromptTooLongError`:371(generate.py:397-407) |
| max_tokens 超出剩余 | `'max_tokens' ... 'is too large: Z'` + input count | `ContextExhaustedError`:379(generate.py:410-434) |

两处解析都是**正则**,文案变了就会回归——这是"vLLM 升级后静默挂掉"的高危点。

## 4. tokenizer 一致性:InterleavedTokenizer

多轮 interleaved 用 `InterleavedTokenizer`(generate.py:126):它把整段对话历史的 token ids **缓存 + 增量拼接**,而非每次重拼字符串再 tokenize。这保证:

- 单轮转多轮的 token 前缀**逐位相同**(`return_token_ids` 语义下模型不会"变脸");
- 训练侧的 mask/loss 对齐不漂移。

## 5. 服务端:7 行 vLLM + 4 个控制路由

inference/server.py 是薄壳:`run_server`(:155)起一个标准 vLLM OpenAI server,复用 vLLM 的 `argmax`/sampler,**只加 4 条控制路由**:

| 路由 | 用途 |
|------|------|
| `/collective_test` :70 | 广播前验证 NCCL 集合连通 |
| `/init_broadcast` :75 | 权重注入入口(worker 侧 `receive_state_dict`) |
| `/load_weights` :88 | 加载权重(lane 组装时 / cold start) |
| `/torch_memory` :95 | 显存遥测(供 orchestrator 做容量感知派单) |

编排侧**永远不会**把组播请求打给这些路由——它们可能挂在独立端口 / 控制 socket,`_pick_server_with_capacity` 只认计算 lane。

## 6. 读时序的四个桩

1. `timing_out` dict:调用方传进来,`start_time/end_time` 原地写(generate.py:502-525),**CancelledError 也会补 end_time**——off-policy 取消不会漏掉遥测;
2. `GenerationRecord`(event_logger.py:681)的 `queue_time/ttft/prefill_time/decode_time/inference_time/e2e_latency/search` 落帧;
3. lane 槽位 `_pop_first_available_lane_slot` 与「free_lane 回到服务器」对应——HTTP 结束即还槽;
4. 单轮在 `safe_rollout_loop` 里收 `stop_reason`(`eos`/`max_tokens`/`tool_call`/`tool_result_interrupt`)。