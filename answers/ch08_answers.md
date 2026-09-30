# 第 8 章 · 服务与监控 — 参考答案（ch08，第 48–50 课）

> **参考答案 · 建议先自己动手再看。** 本文件与仓库 `exercises/ch08/` 下 48–50 号 notebook 一一对应。
> 代码假设与你 notebook 中**已运行过的定义一致**才可独立运行（会标注依赖）；术语保留英文，答案用中文。

---

## 第 48 课 · 端到端部署

> 对应 `exercises/ch08/48_serve_model.ipynb`。依赖：`MockHandler` / `start_mock` / `VLLM_FLAGS` / `parse_completion`，与服务端口 `PORT = 18048`。

### 练习 1：给 MockHandler 增加 POST /v1/chat/completions
**思路**：`do_POST` 里按 `self.path` 分流；chat 版从 `messages` 里取 `role==user` 的 content 拼接后返回 assistant 回答。

```python
class MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(length))
        if self.path.startswith("/v1/chat/completions"):
            user = " ".join(m.get("content","") for m in req.get("messages",[]) if m.get("role")=="user")
            body = {"object":"chat.completion","model":req.get("model","mock"),
                    "choices":[{"index":0,"message":{"role":"assistant","content":f"[mock] 你好，你刚才说:{user}"},"finish_reason":"stop"}]}
            self._reply(json.dumps(body, ensure_ascii=False).encode())
            return
        prompt = req.get("prompt",""); self._reply(...)   # 保留原 completions 逻辑
# 客户端验证
r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
   json={"model":"mock","messages":[{"role":"user","content":"介绍一下 vLLM"}]}, timeout=5)
print(r.status_code, r.json()["choices"][0]["message"]["content"])
```
**预期结果**：返回 assistant 角色的 mock 回答（内含 user 内容），HTTP 200。注意原 `_reply` 已存在，直接复用。

### 练习 2：统计 usage，写一个按 token 计费的小函数
**思路**：计费 = `total_tokens / 1000 × 单价`（每千 token 一个价格）。

```python
def bill(usage, price_per_1k=0.02):     # 每 1k token $0.02
    total = usage["total_tokens"]
    return total / 1000 * price_per_1k
resp = {"usage":{"prompt_tokens":6, "completion_tokens":3, "total_tokens":9}}
print(f"总 token = {resp['usage']['total_tokens']} → 费用 = ${bill(resp['usage']):.5f}")
```
**预期结果**：按 `total_tokens/1000×price` 计费；9 token → `9/1000×0.02 = $0.00018`。可扩展成 `prompt`/`completion` 分开计价。

### 练习 3：把漏掉的 --max-num-seqs 补进表
**思路**：该参数是单次调度的最大序列数（同批 prefill/decode 上限），补一行字典进 `VLLM_FLAGS`。

```python
VLLM_FLAGS.append(dict(flag="--max-num-seqs", default="256", kind="可选",
    desc="单次最大同时调度序列数：调大提吞吐/利用率但吃显存与排队延迟，调小降低长上下文爆显存风险"))
```
**预期结果**：追加进参数表后，`vllm serve ... --max-num-seqs 128` 可显式控制并发序列数。作用：它是"吞吐 vs 显存/延迟"的开关——并发序列越多越吃 KV cache 与计算排队，长上下文并发场景需调小。

### 练习 4：在 app_48 加"并发请求数"滑杆，估算并发时 KV 占用
**思路**：每请求 KV 用量 × 并发数即为总 KV 占用，跟上下文长度线性相关。

```python
L, h_kv, d, ctx, cond = 32, 8, 128, 4096, 2   # Llama-3-8B 类配置
kv_per_req = 2*L*h_kv*d*2*ctx                 # 每请求 KV 字节(fp16)
c = st.slider("并发请求数", 1, 64, 8)
st.metric("KV 总占用(GB)", f"{kv_per_req*c/1024**3:.2f}")
st.caption("并发 × 单请求 KV = 总 KV cache，与上下文长度线性放大。")
```
**预期结果**：滑杆越大，KV 总占用近似线性上升；可直观看到"在固定显存下能支持多少并发长上下文请求"这一容量边界。

---

## 第 49 课 · OpenAI 兼容接口

> 对应 `exercises/ch08/49_openai_api.ipynb`。依赖：`next_token_probs` / `generate` / `start_chat_mock` / `VOCAB` / `EOS_ID`，服务端口 `CHAT_PORT = 18049`。

### 练习 1：给 generate 加 top_k 参数，与 top_p 对比
**思路**：top_k 固定只保留概率最高的 k 个候选再归一；top_p 是动态核（按累计概率）。

```python
def probs_topk(rng, temperature=1.0, top_p=1.0, top_k=0):   # top_k=0 表示关掉
    p = next_token_probs(rng, temperature, top_p)           # 先得基础分布(可含 top_p)
    if top_k > 0:
        k = min(top_k, len(p))
        keep = np.argpartition(p, -k)[-k:]                  # 概率最高的 k 个下标
        mask = np.zeros_like(p, bool); mask[keep] = True
        p = p * mask
        if p.sum() > 0: p = p / p.sum()
    return p
for tk in [1, 5, 50]:
    p = probs_topk(np.random.default_rng(0), top_k=tk)
    print(f"top_k={tk}: 最大概率={p.max():.3f}  有效候选={int((p>1e-9).sum())}")
```
**预期结果**：top_k 越小输出越集中（尽量只从高分 token 采）、抖动越小；top_p 是自适应核、能随分布自动多留/少留候选。二者常叠加使用（先 top_p 再 top_k）。

### 练习 2：100 个种子下 finish_reason=stop 的概率随 max_tokens 变化
**思路**：`P(stop) = P(在 m 步内采到 EOS)`；max_tokens 越大、越有机会遇到 EOS → stop 概率上升，并收敛到一个平台。

```python
import numpy as np
for mt in [4, 8, 16, 32]:
    stops = sum(1 for s in range(100)
                if generate("vLLM", max_tokens=mt, seed=s)[2] == "stop")
    print(f"max_tokens={mt:2d}: P(stop) = {stops/100:.2f}")
# 建议用 pyecharts Line 画 max_tokens vs P(stop),观察单调上升+平台
```
**预期结果**：小 max_tokens 时多数被"length"截断（stop 概率低）；max_tokens 增大后 stop 概率单调上升并趋近某个上限（由 EOS 出现频率决定）。曲线解释：max_tokens 是"截断保险"，给足采样空间则模型自然终止。

### 练习 3：把 mock server 的 SSE 改为一次推两个字符，验证 iter_lines 解析仍正确
**思路**：SSE 协议按 `\n\n` 分隔事件，事件内内容多少（1 个 token 或 2 个字符）不影响事件边界。

```python
# 在 start_chat_mock 的 _send_stream 里,把逐 token 推送改成两个 token 一推:
for i in range(0, len(tokens), 2):
    delta_text = "".join(VOCAB[t] for t in tokens[i:i+2] if t != EOS_ID)
    piece = {"choices":[{"index":0,"delta":{"content":delta_text},"finish_reason":None}]}
    self.wfile.write(f"data: {json.dumps(piece, ensure_ascii=False)}\n\n".encode())
    self.wfile.flush(); time.sleep(0.03)
```
**预期结果**：客户端用 `iter_lines` / 按 `\n\n` 拆事件仍能逐行正确拿到完整文本（只是 chunk 更大、更少）。SSE 行结构与 chunk 字符数解耦，解析不受影响。

### 练习 4：用 requests 同时发 5 个并发 chat 请求，统计 usage 与 finish_reason
**思路**：用 `ThreadPoolExecutor` 并发发请求，验证服务端能并行处理。

```python
import concurrent.futures as cf
url = f"http://127.0.0.1:{CHAT_PORT}/v1/chat/completions"
def one(i):
    r = requests.post(url, json={"model":"q","messages":[{"role":"user","content":f"Q{i}"}],"max_tokens":16}, timeout=10)
    j = r.json()
    return i, j["choices"][0]["finish_reason"], j["usage"]["total_tokens"]
with cf.ThreadPoolExecutor(5) as ex:
    for i, fr, tot in ex.map(one, range(5)):
        print(f"请求{i}: finish_reason={fr}, total_tokens={tot}")
```
**预期结果**：5 个请求并行发出、各自返回独立的 finish_reason 与 usage（total_tokens 互不串扰）——`ThreadingHTTPServer` 默认并发处理每个连接。若加锁/串行则会出现排队延迟。

---

## 第 50 课 · 性能指标与监控

> 对应 `exercises/ch08/50_metrics_monitor.ipynb`。依赖：`prefill` / `decode_step` / `bench` / `simulate` / `parse_avg` / `fake_metrics` / `throughput`，及基线量本身。

### 练习 1：把 prefill 的 token 数 P 从 64 改成 512，看 TTFT 变化
**思路**：prefill 是"一次性并行处理 P 个 token"（大矩阵计算），耗时基本随 P 近线性增长；TTFT≈prefill_time。

```python
for P in [64, 512]:
    t = bench(lambda: prefill(P), iters=5)    # TTFT ≈ prefill 时间
    print(f"P={P:3d}: TTFT ≈ {t*1000:.2f} ms")
```
**预期结果**：P 从 64→512（×8），TTFT 约放大数倍（接近线性，取决于是否撞到并行墙）。原因：prefill 并行度高，但同批处理 8 倍 token 的计算量也约 8 倍，因此**长 prompt 的首字（TTFT）明显更慢**——这是 vLLM 需对 prefill 请求做调度/分块的原因。

### 练习 2：把争抢系数 alpha 从 0.2 改成 0.05，重跑负载扫描，看饱和点是否右移
**思路**：alpha 越小，并发引起的延迟上升越缓，吞吐饱和点（增量趋 0）右移到更大并发。

```python
for alpha in [0.2, 0.05]:
    thr = []
    for c in [1,2,4,8,16,32,64]:
        thr.append(simulate(c, base_ttft, base_tpot, S, alpha)[-1])
    print(f"alpha={alpha}: 各并发吞吐 = {[round(v,1) for v in thr]}")
    print(f"  并发16→32 增量 = {thr[5]-thr[4]:.1f}")
```
**预期结果**：alpha=0.2 时到并发 16~32 吞吐增量已接近 0（饱和）；alpha=0.05 时同样并发下仍在增长、饱和点右移到更大并发。含义：争抢越轻，越可多压并发而不掉吞吐，代价是延迟上限仍随并发缓升。

### 练习 3：用正则把 fake_metrics 里的指标名改成自定义前缀，验证解析仍正确
**思路**：把 `vllm_` 前缀统一替换为用户前缀，改后 `parse_avg` 传新前缀名仍需解析出 sum/count 平均值。

```python
import re
def rename_metrics(text, prefix="app50"):
    return re.sub(r"\b(vllm_)([a-z_]+)", lambda m: prefix + "_" + m.group(2), text)
new = rename_metrics(fake_metrics, "app50")
avg = parse_avg(new, "app50_time_to_first_token_seconds")   # 前缀名为 app50_...
print(f"改名后 TTFT 平均 ≈ {avg*1000:.0f} ms (仍能解析)")
```
**预期结果**：`parse_avg` 传"app50_..."前缀仍得到 `sum/count` 平均值；正则只改指标名、不改数值与结构，Prometheus 文本解析对前缀不敏感。

### 练习 4：写函数：给定 E2E 预算，反推允许的最大并发数
**思路**：在 `simulate` 的 E2E 公式里，对 c 一维扫描/二分，找"E2E 首次达到预算"对应的最大 c。

```python
def max_concurrent_for(budget, S=512, ttft=0.25, tpot=0.035, alpha=0.2):
    c = 1
    while True:
        e2e = ttft*(1+alpha*(c-1)) + S*tpot*(1+alpha*(c-1)*0.5)
        if e2e >= budget: return max(c-1, 1)
        c += 1
for b in [3.0, 5.0, 10.0]:
    print(f"E2E 预算 {b}s → 最大允许并发 ≈ {max_concurrent_for(b)}")
```
**预期结果**：预算越宽松（如 10s），允许并发越高；预算紧（如 3s）则只允许个位数并发。该函数即为运维容量规划：`E2E` 一旦把预算定死，就解出了并发上限（也可用解析式 `c ≤ (预算 -（1-alpha)…）` 化简，二分足够）。

---