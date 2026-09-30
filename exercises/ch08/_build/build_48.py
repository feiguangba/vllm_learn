# -*- coding: utf-8 -*-
"""生成 48_serve_model.ipynb 与 app_48_serve.py"""
from pathlib import Path
from helpers import (D, new_nb, chapter_cover, wrapup, CH08,
                     KMP_HEADER, MOCK_SERVER, VLLM_FLAGS)

APP_FILE = "app_48_serve.py"

APP_48 = D('''
# -*- coding: utf-8 -*-
# app_48_serve.py — LLM 推理服务与 vllm serve 启动参数 🚀
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="🚀 48 · 服务部署", layout="wide")
st.title("🚀 第 48 课 · 端到端部署:vllm serve 与 OpenAI 兼容接口")

st.markdown("""
把训练好的大模型"端上桌",需要一个**推理服务**:它常驻内存、等待客户端发来请求,
并返回模型生成的结果。vLLM 提供了现成的入口 `vllm serve`,底层对外暴露的是
**OpenAI 兼容 API**。下方拖一拖启动参数,实时拼出对应的 `vllm serve` 命令,
并估算这条命令需要多少显存来装 KV cache。
""")

# ---------------------------------------------------------------- 模型配置(估算用)
MODELS = {
    "Qwen2-1.5B":  dict(layers=28, kv_heads=2,  head_dim=128, vram_gb=6.0),
    "Qwen2-7B":    dict(layers=28, kv_heads=4,  head_dim=128, vram_gb=16.0),
    "Llama3-8B":   dict(layers=32, kv_heads=8,  head_dim=128, vram_gb=16.0),
    "DeepSeek-R1-7B": dict(layers=32, kv_heads=8, head_dim=128, vram_gb=16.0),
}

with st.sidebar:
    st.header("🎛️ vllm serve 启动参数")
    model = st.selectbox("--model", list(MODELS.keys()))
    served_name = st.text_input("--served-model-name", "qwen2-7b")
    max_len = st.slider("--max-model-len", 1024, 131072, 32768, 1024)
    util = st.slider("--gpu-memory-utilization", 0.5, 1.0, 0.90, 0.01)
    tp = st.selectbox("--tensor-parallel-size", [1, 2, 4])
    quant = st.selectbox("--quantization", ["auto", "awq", "gptq", "fp8", "none"])
    port = st.number_input("--port", 1024, 65535, 8000, 1)
    st.caption("以上参数会拼成一条可复现的 vllm serve 启动命令。")

cfg = MODELS[model]
# KV cache 每 token 占用:2(K+V) × 层数 × KV头 × 头维度 × 2字节
kv_per_token = 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * 2
kv_max_bytes = kv_per_token * max_len

# ---------------------------------------------------------------- 拼出 vllm serve 命令
cmd = (f"vllm serve {model} "
       f"--served-model-name {served_name} "
       f"--max-model-len {max_len} "
       f"--gpu-memory-utilization {util:.2f} "
       f"--tensor-parallel-size {tp} "
       f"--quantization {quant} "
       f"--port {port}")
st.subheader("🧾 生成的 vllm serve 命令")
st.code(cmd, language="bash")

# ---------------------------------------------------------------- 指标
c1, c2, c3, c4 = st.columns(4)
c1.metric("模型", model)
c2.metric("KV cache / token", f"{kv_per_token/1024:.1f} KiB")
c3.metric("最大 KV cache 总量", f"{kv_max_bytes/1024**3:.2f} GiB")
c4.metric("可用显存(估算)", f"{cfg['vram_gb']*util:.1f} / {cfg['vram_gb']:.0f} GiB")
st.caption("⚠️ 若“最大 KV cache 总量”超过“可用显存”,长上下文请求就会因显存不足而失败(OOM)。")

# ---------------------------------------------------------------- 显存 vs max_len 图
st.subheader("📊 KV cache 显存随上下文长度变化")
seq = list(range(1024, max_len + 1, 1024))
kv = [kv_per_token * s / 1024 ** 3 for s in seq]
budget = cfg["vram_gb"] * util
fig = go.Figure()
fig.add_trace(go.Scatter(x=seq, y=kv, mode="lines", name="KV cache 显存",
                         line=dict(width=3, color="#4C78A8")))
fig.add_hline(y=budget, line_dash="dash", line_color="#E45756",
              annotation_text=f"显存预算 {budget:.1f} GiB", annotation_position="top right")
fig.update_layout(title=f"{model} · KV cache 显存 vs max-model-len(max_len={max_len})",
                  xaxis_title="上下文长度 max-model-len", yaxis_title="KV cache 显存 (GiB)",
                  height=420, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("💡 观察:KV cache 随上下文长度**线性**增长。max-model-len 拉得越大,"
           "能同时服务的并发请求越少;拉得太小,长文本会被硬生生截断。")

# ---------------------------------------------------------------- 启动参数表
st.subheader("📋 vllm serve 常用参数速查")
st.dataframe(pd.DataFrame([
    {"参数": "必填", "默认": "-", "说明": "vllm serve"},
    {"参数": "--model", "默认": "-", "说明": "模型名或路径(唯一必填项)"},
    {"参数": "--served-model-name", "默认": "同模型名", "说明": "对外暴露的名字"},
    {"参数": "--max-model-len", "默认": "模型上限", "说明": "最长上下文"},
    {"参数": "--gpu-memory-utilization", "默认": "0.90", "说明": "显存使用比例"},
    {"参数": "--tensor-parallel-size", "默认": "1", "说明": "张量并行卡数"},
    {"参数": "--dtype / --quantization", "默认": "auto", "说明": "精度与量化"},
    {"参数": "--host / --port", "默认": "0.0.0.0:8000", "说明": "监听地址与端口"},
]), use_container_width=True)

st.markdown("""
> 🔗 **真机部署**:以上命令在 **Linux / Docker(WSL2)** 上直接可用。Windows 不原生支持 vLLM,
> 推荐在 WSL2 或 Docker Desktop 里跑官方镜像:
> `docker run --runtime nvidia -p 8000:8000 vllm/vllm-openai --model Qwen/Qwen2-7B-Instruct`
> 详见 [vLLM 官方文档](https://docs.vllm.ai) 与
> [Docker 安装指南](https://docs.docker.com/engine/install/)。
""")
st.caption("《minivllm: 图解 vLLM 推理引擎》第 8 章 · 第 48 课配套演示")
''')

NB = new_nb("第 48 课 · 端到端部署:vllm serve 与 OpenAI 兼容接口",
            subtitle="什么是 LLM 推理服务?vllm serve 有哪些启动参数?用迷你模拟 server 看懂 /v1/completions 的请求与响应",
            emoji="🚀")

chapter_cover(NB,
    objectives=[
        "理解 LLM 推理服务的定位:常驻内存、等待请求、返回生成结果",
        "认识 OpenAI 兼容 API:base_url、/v1/models、/v1/completions、/v1/chat/completions",
        "掌握 vllm serve 核心启动参数(--model / --max-model-len / --gpu-memory-utilization 等)",
        "用一个迷你模拟 server 亲手演示请求 → 响应的完整格式",
        "理解 KV cache 显存随上下文长度线性增长,以及它对部署的约束",
    ],
    toc=[
        ("直觉:把模型端上桌", "餐厅比喻:推理服务 = 常驻后厨的厨师"),
        ("OpenAI 兼容 API 是什么", "base_url + 端点,一套全世界都会的接口"),
        ("vllm serve 启动参数", "命令行逐参数讲解(Docker/WSL2 部署作链接)"),
        ("迷你模拟 server:请求与响应", "http.server 实现 /v1/models 与 /v1/completions"),
        ("响应结构逐字段拆解", "id / choices / usage 各自代表什么"),
        ("显存账本与双图可视化", "pyecharts 参数分类 + plotly 显存增长曲线"),
        ("配套 Streamlit 演示", "app_48_serve.py:拖参数实时拼 vllm serve 命令"),
    ],
    links=[
        ("vLLM 官方文档", "https://docs.vllm.ai"),
        ("vLLM Serving 文档(OpenAI 兼容)", "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html"),
        ("OpenAI Chat Completions API", "https://platform.openai.com/docs/api-reference/chat"),
        ("Docker 安装指南", "https://docs.docker.com/engine/install/"),
    ])

NB.code(KMP_HEADER, "✅ 第一段代码:设置 KMP 保护、固定环境,并确认 requests / numpy / pandas 就绪。")

NB.md("## 1️⃣ 直觉:把模型端上桌 🍜",
D('''
你训练或下载了一个大模型,它是一副"脑子",但没人能用它——因为使用方(网页、App、脚本)需要通过
**网络请求**来问它问题。于是需要一道"上菜的工序":把模型装进一个**常驻服务**,让它监听端口、
接收请求、跑推理、把结果回给客户端。这个常驻进程,就是 **LLM 推理服务(inference server)**。

把它想成一家**餐厅**:模型是后厨的厨师,推理服务是整个餐厅的接待流程,客户端是顾客。
顾客(客户端)不用进后厨,只要对着菜单(API 规范)点菜(发请求),服务员就把菜端上来(返回结果)。
菜单越标准,顾客越多——这正是 **OpenAI 兼容 API** 的价值:全世界都按同一份菜单点菜。
'''))

NB.md("## 2️⃣ OpenAI 兼容 API:一份全世界都懂的菜单 📋",
D('''
OpenAI 兼容 API 约定了一组 **HTTP 端点(endpoint)**,客户端用标准的 JSON 与它们通信:

- **`GET /v1/models`**:列出服务上有哪些模型;
- **`POST /v1/completions`**:最朴素的"补全"接口,输入 `prompt`,输出续写的 `text`(旧式);
- **`POST /v1/chat/completions`**:带 `messages` 角色的对话接口(当前主流);
- **`POST /v1/embeddings`**:文本向量化。

客户端只需要知道三样东西:**base_url**(服务地址)、**API key**(可空)、**model**(模型名)。
vLLM 默认把 base_url 设为 `http://localhost:8000/v1`。下面先看一眼这两个关键端点长什么样:
'''))

NB.code(D('''
base_url = "http://localhost:8000/v1"

# ① 补全接口 /v1/completions(旧式,输入一段文字让它续写)
completion_req = {
    "model": "qwen2-7b",
    "prompt": "中国的首都是",
    "max_tokens": 16,
    "temperature": 0.7,
}
print("POST", base_url + "/completions")
print(json.dumps(completion_req, ensure_ascii=False, indent=2))

# ② 对话接口 /v1/chat/completions(新式,带角色)
chat_req = {
    "model": "qwen2-7b",
    "messages": [
        {"role": "system", "content": "你是一个乐于助人的助手"},
        {"role": "user", "content": "用一句话介绍 vLLM"},
    ],
    "max_tokens": 64,
}
print("\\nPOST", base_url + "/chat/completions")
print(json.dumps(chat_req, ensure_ascii=False, indent=2))
'''), "🎨 注意对比:`completions` 用 `prompt`,而 `chat/completions` 用 `messages`(带 role)。两者都是 OpenAI 兼容服务必备的端点。")

NB.md("## 3️⃣ vllm serve:一行命令把模型端上桌 🚀",
D('''
vLLM 把整个服务封装成了命令行工具 `vllm serve`。只要给一个**模型名/路径**,它就能启动上面那套
OpenAI 兼容服务。最常用的启动参数(本课 app 里可交互生成这条命令):
'''))

NB.code(VLLM_FLAGS, "📋 用一张表把 vllm serve 最常用的参数整理清楚。注意 `--model` 是唯一必填项,其余都有默认值。")

NB.code(D('''
flags = pd.DataFrame(VLLM_FLAGS)
print(flags[["flag", "default", "kind", "desc"]].to_string(index=False))
print()
print("典型启动命令:")
print("  vllm serve Qwen/Qwen2-7B-Instruct --served-model-name qwen2-7b \\\\")
print("        --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000")
'''), "✅ 一条命令:加载权重、分配显存、开监听端口、暴露 OpenAI 兼容接口。真机在 Linux / Docker 上直接跑即可。")

NB.md("## 4️⃣ 迷你模拟 server:亲手发一次请求 🌐",
D('''
Windows 上无法原生跑 vLLM,但"请求 → 响应"这套格式是完全可以在本机模拟的。
下面用 Python 标准库 `http.server` 写一个**迷你 OpenAI 兼容 server**,只实现
`/v1/models` 与 `/v1/completions`,然后用 `requests` 向它发真实 HTTP 请求——
你会发现:vLLM 对外做的事情,本质就是这套协议。用**单 cell 启动、用完即关**的方式避免端口/线程泄漏:
'''))

NB.code(MOCK_SERVER, "🔧 迷你 server 的逻辑很简单:解析请求 JSON → 拼一个响应 JSON → 写回。")

NB.code(D('''
server, thread = start_mock()                     # 启动模拟 server(后台线程)
try:
    # ① 列出模型
    r1 = requests.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
    print("GET /v1/models →", r1.status_code)
    print(json.dumps(r1.json(), ensure_ascii=False, indent=2))

    # ② 补全请求
    r2 = requests.post(f"http://127.0.0.1:{PORT}/v1/completions",
                       json={"model": "mock-llm", "prompt": "你好 vLLM", "max_tokens": 16},
                       timeout=5)
    print("\\nPOST /v1/completions →", r2.status_code)
    print(json.dumps(r2.json(), ensure_ascii=False, indent=2))
finally:
    server.shutdown()                             # 立即关闭,不留端口
    server.server_close()
    print("\\n[已关闭模拟 server]")
'''), "🎯 你会看到:一次推理请求 = 发一个 JSON、收一个 JSON。真实 vLLM 除了内容不同,协议完全一样。")

NB.md("## 5️⃣ 响应结构逐字段拆解 🔍",
D('''
刚才的响应虽然来自 mock,但字段结构和真实 vLLM 完全一致。逐字段看:

- **`id`**:这次请求的编号,用于日志追踪;
- **`model`**:真正服务的模型名;
- **`choices[0].text`**:模型生成的结果文本(completions 接口用 `text`,chat 接口用 `message.content`);
- **`choices[0].finish_reason`**:为什么结束(`stop` 自然结束 / `length` 达到 max_tokens);
- **`usage`**:本次用了多少 token(输入 + 输出 + 总计),是计费与统计的关键。

写个小函数,把响应里的"关键信息"抽取出来,像真实客户端那样只关心业务字段:
'''))

NB.code(D('''
def parse_completion(resp):
    """把 OpenAI 兼容响应抽成业务字段。"""
    choice = resp["choices"][0]
    return {
        "request_id": resp["id"],
        "model": resp["model"],
        "text": choice["text"],
        "finish_reason": choice["finish_reason"],
        "prompt_tokens": resp["usage"]["prompt_tokens"],
        "completion_tokens": resp["usage"]["completion_tokens"],
    }

demo_resp = {
    "id": "cmpl-demo", "object": "text_completion", "model": "mock-llm",
    "choices": [{"index": 0, "text": "北京。", "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
}
print("客户端只需关心这些字段:")
for k, v in parse_completion(demo_resp).items():
    print(f"  {k:<16}: {v}")
'''), "✅ 上层应用(聊天框、API 网关)通常只取 `text` / `message.content` 和 `usage`,其余字段用于排查问题。")

NB.md("## 6️⃣ 显存账本与双图可视化 📊",
D('''
部署时最需要盯住的资源是**显存**。推理服务的显存大头除了模型权重,还有 **KV cache**。
KV cache 每多缓存一个 token,就要多占
$2 \\times \\text{层数} \\times \\text{KV头} \\times \\text{头维度} \\times 2\\ \\text{字节}$,
所以它随上下文长度**线性**增长。先用 pyecharts 看启动参数类别分布,再用 plotly 画 KV cache 显存曲线:
'''))

NB.code(D('''
from pyecharts.charts import Pie
from pyecharts import options as opts
from collections import Counter

kinds = Counter(f["kind"] for f in VLLM_FLAGS)
pie = (Pie()
       .add("", [list(x) for x in kinds.items()], radius=["40%", "70%"])
       .set_global_opts(title_opts=opts.TitleOpts(title="vllm serve 参数类别占比"))
       .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} 个 ({d}%)")))
pie.render_notebook()
'''), "📊 pyecharts 环形图:大多数参数是“可选”(有合理默认),真正必填的只有 `--model`。")

NB.code(D('''
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go

layers, kv_heads, head_dim = 28, 4, 128
kv_per_token = 2 * layers * kv_heads * head_dim * 2          # K+V × 层 × 头 × 维度 × 2字节
seq = list(range(1024, 65536 + 1, 2048))
kv_gb = [kv_per_token * s / 1024 ** 3 for s in seq]

fig = go.Figure(go.Scatter(x=seq, y=kv_gb, mode="lines+markers",
                           line=dict(width=3, color="#4C78A8")))
fig.add_hline(y=14.0, line_dash="dash", line_color="#E45756",
              annotation_text="16GB 卡 90% 预算 ≈ 14.4GiB", annotation_position="top right")
fig.update_layout(title="KV cache 显存随上下文长度线性增长(单请求)",
                  xaxis_title="上下文长度 max-model-len", yaxis_title="KV cache 显存 (GiB)",
                  height=420, margin=dict(l=10, r=10, t=60, b=10))
print(f"每 token KV cache = {kv_per_token/1024:.1f} KiB")
print(f"max_model_len=32768 时 KV cache ≈ {kv_per_token*32768/1024**3:.1f} GiB(仅单个请求)")
fig
'''), "🎨 plotly 图:红线是显存预算。上下文越长,KV cache 占得越多——这就是部署时要在 max-model-len 与并发度之间做取舍的原因。")

NB.md("## 7️⃣ 配套 Streamlit 演示:拖一拖,拼一条启动命令 🎛️",
D('''
运行同目录下的 `app_48_serve.py`,可以拖动 **max-model-len / gpu-memory-utilization**、
切换模型与张量并行卡数,实时拼出对应的 `vllm serve` 命令,并画出 KV cache 显存曲线:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_48_serve.py
```

浏览器打开 **http://localhost:8501**。建议:把 max-model-len 从 4096 拖到 131072,看 KV cache 显存如何
一路逼近显存预算线;再切换不同模型,比较 KV cache 的差异。完整源码如下
(与同目录 `app_48_serve.py` 一字不差):
'''))

NB.code(f"%%writefile {APP_FILE}\n" + APP_48, "📜 这就是 app_48_serve.py 的完整源码,notebook 与 app 共用同一套参数表与显存估算逻辑,保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "LLM 推理服务 = 常驻进程,监听端口、接收请求、跑推理、返回结果(餐厅比喻)",
        "OpenAI 兼容 API 约定 base_url + 端点:/v1/models、/v1/completions、/v1/chat/completions",
        "vllm serve 一行命令即可部署,唯一必填项是 --model,其余参数均有合理默认",
        "请求/响应是标准 JSON:客户端看 text / message.content 与 usage,其余字段用于排查",
        "KV cache 随上下文长度线性增长,是部署时显存与并发取舍的关键约束",
    ],
    practice=[
        "给 MockHandler 增加 POST /v1/chat/completions(解析 messages 里的 user 内容并返回),用 requests 验证",
        "统计一次请求的 usage,写一个按 token 计费的小函数(每千 token 一个价格)",
        "把 VLLM_FLAGS 里漏掉的 --max-num-seqs(单次最大批处理序列数)补进表里,并解释它的作用",
        "在 app_48 里再加一个'并发请求数'滑杆,估算多个并发时 KV cache 的总占用",
    ],
    links=[
        ("vLLM Serving 文档", "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html"),
        ("OpenAI API Reference", "https://platform.openai.com/docs/api-reference"),
        ("Docker 部署指南", "https://docs.docker.com/engine/install/"),
    ])

NB.save(str(Path(CH08) / "48_serve_model.ipynb"))

app_path = Path(CH08) / APP_FILE
app_path.write_text(APP_48 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
