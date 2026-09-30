# -*- coding: utf-8 -*-
"""ch08 生成公共组件:Notebook 构建 + 各课共享的部署/客户端/指标源码字符串"""
import sys
import textwrap

sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\minivllm\exercises")
from nb_builder import Notebook, chapter_cover, wrapup

CH08 = r"D:\Project\21-Cpp_learn\explore\minivllm\exercises\ch08"
CHAPTER = "第 8 章 · 端到端 vLLM 部署"


def D(s):
    return textwrap.dedent(s).strip()


def new_nb(title, subtitle, emoji):
    return Notebook(title, subtitle=subtitle, emoji=emoji, chapter=CHAPTER)


# ---------------------------------------------------------------- 通用 KMP 保护头
# 每课第一段代码:设置 KMP 保护、固定 seed、统一跑 CPU(本机 Windows 不原生支持 vLLM,
# 全部用"教学演示 + 模拟 server"的方式讲解,不启动真实 vLLM)。
KMP_HEADER = D('''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OMP 冲突防护
import json, time, threading, http.server
import numpy as np
import requests, pandas as pd

print("部署环境就绪:requests =", requests.__version__, "| numpy =", np.__version__, "| pandas =", pd.__version__)
print("注意:本机(Windows)不原生支持 vLLM,本课用【模拟 server + 预生成真实输出】讲解,不启动真实 vLLM。")
''')

# ---------------------------------------------------------------- 模拟 OpenAI 兼容服务
# 第 48 / 49 课共用:一个基于 http.server 的迷你 OpenAI 兼容接口。
# 只实现 /v1/models 与 /v1/completions,返回确定性(但格式正确)的 JSON 响应。
MOCK_SERVER = D('''
PORT = 18048

class MockHandler(http.server.BaseHTTPRequestHandler):
    """迷你 OpenAI 兼容 server:只实现 /v1/models 与 /v1/completions。"""
    def log_message(self, *a):                 # 静默,避免刷屏
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            body = json.dumps({
                "object": "list",
                "data": [{"id": "mock-llm", "object": "model", "created": 1700000000,
                          "owned_by": "vllm"}],
            }).encode()
            self._reply(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        prompt = req.get("prompt", "") or ""
        ans = f"[mock] 收到 {len(prompt)} 个字符的 prompt,模型推理完成。"
        body = json.dumps({
            "id": "cmpl-mock-0001", "object": "text_completion", "created": 1700000000,
            "model": req.get("model", "mock-llm"),
            "choices": [{"index": 0, "text": ans, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 12,
                      "total_tokens": len(prompt) + 12},
        }).encode()
        self._reply(body)

    def _reply(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_mock():
    """启动模拟 server,返回 (server, thread)。调用方负责 shutdown/server_close。"""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
''')

# ---------------------------------------------------------------- vLLM serve 参数表
# 第 48 课:把 vllm serve 最常用的命令行参数整理成表,供讲解与 app 共用。
VLLM_FLAGS = D('''
VLLM_FLAGS = [
    dict(flag="--model",        default="Qwen/Qwen2-7B-Instruct", kind="必填",
         desc="HuggingFace 模型名或本地路径,决定加载哪个权重"),
    dict(flag="--served-model-name", default="qwen2-7b", kind="可选",
         desc="对外暴露的模型名(客户端请求里用的 model 字段)"),
    dict(flag="--max-model-len", default="32768", kind="可选",
         desc="最大上下文长度,过长会爆显存,过短会截断长文本"),
    dict(flag="--gpu-memory-utilization", default="0.90", kind="可选",
         desc="允许使用的显存比例,剩下的留给 KV cache 与计算图"),
    dict(flag="--tensor-parallel-size", default="1", kind="可选",
         desc="张量并行卡数,多卡时把模型切到多张 GPU"),
    dict(flag="--dtype",        default="auto", kind="可选",
         desc="权重精度:auto / float16 / bfloat16 / float32"),
    dict(flag="--quantization", default="auto", kind="可选",
         desc="量化方式:awq / gptq / fp8 / none 等"),
    dict(flag="--port",         default="8000", kind="可选",
         desc="HTTP 服务监听端口"),
    dict(flag="--host",         default="0.0.0.0", kind="可选",
         desc="监听地址,0.0.0.0 表示对外开放"),
    dict(flag="--enforce-eager", default="off", kind="可选",
         desc="关闭 CUDA Graph 优化,便于调试(默认关)"),
]
''')
