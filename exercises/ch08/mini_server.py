# -*- coding: utf-8 -*-
"""mini_server.py — TinyGPT 包装的「本地 OpenAI 兼容 HTTP server」。

用跨章共享库 `vllm_real.TinyGPT`(纯线性层小 GPT,真实前向)写一个 OpenAI 兼容的迷你
推理服务:既可以用 HTTP 真跑 `/v1/models` / `/v1/completions` / `/v1/chat/completions`
(含 `stream=True` 的 SSE 流式),又在每个请求里真实记录 prefill(首 token) / decode 耗时,
供 48 / 49 / 50 三课当「真实本地后端」用。

TinyGPT 很小,本地 CPU 秒起;HTTP 响应延迟、usage token 数都是**真实数字**——这就是
「没有真 vLLM 也能把端到端请求/响应/SSE/监控指标跑出真实数据」的关键。

独立运行(子进程方式,notebook 里 `subprocess.Popen`):
    uv_cuda\\python.exe mini_server.py --host 127.0.0.1 --port 18200

也可当作库 import 拿内部函数(current_stats)。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import string
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 让本文件无论从哪被 import 都能找到 vllm_real
_EX = r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises"
if _EX not in sys.path:
    sys.path.insert(0, _EX)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_NAME = "tinygpt"


def find_free_port() -> int:
    """找一个当前空闲的本地端口(避免多个 notebook / 多次执行撞车)。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


# ---------------------------------------------------------------------------
# 词表:中文常用字 + ASCII,足够 TinyGPT 的 vocab 用,也足够拼出"像话"的文本
# ---------------------------------------------------------------------------
_CN = ("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成"
       "会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等"
       "部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好"
       "应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数"
       "正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很"
       "情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长"
       "求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决"
       "西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志"
       "世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花"
       "带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米"
       "整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书"
       "布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团"
       "往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王"
       "按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"
       "龙飞模型推理引擎部署吞吐延迟监控指标请求响应服务客户端流式输出服务")

_VOCAB = sorted(set(_CN) | set(string.ascii_letters) | {" ", "\u3000", "，", "。", "!", "?", "\n"})


def _pad_vocab(vocab, size):
    """补齐词表到 size,避免 TinyGPT vocab 太小。"""
    if len(vocab) >= size:
        return vocab
    out = list(vocab)
    n = 0
    while len(out) < size:
        out.append(f"<c{n}>")
        n += 1
    return out


class TinyEngine:
    """TinyGPT 的包装:字符级 token 化 + 自回归生成,提供真实前向时序。"""

    def __init__(self, device: str = "cpu", d: int = 128, layers: int = 4,
                 vocab_size: int | None = None, window: int = 32):
        from vllm_real import TinyGPT
        self.device = device
        self.window = window
        self.vocab = _pad_vocab(_VOCAB, min(vocab_size or len(_VOCAB), 512))
        self.iv = {t: i for i, t in enumerate(self.vocab)}
        self.model = TinyGPT(d=d, layers=layers, vocab_size=len(self.vocab), device=device).to(device)
        self.params = self.model.params_count()

    def encode(self, text: str) -> list:
        ids = [self.iv.get(c, self.iv.get(" ", 0)) for c in (text or "")]
        return ids if ids else [0]

    def generate(self, prompt: str, max_tokens: int = 16, temperature: float = 0.7):
        """返回 (text, prompt_tokens, completion_tokens, ttft_s, decode_s, total_s)。

        ttft_s:从开始到**第一个输出 token** 的时间(≈ prefill + 首步);
        decode_s:得到首 token 之后的后续生成耗时(≈ (completion-1) 步)。
        """
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1) - 1)
        ids = self.encode(prompt)
        out_ids = []
        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.tensor(ids, device=self.device).unsqueeze(0)
            logits = self.model.forward_logits(x)          # 真实一次 prefill 前向
            first_t = time.perf_counter()
            for step in range(int(max_tokens)):
                log = logits[0, -1, :]
                if temperature > 1e-5:
                    p = torch.softmax(log / max(temperature, 1e-5), dim=0)
                    nxt = int(torch.multinomial(p, 1).item())
                else:
                    nxt = int(torch.argmax(log).item())
                out_ids.append(nxt)
                ctx = torch.tensor((ids + out_ids)[-self.window:], device=self.device).unsqueeze(0)
                logits = self.model.forward_logits(ctx)    # 真实 decode 一步前向
        tc = time.perf_counter()
        text = "".join(self.vocab[i] for i in out_ids)
        ttft = first_t - t0
        decode = tc - first_t
        return text, len(ids), len(out_ids), ttft, decode, tc - t0


# ---------------------------------------------------------------------------
# 请求级统计(供 /metrics 与 50 课真实监控数据)
# ---------------------------------------------------------------------------
_STATS = []  # list of dict:request, model, p, c, ttft_s, decode_s, total_s


def current_stats() -> list:
    return list(_STATS)


def _summarize(stats) -> dict:
    if not stats:
        return {"total_requests": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0,
                "avg_ttft_s": 0.0, "avg_tpot_s": 0.0, "throughput_tok_s": 0.0,
                "totals": 0.0, "avg_e2e_s": 0.0}
    total_p = sum(r["p"] for r in stats)
    total_c = sum(r["c"] for r in stats)
    total_t = sum(r["total_s"] for r in stats)
    avg_ttft = sum(r["ttft_s"] for r in stats) / len(stats)
    # TPOT:每个请求 decode 生成 (c-1) 个后续 token;合计后求末段平均
    tot_decode = sum(max(r["c"] - 1, 1) for r in stats)
    avg_tpot = sum(r["decode_s"] for r in stats) / max(tot_decode, 1)
    return {
        "total_requests": len(stats), "total_prompt_tokens": total_p,
        "total_completion_tokens": total_c,
        "total_tokens": total_p + total_c, "avg_ttft_s": avg_ttft,
        "avg_tpot_s": avg_tpot,
        "throughput_tok_s": (total_p + total_c) / max(total_t, 1e-12),
        "avg_e2e_s": total_t / len(stats),
    }


def _prometheus_text(stats) -> str:
    s = _summarize(stats)
    lines = ["# HELP tinygpt:time_to_first_token_seconds Distribution of TTFT (real)",
             "# TYPE tinygpt:time_to_first_token_seconds gauge",
             f"tinygpt:time_to_first_token_seconds {s['avg_ttft_s']:.6f}",
             "# HELP tinygpt:time_per_output_token_seconds Average TPOT (real)",
             "# TYPE tinygpt:time_per_output_token_seconds gauge",
             f"tinygpt:time_per_output_token_seconds {s['avg_tpot_s']:.6f}",
             "# HELP tinygpt:num_requests_running Current finished request count",
             "# TYPE tinygpt:num_requests_running gauge",
             f"tinygpt:num_requests_running {s['total_requests']}",
             "# HELP tinygpt:num_tokens_total Total generated tokens",
             "# TYPE tinygpt:num_tokens_total counter",
             f"tinygpt:num_tokens_total {s['total_tokens']}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class MiniHandler(BaseHTTPRequestHandler):
    engine: TinyEngine = None

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "vllm",
                 "meta": {"params": self.engine.params, "device": self.engine.device}}]})
        elif self.path.startswith("/metrics"):
            self._json(200, {"summary": _summarize(_STATS),
                             "requests": [dict(r) for r in current_stats()]})
        elif self.path.startswith("/metrics.prom"):
            body = _prometheus_text(_STATS).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"object": "error", "message": f"no route {self.path}"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0)) or 0
        try:
            req = json.loads(self.rfile.read(length)) if length else {}
        except Exception:
            return self._json(400, {"error": "bad json"})
        model = req.get("model") or MODEL_NAME
        stream = bool(req.get("stream", False))
        max_tokens = max(0, min(int(req.get("max_tokens", 16)), 256))
        temperature = float(req.get("temperature", 0.7))

        if self.path.startswith("/v1/chat/completions"):
            msgs = req.get("messages", [])
            prompt = " ".join(
                (m.get("content", "") or "") for m in msgs if m.get("role") == "user")
            prompt = prompt or " ".join(str(m.get("content", "")) for m in msgs)
            kind = "chat"
        elif self.path.startswith("/v1/completions"):
            prompt = req.get("prompt", "") or ""
            kind = "completion"
        else:
            return self._json(404, {"error": f"no POST route {self.path}"})

        t0 = time.perf_counter()
        text, p, c, ttft, decode, total = self.engine.generate(prompt, max_tokens, temperature)
        rec = {"t_req": t0, "model": model, "kind": kind, "p": p, "c": c,
               "ttft_s": ttft, "decode_s": decode, "total_s": total}
        _STATS.append(rec)
        created = int(time.time())

        if stream:
            self._stream((text, kind, model), created)
        else:
            if kind == "chat":
                body = {"id": "chatcmpl-%016x" % random.getrandbits(64),
                        "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant", "content": text},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": p, "completion_tokens": c,
                                  "total_tokens": p + c}}
            else:
                body = {"id": "cmpl-%016x" % random.getrandbits(64),
                        "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": p, "completion_tokens": c,
                                  "total_tokens": p + c}}
            self._json(200, body)

    def _stream(self, args, created):
        text, kind, model = args
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        buf = bytearray()
        generated = 0
        # 逐 token 推送,读一个发一个,让客户端能抓到真实的首 token 时延
        for i, ch in enumerate(text):
            if kind == "chat":
                piece = {"id": "chatcmpl-stream", "object": "chat.completion.chunk",
                         "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"content": ch},
                                      "finish_reason": None}]}
            else:
                piece = {"id": "cmpl-stream", "object": "text_completion",
                         "created": created, "model": model,
                         "choices": [{"index": 0, "text": ch, "finish_reason": None}]}
            line = f"data: {json.dumps(piece, ensure_ascii=False)}\n\n".encode("utf-8")
            buf.extend(line)
            generated = i + 1
            if len(buf) >= 512:
                self.wfile.write(bytes(buf)); self.wfile.flush(); buf = bytearray()
        if buf:
            self.wfile.write(bytes(buf)); self.wfile.flush()
        final_delta = None
        finish = "stop"
        if kind == "chat":
            final = {"content": None, "role": "assistant", "usage": None}
            piece = {"id": "chatcmpl-stream", "object": "chat.completion.chunk",
                     "created": created, "model": model,
                     "choices": [{"index": 0, "delta": final, "finish_reason": finish}]}
        else:
            piece = {"id": "cmpl-stream", "object": "text_completion",
                     "created": created, "model": model,
                     "choices": [{"index": 0, "text": "", "finish_reason": finish}]}
        self.wfile.write(f"data: {json.dumps(piece, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(host: str = "127.0.0.1", port: int = 0, device: str = "cpu",
          d: int = 128, layers: int = 4) -> tuple:
    """启动并返回 (server, port)。调用方负责 server.shutdown()/server_close()。"""
    engine = TinyEngine(device=device, d=d, layers=layers)
    MiniHandler.engine = engine
    server = ThreadingHTTPServer((host, port), MiniHandler)
    return server, server.server_address[1]


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="TinyGPT OpenAI-compatible local server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    return ap


def main():
    args = _build_arg_parser().parse_args()
    engine = TinyEngine(device=args.device, d=args.d, layers=args.layers)
    MiniHandler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), MiniHandler)
    port = server.server_address[1]
    print(json.dumps({"status": "ok", "model": MODEL_NAME, "host": args.host,
                      "port": port, "device": engine.device,
                      "params": engine.params}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()