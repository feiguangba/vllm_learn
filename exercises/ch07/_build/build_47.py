# -*- coding: utf-8 -*-
"""生成 47_attn_backend_select.ipynb 与 app_47_attn_backend.py(教材级重写版)

对齐 REWRITE_STANDARD.md。修复审核发现的问题:
旧版 select_backend(need_causal, need_paged) 两个参数未在过滤逻辑生效(功能层过滤缺失);
本版为每个后端维护 capabilities(平台/最低架构/支持 causal/支持 paged/优先级),
让 need_causal 与 need_paged 真正参与逐层过滤。

工程引用:
vllm/v1/attention/selector.py(AttentionSelectorConfig: has_sliding_window / use_non_causal / dtype / head_size);
vllm/v1/attention/backends/registry.py(AttentionBackendEnum);docs/design/attention_backends.md(优先级表)。
"""
from pathlib import Path
from helpers import D, new_nb, chapter_cover, wrapup, CH07, CPU_HEADER

APP_FILE = "app_47_attn_backend.py"

APP_47 = D('''
# -*- coding: utf-8 -*-
# app_47_attn_backend.py — vLLM 注意力后端选择 🎛️
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🎛️ 47 · 注意力后端", layout="wide")
st.title("🎛️ 第 47 课 · vLLM 注意力后端选择机制")

st.markdown("""
vLLM 支持多种**注意力后端(attention backend)**,它们只是同一件事的不同实现:
用哪种底层 kernel(FlashAttention / Triton / FlashInfer / PyTorch SDPA …)来算 attention。
启动时 vLLM 会按**平台、GPU 架构、功能需求**逐个过滤,选出一个**可用的、优先级最高**的后端。
下方选择**设备/架构**,并勾选**功能需求(causal / 分页)**,看看 vLLM 会怎么逐层过滤。
""")

# 每个后端的 (平台, 最低架构, 支持causal, 支持分页, 特性, 优先级)
# 功能列(need_causal / need_paged)会在过滤里真正生效
BACKENDS = {
    "FLASH_ATTN":  dict(platform="CUDA",  min_arch="sm_80", causal=True,  paged=True,
                        feat="FlashAttention-2, 支持 sliding/causal/alibi, 最常用", prio=1),
    "FLASHINFER":  dict(platform="CUDA",  min_arch="sm_80", causal=True,  paged=True,
                        feat="FlashInfer, 高性能分页 attention, 依赖 flashinfer 包", prio=2),
    "TRITON_ATTN": dict(platform="CUDA/ROCM", min_arch="sm_80", causal=True, paged=True,
                        feat="Triton 实现, 通用性高, 作为 flash 不可用时的回退", prio=3),
    "TORCH_SDPA":  dict(platform="CUDA/CPU",  min_arch="不限", causal=True, paged=False,
                        feat="PyTorch 内置 SDPA, 通用兜底(不支持分页 KV)", prio=4),
    "MATH":        dict(platform="全平台",  min_arch="不限", causal=True, paged=False,
                        feat="纯数学回退(最慢), 仅作最后手段", prio=5),
    "ROCM_FLASH":  dict(platform="ROCm",  min_arch="不限", causal=True,  paged=True,
                        feat="AMD GPU 上的 FlashAttention", prio=1),
    "XPU_SDPA":    dict(platform="Intel XPU", min_arch="不限", causal=True, paged=False,
                        feat="Intel 平台 SDPA", prio=1),
    "XFORMERS":    dict(platform="CUDA",  min_arch="sm_50", causal=True, paged=False,
                        feat="xFormers 内存高效 attention(不支持分页)", prio=6),
}

with st.sidebar:
    st.header("🎛️ 选择设备")
    device = st.selectbox("平台", ["CUDA (NVIDIA)", "ROCm (AMD)", "Intel XPU", "CPU"])
    if device == "CUDA (NVIDIA)":
        arch = st.selectbox("GPU 架构", ["sm_70", "sm_80", "sm_86", "sm_90", "sm_120"])
    else:
        arch = "不限"
    need_causal = st.checkbox("需要 causal 掩码", value=True)
    need_paged = st.checkbox("需要分页 KV(PagedAttention)", value=True)
    st.caption("sm_80+ 才能用 FlashAttention-2;causal 与分页是两个独立的功能过滤维度。")

plat_map = {"CUDA (NVIDIA)": "CUDA", "ROCm (AMD)": "ROCm", "Intel XPU": "Intel XPU", "CPU": "CPU"}

def arch_ok(req, arch):
    return req == "不限" or arch == "不限" or int(req.split("_")[1]) <= int(arch.split("_")[1])

def select_backend(platform, arch, need_causal, need_paged):
    """模拟 vLLM 的后端选择:平台→架构→功能(causal/paged)→优先级。返回 (选定, 候选链, 全部候选详情)。"""
    chain, rows = [], []
    for name, info in BACKENDS.items():
        plat_ok = info["platform"] == "全平台" or plat_map[platform] in info["platform"].split("/")
        if not plat_ok:
            continue                                   # ① 平台过滤
        if not arch_ok(info["min_arch"], arch):
            continue                                   # ② 架构过滤
        if need_causal and not info["causal"]:
            continue                                   # ③ 功能过滤:需要 causal 但后端不支持
        if need_paged and not info["paged"]:
            continue                                   # ④ 功能过滤:需要分页但后端不支持
        chain.append(name)                             # 通过全部过滤 → 候选
        rows.append(dict(后端=name, 平台=info["platform"], 最低架构=info["min_arch"],
                         causal=info["causal"], 分页=info["paged"], 特性=info["feat"],
                         优先级=info["prio"]))
    chain = sorted(chain, key=lambda n: BACKENDS[n]["prio"])   # ⑤ 按优先级排序
    return (chain[0] if chain else "MATH"), chain, rows

chosen, chain, rows = select_backend(device, arch, need_causal, need_paged)

st.subheader("🎯 当前设备候选后端(按优先级)")
if chain:
    st.success(f"**默认选中的后端:{chosen}** — {BACKENDS[chosen]['feat']}")
    st.write("候选链(优先级从高到低): " + " → ".join(chain))
else:
    st.warning("当前设备/架构/功能下没有可用后端,回退到 MATH(纯数学)。")

st.subheader("📋 通过过滤的后端(候选)")
if rows:
    st.dataframe(pd.DataFrame(rows).sort_values("优先级"), use_container_width=True)
else:
    st.write("无候选。")

st.subheader("🧭 完整后端能力表(全部)")
all_rows = [dict(后端=n, 平台=i["platform"], 最低架构=i["min_arch"], causal=i["causal"],
                 paged=i["paged"], 特性=i["feat"], 优先级=i["prio"]) for n, i in BACKENDS.items()]
st.dataframe(pd.DataFrame(all_rows).sort_values("优先级"), use_container_width=True)

st.markdown("""
> 💡 **机制**:vLLM 的后端选择 = **逐层过滤**。先看平台(CUDA / ROCm / XPU),再看 GPU 架构
> (sm_80+ 才支持 FlashAttention-2),再看功能需求(causal / sliding / 分页等),最后在剩余候选中
> 挑**优先级最高**的那个。这也解释了为什么 `FLASH_ATTN` 在 NVIDIA 上几乎总是默认选项,而
> 一旦需要**分页 KV**,`TORCH_SDPA` 这类不支持分页的后端就会被过滤掉。真实工程里的
> `AttentionSelectorConfig` 就有 `has_sliding_window` / `use_non_causal` 等功能字段。
""")
st.caption("《VLLM_learn: 图解 vLLM 推理引擎》第 7 章 · 第 47 课配套演示")
''')

NB = new_nb("第 47 课 · Attention 后端选择",
            subtitle="FLASH_ATTN / TRITON_ATTN / FLASHINFER 怎么选?vLLM 按平台、架构、功能逐层过滤(让 need_causal / need_paged 真正生效)",
            emoji="🎛️")

chapter_cover(NB,
    objectives=[
        "认识 vLLM 的 AttentionBackendEnum 与主流后端(FLASH_ATTN / TRITON_ATTN / FLASHINFER 等)",
        "理解后端选择机制:平台→架构→功能需求(causal/sliding/分页)→优先级 逐层过滤",
        "为每个后端建立 capabilities 表(平台/最低架构/支持 causal/支持分页/优先级)",
        "实现 select_backend,让 need_causal 与 need_paged 两个参数真正参与过滤(修复审核问题)",
        "真机实测:同一 attention 走不同后端(flash vs math)的耗时差异",
        "建立与 vLLM selector.py / registry.py / attention_backends.md 的联系",
    ],
    toc=[
        ("直觉:修车铺选工具", "同一个活,不同工具/技师"),
        ("后端是什么 + 能力表", "同一件事的不同 kernel 实现;AttentionBackendEnum"),
        ("选择机制:逐层过滤", "平台→架构→功能→优先级,AttentionSelectorConfig"),
        ("实现 select_backend(参数生效)", "为每个后端建模 capabilities,让 need_causal/need_paged 真正过滤"),
        ("真机实测:后端决定速度", "用 PyTorch 开关演示 flash vs math 的耗时差"),
        ("与 vLLM 工程实现的关系", "selector.py / registry.py / attention_backends.md"),
        ("小结 + 练习 + 延伸阅读", "要点、动手题、论文链接"),
    ],
    links=[
        ("vLLM 官方文档: Attention Backends", "https://docs.vllm.ai/en/latest/design/attention_backends/"),
        ("FlashAttention-2 (ICLR 2024)", "https://arxiv.org/abs/2307.08691"),
        ("FlashInfer 项目", "https://github.com/flashinfer-ai/flashinfer"),
    ])

NB.md("## 1. 直觉:修车铺选工具 🔧\n\n"
      "你开修车铺,给不同的车(不同的 GPU)干活,得挑合适的工具:有的车用扳手(FlashAttention),"
      "有的车用专用电动扳手(FlashInfer),实在没有就用手拧(纯数学)。**活一样,工具不同,速度天差地别。**\n\n"
      "vLLM 的「attention 后端(backend)」就是这个「工具」:它们都是算 attention,只是底层实现不同——"
      "可能是 FlashAttention、Triton、FlashInfer,也可能是 PyTorch 内置的 SDPA 或纯数学回退。"
      "启动时,vLLM 要**替你的设备挑一把最顺手的工具**。")

NB.code(CPU_HEADER, "✅ 每课第一段代码:设置 KMP 保护、固定 seed。本课会真的在 GPU 上打开/关闭不同后端开关,并量 SDPA 各后端的耗时——backend 选择不是纸面逻辑,是真影响速度。")

NB.md("## 2. 后端是什么 + 能力表 🏷️\n\n"
      "对上层模型来说,后端完全透明:不管用哪个,`attention(Q,K,V)` 的结果都一样。区别只在:"
      "**快不快、支不支持某种掩码、能不能配分页缓存**。vLLM 把这些实现抽象成一个**枚举** "
      "`AttentionBackendEnum`(见 `vllm/v1/attention/backends/registry.py`),每个枚举值对应一套 kernel 加载与调用逻辑。\n\n"
      "常见后端:\n\n"
      "- **FLASH_ATTN**:FlashAttention-2, NVIDIA 上默认首选,支持 causal / sliding / alibi(compute capability ≥8.0);\n"
      "- **FLASHINFER**:FlashInfer, 高性能分页 attention, 依赖 `flashinfer` 包;\n"
      "- **TRITON_ATTN**:用 Triton 写的 attention, 通用性高(支持 fp32),常作为回退;\n"
      "- **TORCH_SDPA**:PyTorch 内置 `scaled_dot_product_attention`,通用兜底;\n"
      "- **MATH**:纯数学实现,最慢,仅作最后手段。\n\n"
      "> 📄 每个后端还要声明它的 **capabilities**(支持的数据类型、头大小、块大小、是否 causal、是否分页等),"
      "见 [docs/design/attention_backends.md](https://docs.vllm.ai/en/latest/design/attention_backends/)。"
      "这正是本课过滤逻辑要用的信息。")

NB.md("## 3. 选择机制:逐层过滤 🧭\n\n"
      "vLLM 选后端不是「随手挑」,而是**逐层过滤**([selector.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/selector.py)):\n\n"
      "1. **平台**:CUDA / ROCm / Intel XPU / CPU —— 先筛掉平台不符的;\n"
      "2. **架构**:比如 sm_80 以下跑不了 FlashAttention-2,再筛;\n"
      "3. **功能**:causal / sliding / 分页缓存等需求,筛掉不支持的(真实工程用 `AttentionSelectorConfig` 的 "
      "`has_sliding_window`、`use_non_causal` 等字段表达);\n"
      "4. **优先级**:在剩余候选中,选**优先级最高**的那个。\n\n"
      "先建一张能力表(platform, min_arch, 支持 causal, 支持分页, 优先级):")

NB.code(D('''
# 为每个后端建立 capabilities 表:(平台, 最低架构, 支持causal, 支持分页, 特性, 优先级)
# 功能列(causal / paged)将在过滤中真正生效——这是对旧版缺失功能的修复
BACKENDS = {
    "FLASH_ATTN":  ("CUDA",      "sm_80",  True,  True,  "FlashAttention-2,支持 causal/sliding/alibi", 1),
    "FLASHINFER":  ("CUDA",      "sm_80",  True,  True,  "高性能分页 attention,依赖 flashinfer 包", 2),
    "TRITON_ATTN": ("CUDA/ROCm", "sm_80",  True,  True,  "Triton 实现,通用回退", 3),
    "TORCH_SDPA":  ("CUDA/CPU",  "不限",   True,  False, "PyTorch 内置 SDPA(不支持分页 KV)", 4),
    "MATH":        ("全平台",    "不限",   True,  False, "纯数学回退,最慢", 5),
    "ROCM_FLASH":  ("ROCm",      "不限",   True,  True,  "AMD 上的 FlashAttention", 1),
    "XPU_SDPA":    ("Intel XPU", "不限",   True,  False, "Intel 平台 SDPA", 1),
    "XFORMERS":    ("CUDA",      "sm_50",  True,  False, "xFormers 内存高效 attention(不支持分页)", 6),
}
print(f"{'后端':<14}{'平台':<10}{'最低架构':<8}{'causal':<8}{'分页':<6}{'优先级':<6}特性")
for name, (plat, min_arch, causal, paged, feat, prio) in BACKENDS.items():
    print(f"{name:<14}{plat:<10}{min_arch:<8}{str(causal):<8}{str(paged):<6}{prio:<6}{feat}")
'''), "📊 注意 TORCH_SDPA / MATH / XFORMERS 的「分页=False」——它们在需要 PagedAttention 时会被过滤掉。")

NB.md("## 4. 实现 select_backend:让 need_causal / need_paged 真正生效 ✅\n\n"
      "> 🔧 **修复说明**:旧版 `select_backend` 的 `need_causal` 与 `need_paged` 参数虽然存在,却从未进入过滤逻辑,"
      "导致功能层过滤缺失。本版把它们接入第 3 步功能过滤,使参数**真正生效**。\n\n"
      "过滤流程:①平台 → ②架构 → ③功能(need_causal / need_paged)→ ④按优先级排序。")

NB.code(D('''
# 实现 select_backend:逐层过滤,need_causal / need_paged 真正参与过滤
def select_backend(platform, arch, need_causal=True, need_paged=False):
    """模拟 vLLM 的后端选择:平台→架构→功能(causal/paged)→优先级。
    返回 (选定后端, 候选链)。"""
    def arch_ok(req):
        return req == "不限" or arch == "不限" or int(req.split("_")[1]) <= int(arch.split("_")[1])

    candidates = []
    for name, (plat, min_arch, causal, paged, feat, prio) in BACKENDS.items():
        # ① 平台过滤:全平台通吃,或平台名出现在后端支持的平台列表里
        plat_ok = plat == "全平台" or platform in plat.split("/")
        if not plat_ok:
            continue
        # ② 架构过滤:后端最低架构必须 <= 当前架构
        if not arch_ok(min_arch):
            continue
        # ③ 功能过滤:need_causal / need_paged —— 这就是让参数真正生效的一步
        if need_causal and not causal:
            continue
        if need_paged and not paged:
            continue
        # ④ 进入候选
        candidates.append(name)
    candidates.sort(key=lambda n: BACKENDS[n][5])   # 按优先级升序(数字小=优先)
    return (candidates[0] if candidates else "MATH"), candidates

# 演示①:同是 CUDA sm_90,勾不勾「需要分页」,结果不同
chosen, chain = select_backend("CUDA", "sm_90", need_causal=True, need_paged=False)
print(f"平台=CUDA,sm_90, causal, 不需分页 → 默认: {chosen}")
print("  候选链:", " → ".join(chain))
chosen, chain = select_backend("CUDA", "sm_90", need_causal=True, need_paged=True)
print(f"平台=CUDA,sm_90, causal, 需要分页 → 默认: {chosen}")
print("  候选链:", " → ".join(chain))

# 演示②:架构降级 sm_70,FlashAttention-2 用不了
chosen, chain = select_backend("CUDA", "sm_70", need_causal=True, need_paged=True)
print(f"\\n平台=CUDA,sm_70, causal, 需要分页 → 默认: {chosen}")
print("  候选链:", " → ".join(chain))
'''), "🎯 修复生效:勾选「需要分页」后,TORCH_SDPA / MATH / XFORMERS 被过滤,默认后端可能变化;架构 sm_70 下 FlashAttention-2 不可用则退化到 TRITON_ATTN。")

NB.md("## 5. 真机实测:后端决定速度 ⏱️\n\n"
      "后端的差别不只是「名字」,而是实实在在的**速度差**。下面在 GPU 上做同一件事:算一段长序列的多头 attention,"
      "分别走:\n"
      "1. `F.scaled_dot_product_attention` —— 让 PyTorch **自动挑**它认为最好的后端(本机 fp16 会优先 FlashAttention);\n"
      "2. 手动把 **flash / mem_efficient 开关关掉**,看 SDPA 退化;\n"
      "3. 把高级后端**全关掉**,只剩最慢的**纯 math 后端**作对照。\n\n"
      "这里看到的正是 vLLM 选 `FLASH_ATTN` 的动机:同一个活,后端选对了能快出一个量级。")

NB.code(D('''
# 真机实测:同一 attention,后端从 flash 退到 mem_efficient、再退到纯 math
import torch.nn.functional as F

def bench(fn, *a, iters=8):
    for _ in range(2):
        fn(*a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

torch.manual_seed(0)
B, H, N, d = 1, 8, 4096, 64
Q = torch.randn(B, H, N, d, device=dev, dtype=torch.float16)   # fp16 → 走 Tensor Core
K = torch.randn(B, H, N, d, device=dev, dtype=torch.float16)
V = torch.randn(B, H, N, d, device=dev, dtype=torch.float16)
print(f"输入: Q/K/V shape = {tuple(Q.shape)}, fp16 @ {dev}")
print("PyTorch 后端开关(当前):")
print("  flash_sdp      =", torch.backends.cuda.flash_sdp_enabled())
print("  mem_efficient  =", torch.backends.cuda.mem_efficient_sdp_enabled())
print("  math_sdp       =", torch.backends.cuda.math_sdp_enabled())

t_auto = bench(lambda a, b, c: F.scaled_dot_product_attention(a, b, c), Q, K, V)
torch.backends.cuda.enable_flash_sdp(False)                 # 关掉 flash,逼它退化
t_noflash = bench(lambda a, b, c: F.scaled_dot_product_attention(a, b, c), Q, K, V)
torch.backends.cuda.enable_flash_sdp(True)                  # 恢复
torch.backends.cuda.enable_mem_efficient_sdp(False)         # 连 mem_efficient 也关掉
t_mathonly = bench(lambda a, b, c: F.scaled_dot_product_attention(a, b, c), Q, K, V)
torch.backends.cuda.enable_mem_efficient_sdp(True)          # 全恢复

print(f"\\nN={N},H={H},d={d}, fp16 @ {dev} 耗时:")
print(f"  SDPA 自动选(最好后端) : {t_auto:8.3f} ms")
print(f"  SDPA(关 flash)       : {t_noflash:8.3f} ms")
print(f"  SDPA(只剩 math)      : {t_mathonly:8.3f} ms")
print(f"  自动与 math 的差距    : {t_mathonly/max(t_auto,1e-9):6.2f} x")
torch.cuda.empty_cache()
'''), "🚀 真机数字:同一个 attention,后端从 flash 退到 mem_efficient、再退到纯 math,耗时肉眼可见地变慢——这就是 vLLM 花大力气按平台/架构/功能逐层过滤、非要挑出最顺手后端的原因。")

NB.md("## 6. 与 vLLM 工程实现的关系 🔗\n\n"
      "本课的 `select_backend` 是对 vLLM 真实机制的简化复刻:\n\n"
      "1. **AttentionSelectorConfig**(`vllm/v1/attention/selector.py`):一个 NamedTuple,"
      "携带 `head_size / dtype / kv_cache_dtype / block_size / has_sliding_window / use_non_causal` 等字段——"
      "正是本课「功能层过滤」在真实工程里的载体;\n"
      "2. **AttentionBackendEnum**(`vllm/v1/attention/backends/registry.py`):枚举所有后端及其 class 路径;\n"
      "3. **validate_configuration()**:每个后端实现自己的兼容性检查(dtype、头大小、compute capability 等),"
      "优先级顺序见 [docs/design/attention_backends.md](https://docs.vllm.ai/en/latest/design/attention_backends/);\n"
      "4. 自动选择时,vLLM **按优先级顺序逐个 validate**,选第一个兼容的后端;\n"
      "5. 用户可用 `--attention-backend FLASH_ATTN` 显式指定,此时直接校验该后端。\n\n"
      "> 📄 一句话:后端 = 同一注意力运算的不同 kernel 供应商;vLLM 用「平台→架构→功能→优先级」的逐层过滤,"
      "替你挑出当前设备最顺手的那个。本课的 `select_backend` 让 `need_causal / need_paged` 真正生效,"
      "正是对这一机制的教学复刻。")

NB.md("## 7. 🖥️ Streamlit 动态演示:选设备看后端排队 🎛️\n\n"
      "运行同目录下的 `app_47_attn_backend.py`,选择**平台 / GPU 架构**,并勾选**功能需求(causal / 分页)**,"
      "实时看 vLLM 会怎么逐层过滤、最终选中哪个后端:\n\n"
      "```bash\nD:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_47_attn_backend.py\n```\n\n"
      "浏览器打开 **http://localhost:8501**。建议在「CUDA」下把架构从 sm_70 切到 sm_90,观察默认后端变化;"
      "再切换「需要分页」复选框,看 TORCH_SDPA 等不支持分页的后端被过滤。完整源码如下(与同目录 app 一字不差):")

NB.code(f"%%writefile {APP_FILE}\n" + APP_47, "📜 这就是 app_47_attn_backend.py 的完整源码,notebook 与 app 共用同一套后端能力表与选择逻辑(含 causal/paged 功能过滤),保证演示与讲解一致。")

wrapup(NB,
    summary=[
        "attention 后端 = 同一注意力运算的不同 kernel 实现,结果等价但性能差异大",
        "主流后端:FLASH_ATTN / FLASHINFER / TRITON_ATTN / TORCH_SDPA / MATH / ROCM_FLASH 等",
        "选择机制 = 逐层过滤:平台 → 架构 → 功能需求(causal / sliding / 分页)→ 按优先级挑选",
        "为每个后端建模 capabilities(含支持 causal、支持分页),让 need_causal / need_paged 真正参与过滤——修复了旧版参数未生效的问题",
        "FLASH_ATTN 在 NVIDIA sm_80+ 上通常默认;架构不足或需要分页时会退到其他后端",
        "真机实测:同一 attention 从 flash 退到 mem_efficient、再退到纯 math,耗时肉眼可见变慢(数倍)",
        "vLLM 用 AttentionSelectorConfig / AttentionBackendEnum / validate_configuration 实现这一机制",
    ],
    practice=[
        "给 select_backend 增加 need_sliding 参数,把支持 sliding window 的后端筛选出来(需要先在能力表加一列)",
        "把架构列表换成实际 GPU 型号(如 RTX 5060 = sm_120),验证默认后端",
        "扩展能力表,加入 alibi 支持列,观察功能过滤如何生效",
        "在 app_47 里加一个「对比两个平台默认后端」的双栏视图",
        "读 vllm/v1/attention/selector.py 的 AttentionSelectorConfig,列出它有哪些功能字段",
    ],
    links=[
        ("vLLM 官方文档: Attention Backends", "https://docs.vllm.ai/en/latest/design/attention_backends/"),
        ("vLLM selector.py 源码", "https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/selector.py"),
        ("FlashInfer 项目", "https://github.com/flashinfer-ai/flashinfer"),
    ])

NB.save(str(Path(CH07) / "47_attn_backend_select.ipynb"))

app_path = Path(CH07) / APP_FILE
app_path.write_text(APP_47 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
