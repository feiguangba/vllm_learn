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
