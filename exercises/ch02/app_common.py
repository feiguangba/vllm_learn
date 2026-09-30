# -*- coding: utf-8 -*-
"""chXX streamlit 共享组件。

统一提供:
- :func:`terminology`  :渲染一个「专业术语卡」,增强讲解深度(前因后果已在正文铺陈)。
- :func:`real_badge`   :标注当前数据来自真实 GPU 算子实测。
- :func:`foot_note`    :统一页脚。

设计点:真实数据一律集中在各章的 ``real_ops.py``(GPU/CUDA 微基准 + 逻辑仿真),
app 只负责可视化与讲解,不内嵌拍脑袋曲线。
"""
import os
import sys

import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def terminology(items, expand_label="🏷️ 专业术语卡(翻译 · 出处") -> None:
    """渲染「专业术语」卡片。

    Parameters
    ----------
    items : iterable[(术语, 定义, 出处)]  出处可省略。
    """
    with st.expander(expand_label + ")", expanded=False):
        for row in items:
            term, val = row[0], row[1]
            src = row[2] if len(row) > 2 else None
            if src:
                st.markdown(f"**{term}** — {val} · 🔗 {src}")
            else:
                st.markdown(f"**{term}** — {val}")


def real_badge(real: bool = True) -> None:
    st.caption("📡 **数据来源**: "
               + ("真实 GPU 算子实测(CUDA 微基准),非理想曲线。"
                  if real else "逻辑层模拟(参数贴近工业默认值)。"))


def foot_note() -> None:
    st.divider()
    st.caption(
        "© minivllm · 图解 vLLM 推理引擎。参考:PagedAttention (SOSP'23) · "
        "vLLM 官方文档 · FlashAttention · Orca。")


# key 不可解释时的资料来源(供 terminlogy 引用)
SRC = {
    "vllm_docs": "https://docs.vllm.ai",
    "pagedattn": "https://arxiv.org/abs/2309.06180",
    "flashattn": "https://arxiv.org/abs/2205.14135",
    "orca": "https://arxiv.org/abs/2208.14217",
    "llama": "https://arxiv.org/abs/2307.09288",
    "bpe": "https://arxiv.org/abs/1508.07909",
}