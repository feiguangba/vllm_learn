# -*- coding: utf-8 -*-
# app_94_migrate.py — MindSpore ↔ PyTorch API 对照查询 🔄
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="MindSpore ↔ PyTorch 迁移 🔄", layout="wide")
st.title("🔄 第 94 课 · MindSpore ↔ PyTorch 迁移:API 对照查询")

st.markdown("""
PyTorch 和 MindSpore 是**两座说着不同方言的城** —— 结构相似(张量、计算图、自动微分),
拼写不同(API 名)。迁移的第一步就是把 API"翻译"过来。下方选择要查的 API,
看它的 MindSpore 对应写法、注意事项与一个最小示例。
""")

API_MAP = [
    {"torch": "torch.nn.Module", "ms": "mindspore.nn.Cell",
     "note": "模型基类;MindSpore 前向函数叫 construct(不是 forward)",
     "example": "class Net(ms.nn.Cell):\n    def construct(self, x): return self.fc(x)"},
    {"torch": "torch.nn.Linear", "ms": "mindspore.nn.Dense",
     "note": "全连接层;注意参数顺序与初始化细节",
     "example": "self.fc = ms.nn.Dense(768, 512)"},
    {"torch": "model.parameters()", "ms": "model.trainable_params()",
     "note": "取可训练参数列表,喂给优化器",
     "example": "optimizer = ms.nn.optim.Adam(model.trainable_params())"},
    {"torch": "torch.optim.Adam", "ms": "mindspore.nn.optim.Adam",
     "note": "优化器放在 nn.optim 下,需传 trainable_params()",
     "example": "opt = ms.nn.optim.Adam(net.trainable_params(), lr=1e-4)"},
    {"torch": "torch.utils.data.DataLoader", "ms": "mindspore.dataset",
     "note": "数据管道差异最大,MindSpore 用 GeneratorDataset/MnistDataset 等",
     "example": "ds = ms.dataset.MnistDataset('data').batch(32)"},
    {"torch": "torch.no_grad()", "ms": "ms.set_train(False) / model.set_grad()",
     "note": "MindSpore 靠 set_train/set_grad 切换训练/推理上下文",
     "example": "net.set_train(False)   # 推理模式"},
    {"torch": "model.to('cuda')", "ms": "ms.set_context(device_target='Ascend')",
     "note": "设备用全局上下文设置,而不是 model.to()",
     "example": "ms.set_context(device_target='Ascend', device_id=0)"},
    {"torch": "torch.save(state_dict)", "ms": "ms.save_checkpoint(net)",
     "note": "检查点格式不同(.ckpt);权重名以参数 name 为准",
     "example": "ms.save_checkpoint(net, 'net.ckpt')"},
    {"torch": "loss.backward(); opt.step()", "ms": "loss.backward(); opt.step()",
     "note": "训练循环骨架一致;配合 model.set_train(True) 使用",
     "example": "loss.backward(); opt.step(); opt.clear_grad()"},
]

st.sidebar.header("🎛️ 参数")
q = st.sidebar.selectbox("查询一个 PyTorch API", [r["torch"] for r in API_MAP])
kw = st.sidebar.text_input("关键词搜索", "")
show_all = st.sidebar.checkbox("显示完整对照表", value=True)
st.sidebar.caption("MindSpore 用 context 控制设备、用 set_train 控制模式 —— 这两点和 PyTorch 最不一样。")

row = next(r for r in API_MAP if r["torch"] == q)
st.subheader(f"🔍 当前查询: {row['torch']}")
st.markdown(f"**MindSpore 对应**: `{row['ms']}`")
st.info(f"**注意事项**: {row['note']}")
st.code(row["example"], language="python")

c1, c2, c3 = st.columns(3)
c1.metric("对照条目数", len(API_MAP))
c2.metric("当前是否一致命名", "基本一致" if row["torch"].split(".")[-1] in row["ms"] else "差异较大")
c3.metric("迁移难度(1-5)", min(5, 2 + sum(1 for ch in row["note"] if ch in "差注")))

if show_all:
    st.subheader("📋 完整对照表")
    df = pd.DataFrame(API_MAP)[["torch", "ms", "note"]]
    df.columns = ["PyTorch API", "MindSpore API", "注意事项"]
    if kw:
        df = df[df.apply(lambda r: kw in str(r["PyTorch API"]) or kw in str(r["MindSpore API"]), axis=1)]
    st.dataframe(df, use_container_width=True)

st.subheader("📊 迁移工作量分布(示意)")
fig = go.Figure(go.Bar(
    x=["网络定义", "训练循环", "数据管道", "权重转换", "设备/上下文"],
    y=[3, 2, 5, 2, 3], text=[3, 2, 5, 2, 3], textposition="outside",
    marker_color=["#2e86c1", "#27ae60", "#e74c3c", "#f39c12", "#8e44ad"]))
fig.update_layout(title="迁移踩坑/工作量分布:数据管道最难", yaxis_title="相对工作量",
                  height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> 💡 **结论**:API 对照只是第一步,真正的坑在**数据管道**和**图模式**上。
> 迁移策略:小模型先对照表翻译 → 逐模块验证输出 → 全量对齐精度 → 再上昇腾硬件。
""")
