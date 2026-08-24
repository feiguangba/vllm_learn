# -*- coding: utf-8 -*-
"""生成 94_ms_pytorch_migrate.ipynb 与 app_94_migrate.py"""
from helpers import D, STYLE, chapter_cover, wrapup, new_nb, CH11, app_cell, finalize
from pathlib import Path

APP_94 = D('''
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
     "example": "class Net(ms.nn.Cell):\\n    def construct(self, x): return self.fc(x)"},
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
''')

NB = new_nb("第 94 课 · MindSpore ↔ PyTorch 迁移:API 对照与踩坑",
            subtitle="API 对照表 · 四步迁移法 · GRAPH/PYNATIVE 模式 · 权重名映射",
            emoji="🔄")

chapter_cover(NB,
    objectives=[
        "建立 torch → MindSpore 的 API 对照表:模型、张量、优化器、数据、设备上下文",
        "掌握四步迁移法:设备上下文 → 网络定义 → 训练循环 → 数据管道",
        "理解 construct / set_train / 全局 context 等 MindSpore 独有约定",
        "学会权重迁移:torch state_dict → MindSpore 参数 name 的映射规则",
        "看清 GRAPH_MODE 与 PYNATIVE_MODE 的差异,以及动态 shape 等经典坑",
        "用 torch 模拟 MindSpore 风格网络,验证前向与训练行为一致,配 App 对照查询",
    ],
    toc=[
        ("直觉:两座说方言的城", "API 不同名,思想同源"),
        ("核心对照表", "模型/优化器/数据/设备一张表"),
        ("网络定义对照", "用 torch 模拟 MindSpore 风格 Cell"),
        ("四步迁移法", "设备 → 网络 → 循环 → 数据"),
        ("权重迁移与命名", "state_dict ↔ .ckpt 的 key 映射"),
        ("踩坑清单", "GRAPH 模式、动态 shape、BN、广播"),
        ("配套 App", "app_94_migrate.py:API 对照查询"),
    ],
    links=[
        ("MindSpore 官方文档", "https://www.mindspore.cn/docs"),
        ("MindSpore 教程(与 PyTorch 对照)", "https://www.mindspore.cn/tutorials"),
        ("PyTorch 文档", "https://pytorch.org/docs/stable/index.html"),
        ("MindSpore 迁移指南", "https://www.mindspore.cn/migration"),
    ])

NB.code(STYLE, "🧊 本课开篇:KMP 保护 + 会议论文风格绘图头。")

NB.md("## 1️⃣ 直觉:两座说方言的城 🏙️",
D('''
PyTorch 与 MindSpore 的关系,就像两座**说不同方言的城**:城市规划(自动微分、计算图、张量)
完全同源,但路牌(API)写法不同:

- PyTorch 叫 **forward**,MindSpore 叫 **construct**;
- PyTorch 用 `model.to("cuda")` 换设备,MindSpore 用全局 `set_context(device_target="Ascend")`;
- PyTorch 的 `state_dict` 的 key 是参数路径,MindSpore 的检查点按参数 **name** 存取。

迁移的本质不是"重写思想",而是"翻译方言"。把对照表背熟,剩下的就是工程细节。
本课给出一份核心对照表 + 四步迁移法,并用 torch 模拟 MindSpore 风格的写法,验证两者行为一致。
'''))

NB.md("## 2️⃣ 核心对照表:一张表看懂 📋",
D('''
先建立最重要的几组对应关系(生产迁移中 90% 的代码都在这几组里):

| 功能 | PyTorch | MindSpore |
|---|---|---|
| 模型基类 | `torch.nn.Module` | `mindspore.nn.Cell` |
| 前向函数 | `forward(x)` | `construct(x)` |
| 全连接 | `nn.Linear` | `nn.Dense` |
| 可训练参数 | `model.parameters()` | `model.trainable_params()` |
| 优化器 | `torch.optim.Adam` | `ms.nn.optim.Adam` |
| 数据管道 | `torch.utils.data` | `mindspore.dataset` |
| 设备切换 | `model.to("cuda")` | `ms.set_context(device_target="Ascend")` |
| 训练/推理模式 | `model.train()/eval()` | `model.set_train(True/False)` |
| 检查点 | `torch.save(state_dict)` | `ms.save_checkpoint(net)` |
| 推理无梯度 | `torch.no_grad()` | 图模式天然无梯度 / `set_grad()` |
'''))

NB.code(D('''
mapping = {
    "torch.nn.Module": "mindspore.nn.Cell",
    "forward(x)": "construct(x)",
    "nn.Linear": "nn.Dense",
    "model.parameters()": "model.trainable_params()",
    "torch.optim.Adam": "ms.nn.optim.Adam",
    "torch.utils.data": "mindspore.dataset",
    "model.to('cuda')": "ms.set_context(device_target='Ascend')",
    "torch.save/load": "ms.save_checkpoint / load_checkpoint",
}
print("核心对照(按使用频率排序):")
for i, (t, m) in enumerate(mapping.items(), 1):
    print(f"  {i:2d}. {t:24s} → {m}")
print(f"\\n共 {len(mapping)} 组核心 API,覆盖日常迁移 90% 的替换点。")
'''),
"📋 对照表落地:打印出来贴屏幕,迁移时逐行翻译即可。")

NB.md("## 3️⃣ 网络定义:forward 变 construct 🧱",
D('''
把 PyTorch 的 `nn.Module` 翻译成 MindSpore 的 `nn.Cell`,要点是:

1. 基类换成 `Cell`;
2. 前向 `forward` 改叫 `construct`;
3. 参数用 `ms.Parameter` 显式创建,并带 **name**(检查点存取靠它)。

我们在 torch 里**模拟**一个 MindSpore 风格网络(一样的初始化),与标准 torch 网络对比前向输出,
证明"只是写法不同,数值应该一致":
'''))

NB.code(D('''
import torch.nn as nn
import torch.nn.functional as F

class TorchMLP(nn.Module):
    """标准 PyTorch 写法"""
    def __init__(self, din=8, dmid=16, dout=4):
        super().__init__()
        self.fc1 = nn.Linear(din, dmid)
        self.fc2 = nn.Linear(dmid, dout)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class MsStyleMLP:
    """MindSpore 风格(用 torch 模拟):construct + 参数字典(按 name 存取,对应 parameters_dict)"""
    def __init__(self, din=8, dmid=16, dout=4):
        self._params = {
            "fc1.weight": torch.nn.Parameter(torch.randn(din, dmid) * 0.02),
            "fc1.bias":   torch.nn.Parameter(torch.zeros(dmid)),
            "fc2.weight": torch.nn.Parameter(torch.randn(dmid, dout) * 0.02),
            "fc2.bias":   torch.nn.Parameter(torch.zeros(dout)),
        }
    def named_parameters(self):                      # MindSpore 的 parameters_dict()
        return list(self._params.items())
    def construct(self, x):                          # MindSpore 用 construct
        h = F.relu(torch.mm(x, self._params["fc1.weight"]) + self._params["fc1.bias"])
        return torch.mm(h, self._params["fc2.weight"]) + self._params["fc2.bias"]

torch.manual_seed(1)
x = torch.randn(4, 8)
p = TorchMLP(); m = MsStyleMLP()
# 让两套权重完全一致(用 torch 模型的权重喂给 MindSpore 风格)
with torch.no_grad():
    m._params["fc1.weight"].copy_(p.fc1.weight.T); m._params["fc1.bias"].copy_(p.fc1.bias)
    m._params["fc2.weight"].copy_(p.fc2.weight.T); m._params["fc2.bias"].copy_(p.fc2.bias)

with torch.no_grad():
    y_t = p(x); y_m = m.construct(x)
diff = (y_t - y_m).abs().max().item()
print("torch forward 与 mindspore-style construct 的最大输出差:", f"{diff:.2e}")
print("→ 两种写法数值一致,差的只是『方言』。")
'''),
"✅ 关键结论:forward ↔ construct 只是名字不同,数学完全一样 —— 迁移时逐层替换 API,再用这个办法逐层验证输出。")

NB.md("## 4️⃣ 四步迁移法 🧭",
D('''
实战迁移按四步走,每步都有明确的验证点:

1. **设备上下文**:`ms.set_context(device_target="Ascend")` 设置运行设备(替代 `to("cuda")`);
2. **网络定义**:把 `Module` 换成 `Cell`,`forward` 换 `construct`,逐层替换算子;
3. **训练循环**:优化器接 `trainable_params()`,`set_train(True)` 切训练态;梯度反传骨架不变;
4. **数据管道**:把 `DataLoader` 换成 `mindspore.dataset` 的各类 Dataset + `batch()`。

下面模拟第 3 步:用一个极小的线性回归,分别用 torch 与 MindSpore 风格跑 50 步 SGD,
比较损失曲线是否一致:
'''))

NB.code(D('''
torch.manual_seed(0)
Xs = torch.randn(64, 4); Wt = torch.randn(4, 1); Ys = Xs @ Wt + 0.1 * torch.randn(64, 1)

def run_torch_loop():
    lin = nn.Linear(4, 1)
    opt = torch.optim.SGD(lin.parameters(), lr=0.05)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = F.mse_loss(lin(Xs), Ys)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

def run_msstyle_loop():
    w = torch.nn.Parameter(torch.zeros(4, 1))
    b = torch.nn.Parameter(torch.zeros(1))
    lr = torch.tensor(0.05)
    losses = []
    for _ in range(50):
        pred = Xs @ w + b
        loss = F.mse_loss(pred, Ys)
        g_w = 2 * (Xs.T @ (pred - Ys)) / len(Ys)   # 手写梯度(模拟 MindSpore 的自动微分结果)
        g_b = 2 * (pred - Ys).mean()
        with torch.no_grad():
            w -= lr * g_w.reshape_as(w); b -= lr * g_b
        losses.append(loss.item())
    return losses

l1 = run_torch_loop(); l2 = run_msstyle_loop()
print(f"50 步后 torch 损失 = {l1[-1]:.4f} | mindspore-style = {l2[-1]:.4f}")
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(l1, "o-", ms=3, color="#2e86c1", label="torch 训练循环")
ax.plot(l2, "s-", ms=3, color="#27ae60", label="mindspore-style 循环")
ax.set_xlabel("step"); ax.set_ylabel("MSE loss")
ax.set_title("两条训练循环收敛一致:骨架是相通的")
ax.legend(); plt.tight_layout()
'''),
"📈 训练循环:优化器 + 梯度下降的骨架在两边完全一致 —— 迁移时这一块通常改动最小。")

NB.md("## 5️⃣ 权重迁移:key 与 name 的规矩 🏷️",
D('''
权重迁移是最容易出 bug 的一环:PyTorch 的 `state_dict` 用**模块路径**当 key
(`fc1.weight`),而 MindSpore 检查点按参数的 **name** 存取。两者约定不同,要建映射:

- PyTorch: `fc1.weight`、`fc1.bias`、`blocks.0.attn.qkv.weight`;
- MindSpore: 参数在定义时取的名,常为 `fc1.weight`、`dense.weight` 等(也可能带前缀)。

真机上用 MindSpore 官方迁移工具/脚本自动转换。这里演示映射的本质 —— 只改名、不搬值:
'''))

NB.code(D('''
torch_state = {"fc1.weight": torch.randn(16, 8), "fc1.bias": torch.randn(16)}
# 迁移脚本常做三件事:改名、转置(Linear 权重布局差异)、改 dtype
ms_ckpt = {}
for k, v in torch_state.items():
    ms_ckpt["net." + k] = v.T if "weight" in k else v     # 模拟布局转换
print("映射结果:")
for k in ms_ckpt:
    print(f"  {k:22s} shape={tuple(ms_ckpt[k].shape)} dtype={ms_ckpt[k].dtype}")

fig, ax = plt.subplots(figsize=(6.5, 3.2))
keys = list(ms_ckpt)
vals = [ms_ckpt[k].numel() for k in keys]
ax.bar(keys, vals, color=["#2e86c1", "#27ae60"])
for b, v in zip(ax.patches, vals):
    ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha="center", fontsize=9)
ax.set_ylabel("元素个数")
ax.set_title("权重迁移:key 改名 + 布局转换,数值原样搬移")
plt.tight_layout()
'''),
"🏷️ 权重迁移的本质:『改名 + 布局转换』,不是重新训练。转换后用『加载前后输出对比』做验收,和 91 课的精度对齐呼应。")

NB.md("## 6️⃣ 踩坑清单:那些让你熬夜的坑 ⚠️",
D('''
迁移最常见的坑,按出现频率排:

1. **GRAPH_MODE 动态 shape**:MindSpore 静态图模式下,维度不固定的输入(如文本变长)会编译失败
   —— 解决:固定/填充 shape,或用 PYNATIVE_MODE(动态图,性能略低);
2. **自动微分细节差异**:`loss.backward()` 后的梯度语义基本一致,但自定义反向传播、
   in-place 操作、叶子张量的处理在两边规则不同,容易出现"梯度对不上";
3. **参数命名与初始化**:MindSpore 参数初始化方式(He/Xavier)与 torch 不同,直接加载权重前
   要保证布局一致(转置),否则精度莫名崩掉;
4. **BatchNorm 差异**:BN 的统计量更新时机与 `use_batch_statistics` 行为不同,推理前要
   确认 running_mean/var 已冻结;
5. **广播与算子语义**:个别算子的广播规则、取整方向有细微差异,逐层对比输出才能定位。

把"逐层输出对比"养成习惯 —— 每迁一层,喂同样的输入,比较输出差(就像第 3 节那样):
'''))

NB.code(D('''
pits = ["动态 shape", "梯度语义", "参数命名/布局", "BatchNorm 统计", "广播语义"]
freq = [5, 4, 4, 3, 2]       # 踩坑频率(示意)
fig, ax = plt.subplots(figsize=(7.5, 4))
bars = ax.barh(pits, freq, color=["#c0392b", "#e67e22", "#f39c12", "#2e86c1", "#27ae60"])
for b, f in zip(bars, freq):
    ax.text(f + 0.05, b.get_y()+b.get_height()/2, f"×{f}", va="center", fontsize=10)
ax.set_xlabel("踩坑频率(示意)")
ax.set_title("MindSpore 迁移经典坑:动态 shape 排第一")
plt.tight_layout()
'''),
"⚠️ 按频率排好优先级:先解决『动态 shape』,再逐个消灭其他坑 —— 建议每踩一个坑就记一条笔记,迁移完成时就是你的专属手册。")

NB.md("## 7️⃣ 配套 App:API 对照查询 🎛️",
D('''
运行同目录的 `app_94_migrate.py`,**下拉选择 PyTorch API**,立即看到 MindSpore 对应写法、
注意事项与最小示例;也可以搜索关键词、展开完整对照表:

```bash
D:\\uv_envs\\uv_cuda\\Scripts\\python.exe -m streamlit run app_94_migrate.py
```

浏览器打开 **http://localhost:8501**(也可 `--server.port 8694`)。完整源码如下:
'''))

NB.code(app_cell("app_94_migrate.py", APP_94),
"📜 运行本 cell 会覆盖写入 `app_94_migrate.py`,保证 notebook 与 app 始终一致。")

wrapup(NB,
    summary=[
        "核心对照:Module↔Cell、forward↔construct、optim↔nn.optim、DataLoader↔mindspore.dataset",
        "四步迁移法:设备上下文 → 网络定义 → 训练循环 → 数据管道,每步都有验证点",
        "网络写法只是方言不同:construct 与 forward 数学完全一致,可逐层对比输出",
        "权重迁移 = key 改名 + 布局转换,用『加载前后输出对比』验收",
        "踩坑优先级:动态 shape(GRAPH 模式)、梯度语义、参数布局、BN 统计、广播语义",
    ],
    practice=[
        "把第 3 节 MsStyleMLP 改成三层的,并逐层验证与 torch 版本输出一致",
        "给第 4 节训练循环换成 Adam 优化器,对比两条损失曲线是否仍一致",
        "为第 5 节增加一个 Embedding 权重的映射规则,处理 padding_idx 的特殊性",
        "读 MindSpore 迁移指南的『动态 shape』一节,总结 3 种规避方案并各写一句话",
    ],
    links=[
        ("MindSpore 官方文档", "https://www.mindspore.cn/docs"),
        ("MindSpore 与 PyTorch 对照教程", "https://www.mindspore.cn/tutorials"),
        ("MindSpore 模型迁移指南", "https://www.mindspore.cn/migration"),
        ("PyTorch 官方文档", "https://pytorch.org/docs/stable/index.html"),
    ])

out = str(Path(CH11) / "94_ms_pytorch_migrate.ipynb")
NB.save(out)
finalize(out)

app_path = Path(CH11) / "app_94_migrate.py"
app_path.write_text(APP_94 + "\n", encoding="utf-8")
print(f"[ok] {app_path}")
