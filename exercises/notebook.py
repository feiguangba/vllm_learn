import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # 🎮 marimo 演示剧本

        用真实可交互的例子，演示 marimo 反应式 notebook 四大能力：

        1. **交互控件** — slider / dropdown / checkbox / text ...
        2. **数据可视化** — matplotlib 图形随控件实时重绘
        3. **markdown 富文本** — 公式、排版、折叠、提示框
        4. **状态与响应** — 依赖自动重算、DataTable、实时反馈

        > **记住 marimo 的第一定律**: *修改任一控件 → 所有依赖它的 cell 自动重跑* 🌟
        """
    )
    mo.callout("试着拖动或点击下面任何一个控件，看看旁边发生了什么。", kind="success")
    return


@app.cell
def _(mo):
    mo.md("""
    ## ① 交互控件
    """)
    return


@app.cell
def _(mo):
    # 每个 mo.ui.xxx 都会生成一个“元素”, 必须 return 出去才能被其他 cell 使用
    slider = mo.ui.slider(1, 100, value=50, label="📏 滑块")
    dropdown = mo.ui.dropdown(
        {"⚡ 正弦": "sin", "🌊 余弦": "cos", "🪂 抛物线": "para"}, value="⚡ 正弦", label="📌 下拉"
    )
    checkbox = mo.ui.checkbox(value=True, label="✅ 是否开启")
    text = mo.ui.text(value="marimo 真棒", label="🔤 文本输入")
    number = mo.ui.number(0, 100, value=10, label="🔢 数字输入")
    return checkbox, dropdown, number, slider, text


@app.cell
def _(checkbox, dropdown, mo, number, slider, text):
    # 把控件摆成一行展示; 下面那行 md 随控件值实时更新 —— 这就是“响应”
    mo.hstack([slider, dropdown, checkbox, text, number])
    mo.md(
        f"`slider={slider.value} · dropdown={dropdown.value} · 开启={checkbox.value} "
        f"· text={text.value} · number={number.value}`"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## ② 数据可视化
    """)
    return


@app.cell
def _(mo):
    # 用控件驱动绘图
    ftype = mo.ui.dropdown(
        {"正弦": "sin", "余弦": "cos", "抛物线": "para"}, value="正弦", label="📊 函数类型"
    )
    npts = mo.ui.slider(10, 200, value=60, label="🔍 采样点数")
    return ftype, npts


@app.cell
def _(ftype, mo, np, npts, plt):
    x = np.linspace(0, 2 * np.pi, npts.value)
    if ftype.value == "sin":
        y = np.sin(x)
    elif ftype.value == "cos":
        y = np.cos(x)
    else:
        y = x**2 / (2 * np.pi)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, y, lw=2, color="#2f6fd6", marker="o", ms=3)
    ax.set_title(f"{ftype.value} · 采样 {npts.value} 点")
    ax.grid(alpha=0.3)

    fig  # 引用 fig 让图形显示在 cell 输出里
    mo.md(f"> 📈 统计: 均值 `{y.mean():.3f}` · 峰值 `{y.max():.3f}`")
    return (y,)


@app.cell
def _(ftype, mo, y):
    # 依赖 y 与 ftype: 下拉一变, 这里的计算结果跟着更新 —— 反应式链路
    mo.md(f"函数 `{ftype.value}` 的第一个采样点 `y[0]={y[0]:.3f}`")
    return


@app.cell
def _(mo):
    mo.md("""
    ## ③ markdown / 富文本
    """)
    return


@app.cell
def _(mo):
    # mo.md 支持公式、数学环境、代码段、列表、表格
    mo.md(r"""
    ### 公式渲染

    欧拉公式: $e^{i\pi} + 1 = 0$

    块级公式:
    $$
    f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$

    ```python
    def hello():
        return "这是高亮代码块"
    ```

    - 支持 markdown 列表、**加粗**、*斜体*

    | 列 A | 列 B |
    |---|---|
    | 1 | 2 |
    """)
    return


@app.cell
def _(mo):
    # 折叠面板(0.24+:传 dict,标题→内容)+ 提示框 + 结束语
    acc = mo.accordion(
        {
            "🌀 折叠面板 — 点击展开": mo.md(
                "### 展开的内容\n\n这里可以放很长的说明，保持页面整洁。"
            ),
        }
    )
    mo.vstack(
        [
            acc,
            mo.callout("这是一个 `danger` 级别提示框", kind="danger"),
        ]
    )
    mo.md("**演示完成 ✔**")
    return


@app.cell
def _(mo):
    mo.md("""
    ## ④ 状态与响应
    """)
    return


@app.cell
def _(mo, np, pd):
    # DataTable: 可排序/筛选的交互表格, table.value 暴露被选中的行
    df = pd.DataFrame(
        {
            "编号": np.arange(1, 21),
            "随机数据": np.random.uniform(0, 100, 20).round(2),
            "类型": np.random.choice(["A", "B", "C"], 20),
        }
    )
    table = mo.ui.dataframe(df)
    return df, table


@app.cell
def _(mo, table):
    mo.hstack([table, mo.md(f"> 选中了 `{len(table.value)}` 行")])
    return


@app.cell
def _(df, mo):
    # 纯“推导”cell: 依赖 DataFrame, 展示分组聚合结果
    agg = df.groupby("类型")["随机数据"].mean().round(2)
    mo.md(
        f"### 按类型求均值\n\n`{agg.to_string()}`\n\n"
        f"总行数: `{len(df)}` · 总和: `{df['随机数据'].sum():.2f}`"
    )
    return


@app.cell
def _(mo):
    # 开关控件: 仅作交互示例
    mo.ui.switch(value=True, label="🏷 切换开关")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
