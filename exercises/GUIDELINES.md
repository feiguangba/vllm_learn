# VLLM_learn 练习册写作指南(给章节作者的统一规范)

## 总风格(参照《鸢尾花书》数据科学系列)
- 中文写作,口语化、亲切、像一位老师手把手带着学
- 每一课(一个 ipynb):封面 → 学习目标 → 正文章节(讲+码结合)→ 小结 → 练习 → 参考链接
- **多使用 emoji**:🎯 目标 / 💡 直觉 / ⚠️ 坑 / ✅ 验证 / 🔍 深入 / 📊 图表 / 🚀 性能 / 🏷️ 术语 / 🎨 图示 / 🔗 链接
- **多使用超链接**:vLLM 官方文档 https://docs.vllm.ai 、论文(arXiv)、PyTorch 文档、博客
- **形象例子**:用生活化比喻解释概念(餐厅、银行、快递、图书馆、工厂流水线等)
- 每个 notebook 至少:1 个 pyecharts 交互图(render_notebook)+ 1 个 plotly 图(px/go),以及大量文字讲解
- **每课必须配套 1 个 streamlit 动态演示**:生成 `app_XX_短名.py`(同目录),要求:
  - 中文 UI,标题含 emoji,顶部 `st.set_page_config(page_title=..., layout="wide")`
  - 至少 3 个交互控件(slider / selectbox / number_input / radio / multiselect)
  - 用 plotly(px/go)或 pyecharts 渲染**动态可交互**图表,图表随控件参数实时变化
  - 有 `st.markdown` 讲解文字、`st.caption` 说明、`st.metric` 展示关键指标
  - 单文件可独立运行: `streamlit run app_XX_短名.py`
  - notebook 中必须展示该 app 的完整代码 cell(可直接复制运行)+ `st.code`/说明文字 + "运行方法"小节(st.reamlit run 命令、浏览器地址 http://localhost:8501)
  - streamlit 文件用 `D:\uv_envs\uv_cuda\Scripts\python.exe -m py_compile` 验证语法通过,并尝试 `streamlit run --server.headless true --server.port 8599` 短暂启动(3-5 秒后终止)确认无启动异常
- 代码全部可执行,控制单 cell 运行时间 < 30s;GPU 可用(torch.cuda.is_available() 为 True,RTX 5060,sm_120)

## 工具:必须用 nb_builder.py 生成 ipynb
```python
import sys; sys.path.insert(0, r"D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises")
from nb_builder import Notebook, chapter_cover, wrapup
nb = Notebook("第 08 课 · KV Cache 的内存账本", subtitle="...", emoji="🧮", chapter="第 2 章 · KV Cache 与 PagedAttention")
chapter_cover(nb, objectives=[...], toc=[(标题, 描述), ...], links=[("vLLM 文档", "https://docs.vllm.ai"), ...])
nb.md("## 1. ...", "...")
nb.code("...code...", "说明文字")
wrapup(nb, summary=[...], practice=[...], links=[...])
nb.save(r"D:\...\exercises\ch02\08_kv_cache_memory.ipynb")
```
- 代码块要**短小清晰**,注释中文;涉及可视化先 %matplotlib inline 或直接 pyecharts render_notebook / plotly.io.show
- pyecharts: `from pyecharts.charts import Bar, Line, Pie, Scatter, HeatMap, Tree, Gantt...; from pyecharts import options as opts`;在 notebook 里调用 `chart.render_notebook()` 即可显示
- plotly: `import plotly.express as px; import plotly.graph_objects as go`;用 `fig.show()`(需要 plotly 的 notebook 渲染,一般 ipykernel 下直接可用;若不行,用 `import plotly.io as pio; pio.renderers.default = "notebook"`)
- streamlit 文件放同目录,命名 `app_XX_描述.py`

## 内容要求(细致入微)
- 每个 ipynb 15-30 个 cell,讲解文字总量 1500-4000 字
- 概念解释至少 3 层:直觉比喻 → 图示/代码 → 数学/公式(可手写 LaTeX markdown)
- 每个数值结论都要有代码验证(输出数字),不要只写结论
- 每课结尾的练习要有可操作性(改参数、加功能、写小函数)

## 验证要求(必须做)
生成后对每个 notebook 用 uv_cuda 的 python 执行验证:
```
D:\uv_envs\uv_cuda\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace <文件>
```
每个 streamlit app 用 `D:\uv_envs\uv_cuda\Scripts\python.exe -m py_compile` 验证语法 + 短暂 headless 启动确认无异常。
若某 cell 因 notebook 渲染问题失败,调整为安全写法(如用 pyecharts render_notebook 前先 `from pyecharts.globals import CurrentConfig`)。执行成功后才能交付。

## 章节划分与文件命名
目录: D:\Project\21-Cpp_learn\explore\VLLM_learn\exercises\ch01 .. ch08
文件: NN_短名.ipynb (NN 为两位编号,全书连续编号 01-50)