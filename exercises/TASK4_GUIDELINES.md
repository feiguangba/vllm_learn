# 任务 4 补充规范(Triton / AI 编译器 / 华为生态章节专用)

在原有 GUIDELINES.md 基础上,本批新章节额外要求:

## 绘图风格(重要)
- **静态图优先用 matplotlib + seaborn**(AAAI/CVPR/ICML 会议论文风格):浅色网格、简洁、大图。
  推荐统一样式头:
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set_theme(style="whitegrid", font=["DejaVu Sans"])
  plt.rcParams["figure.dpi"] = 120
  plt.rcParams["savefig.dpi"] = 200
  ```
- **动态/交互图才用 plotly 和 pyecharts**(需要拖动参数、悬停、切换的场合)。
- 每课尽量多画图:结构图(annotate)、曲线图、热力图、对比柱状图、示意图都要有。
- 参考风格: D:\Project\24-Beauty-of-Data-Visualization\Book2_Beauty-of-Data-Visualization(鸢尾花书)

## 每课结构(与前 50 章一致,勿变太多)
封面 → chapter_cover(目标/目录/链接) → 正文 6-12 cell(中文 1500-4000 字,直觉比喻→图示/代码→公式结论,emoji/链接丰富) → (涉及交互的课)%%writefile app cell → wrapup(小结/练习/链接)

## 涉及交互才配 streamlit app
- 只有"需要调参观察变化"的课才配 app(如 GEMM tile 调优、融合开关对比)。
- 不配 app 的课:结尾 wrapup 正常写,不要加 app。

## 环境
- Python = D:\uv_envs\uv_cuda\Scripts\python.exe
- **triton 3.7.1 已安装且可用**(可真实写 triton kernel 并运行, torch GPU 可用 RTX 5060)
- seaborn 已安装
- 第一个 code cell 固定 KMP 保护 cell
- cell source 单字符串(nb_builder 已处理)

## 避坑(继续有效)
- plotly:`pio.renderers.default = "notebook"` + 最后 `fig`;pyecharts:`render_notebook()`
- f-string 禁止跨行;Windows 路径正斜杠;中文 UTF-8 无 U+FFFD;单 cell < 20s
- 生成脚本放 `_build/helpers.py`(共享组件)+ `build_NN.py`,用 nb_builder 的 Notebook/chapter_cover/wrapup

## 验证(每课必须)
1. 运行 build_NN.py 生成 ipynb(+ app 如有)
2. nbconvert:`$env:KMP_DUPLICATE_LIB_OK="TRUE"; & "D:\uv_envs\uv_cuda\Scripts\python.exe" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=uv_cuda --ExecutePreprocessor.timeout=180 <ipynb>`
   若遇间歇性 "Kernel died" 竞态,改用备选:exec 逐 cell 执行(跳过 %%writefile、替换 .show()/.render_notebook() 为 None),逻辑全过即算通过并注明
3. app(如有):py_compile + headless 启动 5 秒验证
最后回复:每个 ipynb(+app)路径、验证结果(Pass/Fail + 方式)、每课一句话介绍。