# 任务 5 Review 增强规范

## 目标
对既有 50 章(ch01-ch08)做 review 增强:
1. **细化实现细节**:补充公式推导、中间张量形状、边界条件、数值细节、代码注释
2. **多画 matplotlib/seaborn 静态图**(AAAI/CVPR/ICML 会议风格):分布图/曲线/热力图/对比柱状/误差棒/示意图
3. **动态交互才用 plotly/pyecharts**(已有保留即可)
4. 风格参考:D:\Project\24-Beauty-of-Data-Visualization\Book2_Beauty-of-Data-Visualization(鸢尾花书)

## 会议风格模板(新增 matplotlib cell 时用)
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

## 操作方式
- 直接修改 ipynb 的 cell(用 python 脚本读取/修改 JSON,source 用单字符串)
- 优先**新增** matplotlib/seaborn 图 cell(插在对应代码 cell 之后),保留原有内容
- 也可在关键代码 cell 的 markdown 前置补充"实现细节"讲解
- **不要删除**原有 plotly/pyecharts 交互图(它们是动态展示)
- 改完必须执行验证:每个 notebook 用 nbconvert 跑通(设 KMP_DUPLICATE_LIB_OK)

## 每章重点增强(选择 3 个代表 notebook 深改,其余轻改或不改)
选最有教学代表性的:如 ch01 的 02/03/05, ch02 的 07/08/10, ch03 的 15/19, 等。

## 验证(必须)
修改后:`$env:KMP_DUPLICATE_LIB_OK="TRUE"; & "D:\uv_envs\uv_cuda\Scripts\python.exe" -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=uv_cuda --ExecutePreprocessor.timeout=180 <ipynb>`
遇 Kernel died 竞态用备选 exec 逐 cell(注明)。
最后回复:增强了哪些 notebook、新增多少张 matplotlib/seaborn 图、验证结果。