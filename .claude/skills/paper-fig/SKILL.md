# Paper-Fig：顶会级科研绘图 Skill

> 来源：提炼自《AI 顶会绘图黄金 8 原则》（知乎 zhuanlan.zhihu.com/p/1973007126491308538），
> 并结合 NeurIPS/CVPR/AAAI 论文中架构图（Transformer、ViT、LLaMA、vLLM 技术报告）的通用视觉语言。
> 适用：本项目所有 SVG 架构图 / 流水线图 / 状态机图 / 对比图。

## 0. 使命

**3 秒规则**：读者 3 秒内看懂这张图讲了什么。
图的本质是 **信息工程**：结构 ≫ 文本，空间 ≫ 色彩，形状 ≫ 字句。

## 1. 黄金 8 原则（画图前必须逐条自检）

| # | 原则 | 落地要求 |
|---|------|---------|
| 1 | 一眼看懂，不要句子要结构 | 不写长句；用模块、信息流、语义分组、对齐表达；每块文字 ≤ 8 字/词组 |
| 2 | 重点必须强烈高亮 | 基础/冻结组件一律灰色；核心/创新模块用亮色 + glow（橙色系）；一眼能指出"这张图的亮点" |
| 3 | 字体统一 | 单一无衬线字体栈；标题 20px、模块名 15px、注释 12px、脚注 11px；严禁混用字号 |
| 4 | 模块形状与线条统一 | 全部圆角矩形 rx=10；边框统一 2px；箭头统一 marker；圆角/线宽全图一致 |
| 5 | 色彩克制 2~4 种 | 只用本 skill 的语义色板；色彩用于区分结构，不是装饰 |
| 6 | 留白是高级感核心 | 画布四周 ≥ 40px 边距；模块间距 ≥ 28px；禁止塞满 |
| 7 | 认知一致性 | 同类数据流永远同色同箭头（如张量=蓝、KV=紫、高亮路径=橙），跨图保持 |
| 8 | 全套图风格一致 | 必须复制 §5 的 SVG 骨架（defs/字体/marker/filter）作为开头；像同一位设计师画的 |

## 2. 语义色板（唯一允许的色板）

| 语义 | 描边 | 填充 | 用途 |
|------|------|------|------|
| 主流程/普通模块 | `#2563EB` 蓝 | `#DBEAFE` | 常规模块、网络层 |
| 重点/创新/高亮 | `#F59E0B` 橙 | `#FEF3C7` | 本图核心模块，加 glow 滤镜 |
| 数据流/张量/KV | `#7C3AED` 紫 | `#EDE9FE` | 数据实体、缓存、矩阵 |
| 基础/冻结/背景 | `#64748B` 灰 | `#F1F5F9` | 固定组件、上下文 |
| 输出/成功（少量） | `#10B981` 绿 | `#D1FAE5` | 最终输出、通过状态 |
| 文本主色 | — | — | `#1E293B`；次级 `#475569`；脚注 `#94A3B8` |
| 背景 | — | `#FFFFFF` | 分区面板可用 `#F8FAFC` |

禁令：禁止出现色板之外的颜色；禁止渐变彩虹；禁止深色背景。

## 3. 图型配方

### 3.1 神经网络架构图（Visio 风）
- 从下到上（或左到右）依次分区并明确标注：**输入层 → 隐藏层堆叠（N×）→ 输出层**；
- 每个块内注明**所用网络/算子**：如 `Embedding`, `Linear(Q/K/V)`, `Scaled Dot-Product Attention`,
  `Concat + Linear`, `LayerNorm`, `FFN(Linear 4x → GELU → Linear)`, `lm_head + Softmax`；
- 大括号/堆叠卡片表示 `× N layers` 重复；残差连接用旁路曲线箭头；
- 每块右侧标注张量形状 `[B, T, d]`；
- 同类层必须同色（所有 Linear 蓝色、所有 LayerNorm 灰色、核心 Attention 橙色高亮）。

### 3.2 流水线 / Pipeline 图
- 主路径：从左到右一条**粗大箭头**贯穿，4~6 个语义原子模块，不跳跃；
- 信息分层：高层流程名 → 模块名 → 1~3 个关键字 →（可选）一行小字注释；
- 分支/旁路（如 cache 命中）用细箭头 + 虚线，永远不与主路径混淆。

### 3.3 对比图（A vs B）
- 上下两个面板，面板标题左对齐；两面板模块坐标严格对齐，让差异一眼可见；
- 用灰底"空泡/闲置"表达浪费，用橙色表达高效区。

### 3.4 状态机图
- 圆角矩形 = 状态，箭头 = 迁移并标注触发事件；初始状态双圈或加粗边框；
- 状态颜色：正常蓝、异常/抢占橙、终止绿。

## 4. 文案规范

- 中文为主 + 英文术语原文（如 `分页 KV Cache (PagedAttention)`）；
- 每图底部一行 11px 灰色脚注：一句话核心结论，如"KV Cache 用 O(T) 增量写入替代 O(T²) 重算"；
- 数字/公式用直角引号或等宽字体呈现，单独放在小标签框内。

## 5. SVG 骨架（每张图必须以此为开头，逐字复制后修改尺寸）

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 800" width="1400" height="800" font-family="Inter, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="arrB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#2563EB"/></marker>
    <marker id="arrP" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#7C3AED"/></marker>
    <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#F59E0B"/></marker>
    <marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#64748B"/></marker>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="0" stdDeviation="6" flood-color="#F59E0B" flood-opacity="0.45"/></filter>
    <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0F172A" flood-opacity="0.10"/></filter>
  </defs>
  <rect width="1400" height="800" fill="#FFFFFF"/>
  <!-- 标题：20px bold #1E293B，左上角 x=48 y=52 -->
  <!-- 正文从这里开始 -->
</svg>
```

技术约束：
- 纯 SVG、无外部图片/无 CSS 引用/无 JS；文字必须用 `<text>`（禁止把文字转路径导致不可改）；
- 所有块用 `<g>` 分组并带注释 `<!-- 模块名 -->`，方便审校定位；
- 主箭头 `stroke-width="3"`，普通连接线 `2`，虚线 `2 dasharray="6 4"`；
- text 禁止溢出块边界：先估字符宽（中文≈fontsize，英文≈0.6×fontsize），必要时两行；
- 画布尺寸：架构总图 1400×800，对比图 1400×900，小图 1200×700。

## 6. 工作流

1. **定主路径**：只保留 4~6 个关键模块，先画骨架再填充；
2. **语义分组**：相关模块放进同一浅灰面板（`#F8FAFC`, rx=16, 虚线或实线 1px `#E2E8F0`）；
3. **上色**：按 §2 语义色板；核心模块橙+glow，其余克制；
4. **文字**：按 §3/§4 层级写词组，不写句子；
5. **自检**：跑 §7 审校清单，逐项打钩才算完成。

## 7. 审校清单（Reviewer 用同一张表）

- [ ] 3 秒能说出本图核心（主路径/亮点清晰）
- [ ] 颜色 ≤ 色板范围，语义一致
- [ ] 字号只有 20/15/12/11 四档，同一字体栈
- [ ] 所有矩形 rx=10、线宽统一、箭头 marker 统一
- [ ] 模块对齐（同列同 x、同行同 y，间距均匀 ≥ 28px）
- [ ] 无文字溢出、无重叠、箭头不穿块
- [ ] 重点模块有橙+glow，基础组件灰色
- [ ] 架构图标注了输入层/隐藏层/输出层及每块所用网络
- [ ] 底部有 11px 结论脚注
- [ ] 纯 SVG 可缩放，text 未转路径
- [ ] 技术内容准确（术语、公式、方向、因果无错）

## 8. 雷区（审稿人扣分点）

颜色太多太艳 / 字太小混乱 / 块箭头不对齐 / 图太密无留白 / 亮点不突出 /
文字解释成句堆砌 / 截图拼图 / 低分辨率位图 / 风格不统一 —— 全部禁止。
