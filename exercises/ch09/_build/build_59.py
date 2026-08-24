# -*- coding: utf-8 -*-
"""生成 59_triton_debug.ipynb(不配 app)"""
from helpers import D, CH09, TRITON_HEADER, ASSERT_CLOSE, new_nb
from pathlib import Path
from nb_builder import chapter_cover, wrapup

NB = new_nb("第 59 课 · 调试与验证",
            subtitle="越界、类型、布局:把 triton kernel 的 bug 揪出来,再用对拍与误差分析证明它是对的",
            emoji="🐛")

chapter_cover(NB,
    objectives=[
        "认识 triton kernel 三类常见错误:越界访问、类型/精度、布局/维度",
        "掌握 print 调试三件套:tl.static_print / tl.device_print / Python print",
        "学会用 naive 参考实现对拍验证 kernel 正确性,并把误差可视化",
        "会做数值误差分析,设定合理的 atol / rtol 阈值",
    ],
    toc=[
        ("直觉:kernel 调试是戴着显微镜找虫子", "错误信息、print、对拍、误差,四件工具"),
        ("越界访问:静默的“读垃圾”", "无 mask 的越界不报错,只会污染结果"),
        ("类型与精度:fp16 陷阱", "累加溢出、dot 输入要求、静默的 inf/NaN"),
        ("布局与维度:编译器替你画红线", "tl.arange 必须 2 的幂;dot 两侧维度必须对齐"),
        ("print 调试三件套", "static_print / device_print / python print 各显神通"),
        ("对拍验证:写一个 assert_allclose 工具箱", "小规模对拍 + 逐元素误差热力图"),
        ("数值误差分析:误差从哪来,阈值怎么定", "fp16 vs fp32 的误差增长曲线与阈值建议"),
        ("本课小结与练习", "无配套 app:wrapup 照常写"),
    ],
    links=[
        ("Triton 错误信息文档", "https://triton-lang.org/main/programming-guide/errors.html"),
        ("torch.allclose 文档(atol/rtol 语义)", "https://pytorch.org/docs/stable/generated/torch.allclose.html"),
        ("Numerical Computation(数值分析经典教材)", "https://www.cs.utexas.edu/users/flame/laff/alaff/"),
    ])

NB.md("## 1️⃣ 直觉:kernel 调试是戴着显微镜找虫子 🔬",
D('''
普通 Python 程序有断点、有堆栈、有 IDE——kernel 什么都没有:它在 GPU 上运行,几百个线程同时跑,
你没法“暂停到第 3 行”。但别慌,kernel 调试只需要四件工具:

1. **错误信息**:编译期错误是最好抓的,编译器会告诉你“哪一行、什么类型”;
2. **print 调试**:`tl.static_print`(编译期打印)与 `tl.device_print`(运行时打印);
3. **对拍验证**:写一个 naive 参考实现,逐元素比对——这是“有没有 bug”的最终裁判;
4. **误差分析**:即使对拍通过了,也要理解误差来源,设合理的 `atol/rtol`,别拿 fp16 的结果
   去要求 fp64 的精度。

这四件工具就像显微镜、解剖刀、对照实验和误差棒。本课把常见的坑一个个踩给你看,再教你修。
'''))

NB.code(TRITON_HEADER, "✅ 环境自检:调试需要反复编译,先设好 ptxas 保证编译稳定。")

NB.md("## 2️⃣ 越界访问:静默的“读垃圾” 🌪️",
D('''
Python 里数组越界会抛 `IndexError`;但 triton 的 `tl.load` 越界时,**默认什么都不报**——
只是读到越界地址上的垃圾值。这是 kernel 里最阴险的一类 bug:程序“正常”跑完,结果悄悄变错。

演示一下:故意把偏移量加得很大,让 load 直接读到显存别的区域:
'''))

NB.code(D('''
@triton.jit
def oob_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK) + 1000000   # 故意越界,没有任何 mask
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + tl.arange(0, BLOCK), x)

x = torch.randn(256, device="cuda")
o = torch.zeros(256, device="cuda")
try:
    oob_kernel[(1,)](x, o, BLOCK=256)
    torch.cuda.synchronize()
    print("越界 load 没有报错!读到的“垃圾”前 5 个:", o[:5].tolist())
    print("正常数据前 5 个:", x[:5].tolist())
except Exception as e:
    print("抛出了异常:", type(e).__name__, e)
'''), "⚠️ 看到了吗——**没有 IndexError**。GPU 不管地址合不合法,它只负责把 `x_ptr + 偏移` 处的字节读回来。")

NB.code(D('''
# 修复:加 mask,越界位置读 0 或直接不读
@triton.jit
def safe_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n                                   # 边界掩码
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)   # 掩码外读 0
    tl.store(o_ptr + offs, x, mask=mask)              # 掩码外不写

x = torch.randn(256, device="cuda")
o = torch.zeros(256, device="cuda")
safe_kernel[(1,)](x, o, 100, BLOCK=256)
torch.cuda.synchronize()
print("掩码外(o[100:])保持 0:", (o[100:] == 0).all().item(), "| 掩码内正确:",
      torch.allclose(o[:100], x[:100]))
'''), "✅ 修法就一行:`mask` + `other`。mask 告诉 GPU“这些地址别碰”,`other` 指定掩码外的占位值。")

NB.md("## 3️⃣ 类型与精度:fp16 陷阱 🎭",
D('''
第二种常见 bug 是**类型**。最典型的两幕:

- **累加溢出**:fp16 最大值只有 ~65504。两个 60000 相加 → `inf`,然后所有 downstream 计算全变 NaN;
- **累加精度**:即使不溢出,fp16 累加也会累积舍入误差——所以 `tl.dot` 的累加器一定要用 fp32。

先看溢出:
'''))

NB.code(D('''
@triton.jit
def fp16_overflow_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    acc = tl.zeros([BLOCK], dtype=tl.float16)   # 错误的累加器:fp16
    acc += x
    acc += x                                    # 60000 + 60000 = 120000 > 65504
    tl.store(o_ptr + offs, acc)

x = torch.full((256,), 60000.0, device="cuda", dtype=torch.float16)
o = torch.empty_like(x)
fp16_overflow_kernel[(1,)](x, o, BLOCK=256)
torch.cuda.synchronize()
print("60000+60000 =", o[0].item(), "(inf 就是溢出了!)")
'''), "⚠️ `inf` 一旦出现,`exp`、`softmax`、`norm` 全都会跟着变成 NaN——而且完全静默。")

NB.code(D('''
# 修复:累加器换 fp32,最后再转回输出 dtype
@triton.jit
def fp32_acc_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    acc = tl.zeros([BLOCK], dtype=tl.float32)   # fp32 累加器
    acc += x
    acc += x
    tl.store(o_ptr + offs, acc.to(tl.float16))  # 结尾再转回

x = torch.full((256,), 60000.0, device="cuda", dtype=torch.float16)
o = torch.empty_like(x)
fp32_acc_kernel[(1,)](x, o, BLOCK=256)
torch.cuda.synchronize()
print("60000+60000 =", o[0].item(), "(fp32 累加,结果正确)")
'''), "✅ 黄金法则:`tl.dot` 的 acc 用 `tl.float32`,需要 fp16 的地方显式 `.to(...)` 转换。")

NB.md("## 4️⃣ 布局与维度:编译器替你画红线 📏",
D('''
第三类是**布局/维度**错误。这类错误好在——**编译期就报**!triton 对 tile 的形状约束很死,
不满足就抛 `CompilationError`,并把出错的行号指给你。

两个最经典的“红线”:

- `tl.arange(0, N)` 的 N **必须是 2 的幂**(因为 tile 要均匀摊给线程);
- `tl.dot(a, b)` 两侧的 reduction 维度必须一致。

逐一演示:
'''))

NB.code(D('''
@triton.jit
def bad_arange_kernel(x_ptr, o_ptr):
    offs = tl.arange(0, 100)      # 100 不是 2 的幂 → 编译错误
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x)

try:
    x = torch.randn(256, device="cuda"); o = torch.empty_like(x)
    bad_arange_kernel[(1,)](x, o)
except Exception as e:
    print("类型:", type(e).__name__)
    print("报错:", str(e).strip().splitlines()[-1])
'''), "🎯 修复很简单:`tl.arange(0, triton.next_power_of_2(N))` + mask 兜底(第 2 节学过的组合)。")

NB.code(D('''
@triton.jit
def bad_dot_kernel(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :])
    b = tl.load(b_ptr + tl.arange(0, BLOCK // 2)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :])
    c = tl.dot(a, b)      # a:(BLOCK,BLOCK) × b:(BLOCK/2,BLOCK) → reduction 维度不一致
    tl.store(c_ptr + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :], c)

try:
    a = torch.randn(64, 64, device="cuda"); b = torch.randn(64, 64, device="cuda")
    c = torch.empty(64, 64, device="cuda")
    bad_dot_kernel[(1,)](a, b, c, BLOCK=64)
except Exception as e:
    print("类型:", type(e).__name__)
    print("报错:", str(e).strip().splitlines()[-1])
'''), "🔍 这类错误的价值在于:编译器把你“形状没对齐”的意图当场拦住,而不是跑到一半才出 NaN。")

NB.md("## 5️⃣ print 调试三件套 🖨️",
D('''
当 bug 不是编译期错误、而是“结果悄悄错”时,就该掏出 print。triton 提供两种内置打印,
加上一种“外挂”:

| 工具 | 时机 | 用途 |
|------|------|------|
| `tl.static_print` | 编译期 | 打印 `tl.constexpr`(BLOCK、形状),每编译一次打一次 |
| `tl.device_print` | 运行时 | 在 GPU 上打印数值,注意要 `if pid == 0` 守卫 |
| Python `print` | kernel 外 | 把 kernel 内部步骤搬到 torch 复现,打印中间张量 |

先看前两个:
'''))

NB.code(D('''
@triton.jit
def dbg_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.static_print("BLOCK =", BLOCK)          # 编译时打印一次
    x = tl.load(x_ptr + offs)
    if pid == 0:                                        # 守卫:只让第一个 program 打
        tl.device_print("x.sum =", tl.sum(x))            # GPU 上打印标量(标签须为 ASCII)
        tl.device_print("x.max =", tl.max(x))
    tl.store(o_ptr + offs, x * 2)

x = torch.randn(256, device="cuda")
o = torch.empty_like(x)
dbg_kernel[(1,)](x, o, BLOCK=256)
torch.cuda.synchronize()
print("kernel 正常执行完")
print("Python 侧核对: x.sum =", x.sum().item(), " x.max =", x.max().item())
'''), "🖨️ 你会看到:static_print 在编译期打一次;device_print 在运行时打印,且与 Python 侧数值一致——这就是三件套的用法。")

NB.code(D('''
# 用三件套定位一个“悄悄错”的 bug:统计每个程序处理的元素数
# bug:grid 算错,导致最后一个 program 覆盖不全
@triton.jit
def count_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.static_print("BLOCK =", BLOCK)
    if pid == 0:
        tl.device_print("num_programs =", tl.num_programs(0))   # 实际起了几个 program
    cnt = tl.sum((offs < n).to(tl.int32))
    tl.store(o_ptr + pid, cnt)

n = 1000
BLOCK = 256
x = torch.zeros(n, device="cuda")
o = torch.zeros(triton.cdiv(n, BLOCK) + 2, device="cuda")
count_kernel[(triton.cdiv(n, BLOCK) + 1,)](x, o, n, BLOCK=BLOCK)   # grid 多开了一个 program
torch.cuda.synchronize()
print("每个 program 覆盖的元素数:", o[:triton.cdiv(n, BLOCK) + 1].tolist())
print("期望:第 4 个 program 应覆盖 1000-768=232 个(第 0~2 个各 256)")
'''), "🎯 用 device_print 确认了 program 数,再用输出数组核对每个 program 覆盖的元素数——“多开了一个 program”这个 bug 当场现形。")

NB.md("## 6️⃣ 对拍验证:写一个 assert_allclose 工具箱 🧰",
D('''
print 只能救急,真正让 kernel “被证明正确”的是**对拍**:写一个朴素参考实现,同输入、比输出。
把对拍工具封装好,以后每个 kernel 都先用它验一遍再谈性能。

下面这个 `assert_allclose` 会打印 max/mean 绝对误差与 max 相对误差,并返回 `torch.allclose`
判定结果(可指定 `atol/rtol`)。我们用第 60 课的 softmax 当例子跑一遍完整流程:
'''))

NB.code(ASSERT_CLOSE, "🧰 对拍工具箱:max_abs 看最大偏差,mean_abs 看整体偏差,max_rel 看相对放大,allclose 给出最终判定。")

NB.code(D('''
# 场景 1:写一个“有 bug 的” softmax —— 忘了减 max(数值不稳定)
def softmax_buggy(x):
    p = torch.exp(x)                       # 大值时直接溢出
    return p / p.sum(dim=-1, keepdim=True)

x = torch.randn(8, 16, device="cuda") * 300  # 放大到 ±900,exp 必炸
y_bug = softmax_buggy(x)
y_ok = torch.softmax(x, dim=-1)
print("buggy 输出的 NaN 数量:", int(torch.isnan(y_bug).sum()))
assert_allclose(y_bug, y_ok, name="buggy softmax(应失败)")
'''), "⚠️ 看到 NaN 与 allclose=False:这就是“没减 max”的 bug 在对拍下的原型。")

NB.code(D('''
# 场景 2:triton softmax(先减 max)与 torch 参考对拍
import triton.language as tl

@triton.jit
def softmax_fwd(x_ptr, o_ptr, M, N, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=0)
    p = tl.exp(x - m)
    s = tl.sum(p, axis=0)
    tl.store(o_ptr + row * row_stride + offs, p / s, mask=mask)

def softmax_triton(x, BLOCK_N=16):
    M, N = x.shape
    o = torch.empty_like(x)
    softmax_fwd[(M,)](x, o, M, N, N, BLOCK_N=triton.next_power_of_2(N))
    return o

x = torch.randn(8, 16, device="cuda") * 300     # 同样的大输入
y_tri = softmax_triton(x)
y_ref = torch.softmax(x, dim=-1)
assert_allclose(y_tri, y_ref, atol=1e-5, rtol=1e-5, name="triton softmax(大值)")
'''), "✅ triton 版先减 max,大输入下依然与 torch 完全一致——对拍通过。")

NB.code(D('''
# 把误差画出来:逐元素误差热力图
diff = (y_tri - y_ref).abs()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
im0 = axes[0].imshow(y_tri.cpu(), cmap="RdBu_r", aspect="auto")
axes[0].set_title("triton softmax 输出")
fig.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(diff.cpu(), cmap="hot", aspect="auto")
axes[1].set_title("|triton - torch| 逐元素误差")
fig.colorbar(im1, ax=axes[1], fraction=0.046)
plt.tight_layout()
'''), "🎨 误差热力图一眼看穿:右图整片深色(≈0)说明“处处一致”;如果有亮斑,那就是某个区域的 bug。")

NB.md("## 7️⃣ 数值误差分析:误差从哪来,阈值怎么定 📊",
D('''
对拍通过不等于“精确”——fp16/fp32 本身就有舍入误差,而且误差会随计算规模**累积**。
做误差分析时,先搞清楚三件事:

1. **输入精度**:fp16 尾数只有 10 bit(相对精度 ~1e-3),fp32 有 23 bit(~1e-7);
2. **累加顺序**:`tl.sum` 用树状归约,和 torch 的归约顺序不同,误差会略有差异;
3. **操作数量**:误差随累加项数增长,长得像 $\\sqrt{n} \\cdot \\epsilon$。

下面画一张“误差随向量长度增长”的曲线:
'''))

NB.code(D('''
# 误差随累加规模增长:fp32 vs fp16 累加
def sum_fp32(x):
    return x.double().to(torch.float32).sum().float()   # 视 fp32 累加为“参考”

lengths = [64, 256, 1024, 4096, 16384]
err_fp32, err_fp16 = [], []
for n in lengths:
    x = torch.randn(n, device="cuda")
    s32 = x.float().sum()
    s16 = x.half().float().sum()
    sref = x.double().sum()
    err_fp32.append((s32 - sref).abs().item() / sref.abs().item())
    err_fp16.append((s16 - sref).abs().item() / sref.abs().item())
    print(f"n={n:6d}: 相对误差 fp32={err_fp32[-1]:.2e}  fp16={err_fp16[-1]:.2e}")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(lengths, err_fp32, "o-", label="fp32 累加", color="#4C78A8")
ax.plot(lengths, err_fp16, "s-", label="fp16 累加", color="#E45756")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("累加元素数 n")
ax.set_ylabel("相对误差")
ax.set_title("求和相对误差 vs 元素数:fp16 比 fp32 差约 2 个数量级且增长更快")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
'''), "📊 两条线都随 n 缓慢爬升(fp16 更陡),这解释了为什么 kernel 里能用 fp16 存数据、但累加必须用 fp32。")

NB.code(D('''
# 阈值怎么定:经验法则表
print("""
对拍阈值经验法则(atol=绝对容差, rtol=相对容差):
  fp32 参考 + fp32 kernel : atol=1e-5, rtol=1e-4  (多数情况)
  fp16 输入 + fp32 累加   : atol=1e-2, rtol=1e-2  (让出 fp16 的舍入余量)
  大规模(累加项 >1e5)     : 至少放宽一个量级
  flash-attn vs naive      : atol=2e-2 常用(官方就是这么定的)
""")
'''), "🏷️ 结论:阈值不是“越小越好”,而是“比噪声大一个量级、比 bug 小一个量级”。把 fp16 的噪声(1e-3)误判成 bug,或把真实 bug 放过,都不好。")

wrapup(NB,
    summary=[
        "越界访问默认不报错:用 mask + other 兜底;无 mask 时读到的全是垃圾值",
        "类型陷阱集中在累加:fp16 累加会溢出成 inf、累积误差大,累加器一律用 fp32",
        "布局/维度错误由编译器在编译期拦截(tl.arange 必须 2 的幂、dot 维度要对齐),读报错行号即可",
        "print 三件套:tl.static_print(编译期)、tl.device_print(运行时 + if pid==0 守卫)、Python print(外部核对)",
        "对拍是正确性的最终裁判:assert_allclose 打印 max/mean 误差并给 allclose 判定;再画误差热力图定位",
        "误差随规模增长、fp16 比 fp32 差两个量级;atol/rtol 按“比噪声大一量级、比 bug 小一量级”来设",
    ],
    practice=[
        "给第 2 节的 safe_kernel 加上一个“读满两个块”的版本(先算 num_blocks 再循环),验证不越界",
        "把第 7 节曲线改画成“不归一化、直接画绝对误差”,观察绝对误差随 n 的线性增长",
        "写一个 triton 求和的 kernel,故意用 fp16 累加,跑第 7 节的对比,确认误差数量级",
        "把第 6 节的 assert_allclose 加到第 58 课的 paged attention 上,给每个 (seq,head) 打对拍报告",
    ],
    links=[
        ("Triton 编程指南:错误处理与调试", "https://triton-lang.org/main/programming-guide/"),
        ("PyTorch 数值稳定性注意事项", "https://pytorch.org/docs/stable/notes/numerical_accuracy.html"),
        ("Goldberg: What Every Computer Scientist Should Know About FP", "https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html"),
    ])

NB.save(str(Path(CH09) / "59_triton_debug.ipynb"))
