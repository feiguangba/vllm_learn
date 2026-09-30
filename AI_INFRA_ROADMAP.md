# AI Infra 瀛︿範浣撶郴 路 鎬昏矾绾垮浘

> 宸ヤ綔鐩綍:`D:\Project\21-Cpp_learn\explore`
> 鐜:RTX 5060 Laptop (sm_120) 路 CUDA 13.3 路 MSVC 14.44 (D:\VS_BuildTools) 路 GCC 15.2 路 uv_cuda (Python 3.12 + torch 2.11.0+cu128)
> 鐩爣:浠?C++/CUDA 鍒?AI 鎺ㄧ悊寮曟搸,寤虹珛"浼氬啓 kernel 鈫?浼氬啓绯荤粺"鐨勫畬鏁磋兘鍔涙爤銆?
---

## 馃椇锔?鎬讳綋鏋舵瀯

```
C++ 鐜颁唬鐗规€?(01-70) 鈹€鈹€鈻?CUDA 缂栫▼ (71-100) 鈹€鈹€鈻?vLLM 鎺ㄧ悊寮曟搸 (minivllm)
        鈹?                       鈹?                       鈹?  璇█涓庡伐绋嬪熀纭€            GPU 骞惰璁＄畻鍩虹          绯荤粺绾х悊瑙?+ 婧愮爜绮捐
        鈹?                       鈹?                       鈹?        鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                   鈻?                       鈻?             23-CMU-15418              AI Infra 瀹炴垬璺嚎
        骞惰绯荤粺鍏ラ棬(鐞嗚/浣滀笟)       (鏈矾绾垮浘鐨勯暱鏈熺洰鏍?
```

---

## 馃摎 瀛︿範妯″潡绱㈠紩

### 妯″潡 A:C++ 鐜颁唬鐗规€?鏁欑▼ 01-70,25 涓?.cpp)

| 闃舵 | 缂栧彿 | 涓婚 |
|---|---|---|
| 璇█鍩虹 | 01-25 | 璇硶/STL/闈㈠悜瀵硅薄(鏁扮粍銆乿ector銆乵ap銆乻et銆乴ambda銆佹櫤鑳芥寚閽堛€丷AII銆乵ove 璇箟鈥? |
| 鐜颁唬鐗规€?| 26-45 | 妯℃澘/娉涘瀷/鐜颁唬 C++(auto銆乧onstexpr銆佹姌鍙犺〃杈惧紡銆丼FINAE鈥? |
| 杩涢樁 | 46-70 | **鐜颁唬 C++ 鐗规€?*(span銆乵dspan銆乿ariant銆乺anges銆乧oroutines銆乧oncepts銆乫ormat銆乪xpected銆丼IMD銆佸唴瀛樻睜銆佺被鍨嬫摝闄ゃ€丄oS/SoA銆佹暟鍊肩ǔ瀹氭€с€乥enchmark鈥? |

> 鐩爣:鍐欎换浣?C++ 浠ｇ爜閮界敤鐜颁唬鎯敤娉?涓?CUDA 涓?AI 宸ョ▼鎵撳簳銆?
### 妯″潡 B:CUDA 缂栫▼(鏁欑▼ 71-100,30 涓?.cu)

| 闃舵 | 缂栧彿 | 涓婚 |
|---|---|---|
| 鍩虹 | 71-75 | 绋嬪簭妯″瀷/鍐呭瓨妯″瀷/鍚戦噺鍔?鏈寸礌 GEMM/鍒嗗潡 GEMM |
| 鏍稿績妯″紡 | 76-80 | 褰掔害/softmax/鍘熷瓙鎿嶄綔/鎵弿/杞疆(bank conflict) |
| 楂樼骇鐗规€?| 81-88 | 鍗风Н/甯搁噺鍐呭瓨/娴佷笌寮傛/璁℃椂/缁熶竴鍐呭瓨/澶?GPU/shuffle/瀵勫瓨鍣ㄥ垎鍧?|
| 寮犻噺鏍?| 89-92 | WMMA/mma.sync/FlashAttention 绠€鍖?閲忓寲 |
| 宸ョ▼鍖?| 93-100 | CUDA Graphs/Cooperative Groups/roofline/cuBLAS/RAII/sanitizer/绠楀瓙搴?鎬荤粨 |

> 鍏抽敭鎬ц兘杞ㄨ抗(GEMM 1024鲁):鏈寸礌 1.1 鈫?鍒嗗潡 1.4 鈫?瀵勫瓨鍣?3.3 鈫?WMMA 5.5 鈫?cuBLAS 9.4 TFLOPS銆?
### 妯″潡 C:minivllm 鈥斺€?鍥捐В vLLM 鎺ㄧ悊寮曟搸

- **repowiki 鏋舵瀯鏂囨。**(8 绡?甯?`鏂囦欢:琛屽彿`,鍩轰簬 vendor/vllm v0.23-dev)
  01 绯荤粺鏋舵瀯 / 02 PagedAttention+KV Cache / 03 Scheduler / 04 LLMEngine / 05 ModelRunner+CUDA Graph / 06 閲忓寲 / 07 TP-PP-DP / 08 Attention 鍚庣
- **缁冧範鍐?*(50 璇?ipynb + 50 涓?streamlit app,鍙傜収"楦㈠熬鑺变功"椋庢牸)
  ch01 鎺ㄧ悊鍩虹(01-06)鈫?ch02 KV Cache(07-13)鈫?ch03 璋冨害(14-20)鈫?ch04 鎵ц+CUDA Graph(21-27)鈫?ch05 閲忓寲(28-34)鈫?ch06 骞惰(35-41)鈫?ch07 Attention Kernel(42-47)鈫?ch08 閮ㄧ讲(48-50)
- 璇﹁ [`minivllm/README.md`](minivllm/README.md)

### 妯″潡 D:23-CMU-15418(骞惰绯荤粺鍏ラ棬)

- `23-CMU-15418/` 鐩綍:璇剧▼璧勬枡(lectures/assignments/demo),鍚?threads 绔炰簤銆丱penMP銆丼IMD銆丆UDA 婕旂ず
- 瀹氫綅:琛ヨ冻骞惰绯荤粺鐞嗚(鍐呭瓨涓€鑷存€с€佷换鍔″苟琛屻€佹祦姘寸嚎銆佹€ц兘妯″瀷)

### 妯″潡 E:vendor 婧愮爜搴?绮捐瀵硅薄)

| 浠撳簱 | 鐢ㄩ€?|
|---|---|
| `vendor/vllm` | vLLM 婧愮爜(repowiki 渚濇嵁,commit 967e104) |
| `vendor/llama.cpp` | 杞婚噺鎺ㄧ悊寮曟搸鍙傝€?|
| `vendor/triton` | 鑷畾涔?kernel DSL 鍙傝€?|
| `vendor/tinygrad` | 娣卞害瀛︿範妗嗘灦鏈€灏忓疄鐜板弬鑰?|

---

## 馃摉 鎺ㄨ崘瀛︿範椤哄簭

```
绗竴闃舵(璇█涓庡熀纭€)
  01-45 C++ 鍩虹涓?STL 鈫?46-70 鐜颁唬 C++ 鐗规€?  鈫?71-75 CUDA 鍩虹 鈫?76-80 鏍稿績绠楁硶妯″紡
  鈫?81-88 楂樼骇 CUDA 鈫?89-92 寮犻噺鏍?鈫?93-100 宸ョ▼鍖?
绗簩闃舵(绯荤粺鐞嗚В)
  23-CMU-15418 骞惰绯荤粺鐞嗚
  鈫?minivllm/repowiki 01 绯荤粺鏋舵瀯
  鈫?minivllm/exercises ch01-ch04(鎺ㄧ悊鍩虹鈫掓墽琛?

绗笁闃舵(娣卞叆)
  minivllm/exercises ch05-ch08(閲忓寲/骞惰/Kernel/閮ㄧ讲)
  鈫?repowiki 02-08 瀵圭収婧愮爜绮捐
  鈫?vendor/vllm 鏍稿績鏂囦欢娣卞叆(core.py/scheduler.py/block_pool.py/gpu_model_runner.py)

绗洓闃舵(瀹炴垬)
  鐢?99 璇剧畻瀛愬簱楠ㄦ灦鎺?PyTorch 鎵╁睍
  鈫?璇?cutlass / flash-attn 婧愮爜
  鈫?鍦?RTX 5060 涓婂仛鐢熶骇绾ф帹鐞嗕紭鍖?鍐呭瓨姹?娴佹按/CUDA Graphs)
```

---

## 馃幆 閲岀▼纰戞鏌ョ偣

- [x] **M1**:C++ 01-70 鍏ㄩ儴瀹屾垚(鐜颁唬 C++ 鐗规€?46-70 鍚?span/ranges/coroutines/SIMD)
- [x] **M2**:CUDA 71-100 鍏ㄩ儴缂栬瘧杩愯楠岃瘉閫氳繃(鍚?WMMA 6x銆乧uBLAS 8.8x 鎬ц兘瀵规瘮)
- [x] **M3**:uv_cuda 鐜(Python 3.12 + torch 2.11 + pyecharts/streamlit/plotly)
- [x] **M4**:minivllm 缁冧範鍐?50 璇惧叏閮ㄧ敓鎴愬苟楠岃瘉 + 8 绡?repowiki 鏂囨。
- [ ] **M5**:23-CMU-15418 鐩綍鏁寸悊涓?demo 杩愯(鍙€?宸查儴鍒嗗畬鎴?
- [ ] **M6**:鐢熶骇绾т紭鍖栧疄鎴?寮犻噺鏍?GEMM + FlashAttention 瀹屾暣鐗?+ 澶氬崱 NCCL)

---

## 馃敆 鍏抽敭璧勬簮

- vLLM 瀹樻柟鏂囨。:https://docs.vllm.ai
- PagedAttention 璁烘枃:https://arxiv.org/abs/2309.06180
- FlashAttention 璁烘枃:https://arxiv.org/abs/2205.14135
- CUDA Graphs 鍗氬:https://developer.nvidia.com/blog/cuda-graphs/
- NVIDIA 瀛︿範涓績:https://developer.nvidia.com/learn

---

*鏇存柊鏃ユ湡:2026-08-18 路 鐢卞涔犺繃绋嬭嚜鍔ㄧ淮鎶?璇﹁鍚勫瓙鐩綍 README銆?
