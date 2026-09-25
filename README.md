# 点选验证码形状匹配 · Rust 版

[click-captcha-matcher](https://github.com/SUSTechHSAS/click-captcha-matcher) 推理部分的 Rust 重写：
输入 250×80 点选验证码 JPEG，输出按提示字顺序排列的 4 个点击坐标。

- `no_std` Rust，除 libc 外零依赖：JPEG 解码、切字、双塔 CNN、360 种分配全部自带
- 模型编进库里（124 KB），运行时不需要 onnxruntime / numpy / Pillow
- 产物：`libccm`（C ABI，见 `include/ccm.h`）、`ccm-cli`（命令行）、
  `python/solver.py`（ctypes 封装，接口与原版 `solver.py` 相同）

训练仍在原 Python 仓库里做（PyTorch）；训练好的模型用 `tools/export_ccm.py` 转成 `.ccm`。

## 1. 结果

### 体积

| | 原版（Python） | 本仓库 |
| --- | --- | --- |
| 推理所需 | onnxruntime + numpy + Pillow + 0.7 MB 的 ONNX | `libccm.so` 198 KB（含模型） |
| course-grabber 单文件包（Linux x64，PyInstaller） | 41.3 MB | **8.6 MB** |

`libccm.so` 的构成：模型 124 KB，代码约 55 KB，其余是 ELF 开销（macOS arm64 的 `.dylib` 为 216 KB）。
course-grabber 不含识别的轻量版是 8.4 MB，也就是说识别能力从 +33 MB 变成了 +0.25 MB（实测见第 5 节）。

### 速度

单线程、端到端（解码 + 切字 + 模型 + 分配），同一台机器、同一批 300 / 100 张实测图：

| 机器 | 模型 | Python + onnxruntime 1.30 | 本仓库 |
| --- | --- | ---: | ---: |
| Ryzen 7 5700U（锁在 1.8 GHz） | w16 | 2.92 ms | **1.64 ms** |
| | s1 | 6.36 ms | **5.11 ms** |
| Apple M4 | w16 | 1.08 ms | **0.66–0.75 ms** |
| | s1 | 2.99 ms | **2.63 ms** |

计算核心只有一个：3×3 卷积和全连接都归结为同一个小矩阵乘。x86_64 运行时检测到 AVX2+FMA 就用
手写的 AVX2 版（6 像素 × 16 通道），aarch64 用手写的 NEON 版（5 × 16），其他平台用可移植版本。
在 Ryzen 上已经跑到 FMA 理论峰值的 75–80%。

### 准确性

与原版 Python 求解器（Pillow 解码 + onnxruntime）逐张对比，数据是 19,107 张实测验证码，外加 704 张
在线提交过、服务器给了对错的验证码（oracle，其中 701 张通过）：

| 检查项 | 结果 |
| --- | --- |
| 解码后的灰度图 vs Pillow `Image.open(f).convert("L")` | 19,811 张全部逐像素一致 |
| 候选字点击中心 | 全部一致 |
| fp32 权重时的 4×6 相似度 | 最大误差 7×10⁻⁷，答案 0 处不同 |
| **内置量化模型的答案** | 19,107 张中 6 张不同，704 张 oracle 中 2 张不同 |
| 服务器判对的 701 张 | 复现 697 张，与原版 w16 完全相同（697 张） |

答案不同的这 8 张，原版自己的 margin 都 ≤ 0.0041（原版 margin 的 1% 分位是 0.022），
也就是原版本身就在"抛硬币"。oracle 里的 2 张正好一得一失。作为对照，普通 int8 量化在 19,107 张中也有 8 张不同。
s1（int8）在 704 张 oracle + 4,000 张实测上 0 处不同，服务器判对的复现 699/701，也与原版相同。

## 2. 用法

### 构建

```bash
cargo build --release   # target/release/libccm.so（macOS: libccm.dylib，Windows: ccm.dll）与 ccm-cli
cargo test --release    # 解码逐像素对照、端到端、损坏输入模糊测试等
```

需要 Rust ≥ 1.85（stable 即可），不需要其他依赖。

### Python：原 `solver.py` 的替代品

```python
from solver import CaptchaSolver           # python/solver.py；libccm 放在它旁边
solver = CaptchaSolver()                   # 内置 w16；或 CaptchaSolver("models/s1.ccm")
points, margin = solver.solve(jpeg_bytes)  # points: [[x, y]] * 4，按提示字顺序
if margin < solver.min_margin:             # 置信度不足
    ...                                    # 换一张图，而不是提交
```

- `solve()` 接受 JPEG 字节、文件路径，或 numpy 灰度 / 彩色数组（与原版 `to_gray` 相同的处理）
- 失败时抛 `CaptchaError`（`ValueError` 的子类）：不是 JPEG、不是 250×80、渐进式 JPEG 等
- 线程安全，可以多个线程共用一个实例；`threads` 参数只为兼容保留（单线程已经是毫秒级）
- 库的查找顺序：`$CCM_LIB` → `solver.py` 同目录 → PyInstaller 的 `_MEIPASS` → `../target/release`
- 另有 `decode()`（解码成与 Pillow 一致的灰度图）和 `similarity()`（4×6 相似度 + 6 个中心），便于排查

### C

```c
#include "ccm.h"
ccm_solver *s = ccm_new(NULL, 0, NULL);        // NULL: 内置模型；或传 .ccm 文件内容
int32_t p[8]; float margin;
if (ccm_solve(s, jpeg, len, p, &margin) == CCM_OK)
    printf("%d-%d,%d-%d,%d-%d,%d-%d\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
ccm_free(s);
```

完整例子见 `examples/solve.c`（`cc examples/solve.c -Iinclude -Ltarget/release -lccm`）。

### 命令行

```bash
ccm-cli captcha.jpg              # 101-45,179-48,14-51,51-49 0.225842
ccm-cli -j a.jpg b.jpg           # 每张一行 JSON：{"file": ..., "points": [[x, y], ...], "margin": ...}
ccm-cli -b 20 samples/*.jpg      # 基准测试
ccm-cli -m models/s1.ccm x.jpg   # 换模型；文件名写 - 则从 stdin 读
```

输出的 `x-y,x-y,...` 就是 course-grabber 提交用的 verify 串格式。

## 3. 模型与量化

`.ccm` 是一个很小的权重格式（BatchNorm 已折叠进卷积，布局见 `tools/export_ccm.py` 开头）。
推理始终是 float32，量化只用来减小文件：加载时一次性反量化，不影响速度。

| 方案（w16） | 模型文件 | 答案变化 | margin 漂移（p99） |
| --- | ---: | ---: | ---: |
| fp32 | 649 KB | 0 | 0 |
| int8，就近取整 | 166 KB | 8 | 0.0080 |
| int8 + GPTQ | 166 KB | 0 | 0.0035 |
| **全连接 4 bit + 最后一层卷积 6 bit + GPTQ（内置）** | **124 KB** | **4** | **0.0082** |
| 全连接 5 bit + 最后一层卷积 6 bit，就近取整（无校准数据时的默认） | 132 KB | 7 | 0.0165 |
| 在内置方案基础上再把 4 个中间层降到 6 bit + GPTQ | 113 KB | 6 | 0.0107 |

（在 17,059 张未参与校准的实测图上统计，相对原版 fp32 的答案变化数。）

- 每个输出通道一个缩放系数，每层单独的位宽。提示字塔的全连接层占全部权重的 40%，也最耐低位宽；
  前几层卷积很敏感，保持 8 bit。
- GPTQ（Frantar et al., 2022）按前向顺序逐层量化：每取整一个权重，就把误差摊到还没取整的权重上，
  使这一层在真实验证码上的输出变化最小。它需要一份校准用的切字（原仓库 `prepare_real.py` 的输出）和 torch。
- 内置模型的校准数据是 19,107 张实测图中按文件名排序的前 2,048 张，上表在其余 17,059 张上评估。
- 想要与原版最接近：`--bits 8 --calib real.npz`（int8 + GPTQ，模型大 42 KB，上表中 0 处变化）。

```bash
python tools/export_ccm.py runs/w16/matcher.onnx models/w16.ccm --calib real.npz   # 内置模型的做法
python tools/export_ccm.py runs/s1/matcher.onnx models/s1.ccm --bits 8             # s1：int8
python tools/export_ccm.py runs/w16/model.pt w16-f32.ccm --dtype f32               # 不量化
```

读 `.onnx` 只需要 numpy（自带一个极简 protobuf 解析）；读 `.pt` 或用 `--calib` 需要 torch。
`models/` 里附带 `w16.ccm`（内置的那个）和 `s1.ccm`（int8，505 KB，需要时用 `-m` / 构造参数加载）。

## 4. 与原版一致性是怎么保证的

- **解码**：Pillow 用 libjpeg-turbo 的默认参数解码，再转灰度。这里逐步复刻了它的整数运算：
  ISLOW 整数 IDCT、"fancy" 色度上采样（h2v1 / h2v2）、YCbCr→RGB 的定点表，以及 Pillow 的
  `L = (19595 R + 38470 G + 7471 B + 0x8000) >> 16`。所以模型看到的像素与原版完全相同。
  支持 baseline / extended sequential Huffman、1 或 3 个分量、1×1 / 2×1 / 2×2 采样、restart marker；
  渐进式、12 bit、CMYK 等返回 `CCM_EUNSUPPORTED`。
- **切字**：`geometry.py` 逐行移植，连 numpy 的 float64 运算顺序和 `round()` 的"四舍六入五成双"都一致。
- **网络**：float32，按 HWC 布局计算；只有累加顺序与 onnxruntime 不同（误差 10⁻⁷ 量级）。
- **分配**：按 `itertools.permutations` 的顺序枚举 360 种排列，打分方式与 numpy 相同。
- `tools/parity.py` 可以在你自己的样本上复查以上所有项：
  `python tools/parity.py --ref ../click-captcha-matcher samples/ oracle/w16`

## 5. 集成到 course-grabber（已实测，未改动其仓库）

在 course-grabber 的一个临时副本里验证过：把 `python/solver.py` 和 `libccm.so` 打进包里，
去掉 onnxruntime / numpy / Pillow，单文件包从 41.3 MB 降到 8.6 MB，打包后的可执行文件能正确解出
服务器验证过的验证码。改动只有三处：

1. `school_auth.CaptchaSolver.__init__` 开头：能 `import solver` 且它是本仓库的封装（有 `load_library`）时，
   直接 `solver.CaptchaSolver(None)`，跳过模型文件与 onnxruntime 的检查；
2. `packaging/build.py`：`--add-binary libccm.so:.`、`--hidden-import solver`，并排除
   `onnxruntime`、`numpy`、`PIL`（Windows 上是 `ccm.dll`，macOS 上是 `libccm.dylib`）；
3. Release 工作流里先 `cargo build --release` 本仓库（或下载本仓库 CI 的产物）。

顺带解决了 onnxruntime 在 HOME 不可写时往当前目录写 `:memory:.ses` 的问题：已经没有 onnxruntime 了。

## 6. 平台

| 平台 | 状态 |
| --- | --- |
| Linux x86_64 | 构建、测试、2 万张对照、基准 |
| macOS arm64（Apple M4） | 构建、测试、基准，系统自带 Python 3.9 可直接用 `solver.py` |
| Linux arm64、macOS x86_64、Windows x64 / arm64 | 本地编译检查通过；`.github/workflows/ci.yml` 会在 GitHub 上构建与测试 |

`libccm.so` 只要求 glibc ≥ 2.14；`ccm-cli` 取决于构建机的 glibc（CI 在 Ubuntu 22.04 上构建）。
`.cargo/config.toml` 里有几项体积设置：Linux 去掉 unwind 表与 build-id（panic 直接 abort，不需要展开），
macOS 把 dylib 的 install name 设为 `@rpath/libccm.dylib`。

## 7. 文件

| 文件 | 作用 |
| --- | --- |
| `src/jpeg.rs` | 与 Pillow 逐像素一致的 baseline JPEG 解码 |
| `src/geometry.rs` | 固定版面切字 + 点击中心（`geometry.py` 的移植） |
| `src/model.rs` | `.ccm` 加载与双塔前向 |
| `src/kernel.rs` | 唯一的计算核心：AVX2 / NEON / 可移植三个版本 |
| `src/assign.rs` | 360 种排列的分配与 margin |
| `src/capi.rs`、`include/ccm.h` | C ABI |
| `src/main.rs` | `ccm-cli` |
| `src/rt.rs` | `no_std` 所需：libc 分配器、panic 即 abort |
| `python/solver.py` | ctypes 封装（原 `solver.py` 接口） |
| `tools/export_ccm.py` | `.onnx` / `.pt` → `.ccm`（折叠 BN、按层位宽、可选 GPTQ） |
| `tools/parity.py` | 与原版 Python 求解器逐张对照 |
| `tools/make_fixtures.py` | 重新生成 `tests/fixtures` |
| `models/w16.ccm`、`models/s1.ccm` | 内置模型、s1 模型 |
| `tests/fixtures/` | 测试用图：用原仓库 `synth.py` 和开源字体（Droid Sans Fallback、Noto Sans CJK）合成的验证码，以及同一张图的各种 JPEG 变体；不含实测样本 |

## 8. 许可

MIT，见 `LICENSE`。
