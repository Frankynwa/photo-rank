# PhotoRank 改进执行手册

> **⚠️ 不可变约束 — 任何 agent 修改 main.py 前必读**
>
> 以下安全修复已验证，**任何后续修改必须保留，禁止删除或重写**：
>
> | 约束 | 位置 | 说明 |
> |------|------|------|
> | `secure_filename()` 函数 | main.py | 防路径遍历，所有接受文件名参数的路由必须调用 |
> | `MAX_UPLOAD_SIZE = 50 * 1024 * 1024` | main.py | 上传文件大小限制 |
> | `ALLOWED_EXTENSIONS` 白名单 | main.py | 只允许图片扩展名 |
> | `ALLOWED_MIME_TYPES` 白名单 | main.py | 只允许图片 MIME 类型 |
> | `asyncio.to_thread()` | main.py | 同步代码不阻塞事件循环 |
> | HTTP 500 错误脱敏 | main.py | 不向客户端泄露内部堆栈 |
> | `from config import ...` 统一配置 | main.py | 禁止本地重复定义 BASE_DIR 等 |
> | `except Exception as e:` | 全项目 | 禁止裸 `except:` 或 `except: pass` |
> | `dotenv + os.environ` 读配置 | layer5_analysis.py, config.py | 禁止硬编码路径读取密钥 |
>
> **违反以上约束的 commit 将被拒绝。**

---

## 概览

PhotoRank 是一个 Python 旅行照片 AI 智能筛选平台，采用五层流水线架构（客观指标 → 相似聚类 → 审美评分 → 人脸质量 → 深度分析）。项目刚完成了一轮 **P0 级紧急修复**，解决了安全漏洞、配置硬编码、依赖缺失和 asyncio 阻塞等关键问题。当前代码可以正常运行，但在工程健壮性、模型精度、架构设计等方面仍有较大提升空间。

本手册将剩余改进分为 **三个阶段 + 模型升级路线**，按优先级从高到低排列，每个任务都包含具体的操作步骤和学习要点，适合独立完成。

---

## 已完成的改进（P0 级修复）

> 以下修复已全部完成并验证，无需再改动。了解这些内容有助于理解当前代码状态。

### 安全漏洞修复
- `main.py`：添加 `secure_filename()` 函数（第 52-64 行），防止路径遍历攻击（如 `../../etc/passwd`）
- 文件上传添加 50MB 大小限制（`MAX_UPLOAD_SIZE`，第 40 行）
- 文件类型白名单校验：扩展名（第 43 行 `ALLOWED_EXTENSIONS`）+ MIME 类型（第 46 行 `ALLOWED_MIME_TYPES`）
- `/api/photos/{filename}` 和 `/api/results/` 路由也添加了路径遍历校验

### 配置硬编码消除
- `layer5_analysis.py`：删除硬编码 Windows `.env` 路径，改用 `python-dotenv` + `os.environ` 双路径读取（第 22-39 行）
- `config.py`：`PROXY`、`QWEN_BASE_URL`、`QWEN_MODEL` 改为从环境变量读取（第 57-67 行）
- `main.py`：消除与 `config.py` 的重复配置定义，统一 `from config import ...`（第 18-24 行）

### 依赖补全
- `requirements.txt` 补全 7 个缺失依赖：`opencv-python`、`numpy`、`imagehash`、`pyiqa`、`mediapipe`、`python-dotenv`、`pillow-heif`
- `torch` 以注释 + 安装说明形式保留（因 CUDA 版本匹配问题，不宜直接声明）

### 异常处理修复
- 所有裸 `except: pass` 改为 `except Exception as e:` + 日志记录
- HTTP 500 错误信息脱敏，不再返回内部堆栈给客户端（`main.py` 第 267 行）

### 并发阻塞修复
- `main.py` 第 246 行：`run_pipeline()` 改用 `asyncio.to_thread()` 放到线程池执行，不再阻塞事件循环

### 修改文件清单
| 文件 | 变更概述 |
|------|----------|
| `config.py` | 环境变量化、消除硬编码 |
| `core/layer5_analysis.py` | API Key 读取方式重构 |
| `main.py` | 安全校验、asyncio 修复、统一配置引用 |
| `requirements.txt` | 补全 7 个缺失依赖 |

> **统计**：4 files changed, 181 insertions(+), 43 deletions(-)

---

## 阅读指南

### 难度标记
- 🟢 **入门**：只需基础 Python 知识，适合第一次接触工程实践的同学
- 🟡 **进阶**：需要理解一些设计模式或第三方库，有一定挑战性
- 🔴 **挑战**：涉及架构设计或复杂集成，需要综合知识

### 每个任务包含
- **目标**：这个任务要达成什么
- **涉及文件**：需要修改哪些文件
- **当前问题**：现在代码怎么写的、为什么不好（附行号）
- **操作步骤**：具体到文件和位置的修改指引
- **学习要点**：核心概念解释
- **验收标准**：怎么算完成
- **参考代码**：关键片段（非完整实现）
- **预估耗时**：帮助你规划时间
- **前置依赖**：需要先完成哪些任务

---

## 关键知识点预习

### 1. Python logging 模块
Python 内置的结构化日志系统，比 `print` 强大得多。它支持日志级别（DEBUG < INFO < WARNING < ERROR < CRITICAL）、格式化输出、多目标（控制台 + 文件）、按模块分层控制。
- 📖 [官方教程](https://docs.python.org/3/howto/logging.html)
- 📖 [Logging Best Practices](https://coralogix.com/blog/python-logging-best-practices-tips/)

### 2. 优雅降级（Graceful Degradation）
当某个组件不可用时（如 GPU、MediaPipe），系统不崩溃，而是自动降级到可用的替代方案（如 CPU）。这是生产系统的基本素养。
- 📖 [Circuit Breaker Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)

### 3. PyTorch 设备管理
`torch.device` 控制模型和数据在 CPU 还是 GPU 上运行。`torch.cuda.is_available()` 检测 GPU 是否可用。理解这个对后续所有 GPU 相关任务至关重要。
- 📖 [PyTorch Device 文档](https://pytorch.org/docs/stable/tensorboard.html)

### 4. 图像质量评估（IQA）模型
- **NIMA**：Google 提出的神经网络图像美学评估，基于 VGG16  backbone，在 AVA 数据集上训练。SRCC（与人类判断的相关系数）约 0.657。
- **TOPIQ**：更先进的 IQA 模型，基于 Swin Transformer，SRCC 达 0.791。Swin Transformer 是一种分层视觉 Transformer，能捕获多尺度特征，比传统 CNN 更适合理解图像全局美感。
- 📖 [pyiqa 文档](https://github.com/chaofengc/IQA-PyTorch)

### 5. CLIP 模型
OpenAI 的 Contrastive Language-Image Pre-Training，能将图片和文本映射到同一向量空间。用于"语义级"相似度判断——两张内容不同但"感觉相似"的照片会被归为一组。
- 📖 [OpenAI CLIP](https://github.com/openai/CLIP)

### 6. Pydantic 数据验证
Python 最流行的数据验证库（FastAPI 的核心依赖）。用类型注解定义数据结构，自动验证、序列化。比 `dataclass` 多了运行时校验能力。
- 📖 [Pydantic V2 文档](https://docs.pydantic.dev/latest/)

### 7. asyncio 与并发控制
`asyncio.Lock` 和 `asyncio.Semaphore` 用于控制并发。`Lock` 保证同一时间只有一个任务执行某操作；`Semaphore` 允许限制同时执行的任务数量。
- 📖 [asyncio 同步原语](https://docs.python.org/3/library/asyncio-sync.html)

---

## 第一阶段：工程基础加固（P1，约 1 周）

---

### 1.1 引入 logging 替换 print 🟢

**目标**：用 Python `logging` 模块替换全项目约 30 处 `print()` 调用，实现结构化日志。

**涉及文件**：`core/pipeline.py`、`core/layer1_objective.py`、`core/layer2_similarity.py`、`core/layer3_aesthetic.py`、`core/layer4_face.py`、`core/layer5_analysis.py`、`main.py`

**当前问题**：
- `pipeline.py` 约 20 处 `print()`（第 27、30-36、41、46-50、55、64-66、72、76-79、84、90-91、105、115、134、139、155、164、170-183 行）
- `layer3_aesthetic.py` 第 12、14、43、45 行用 `print` 输出模型加载状态
- `layer4_face.py` 第 47、49、62、65、128 行用 `print` 输出初始化/分析状态
- `main.py` 第 85、262 行用 `print` 记录错误
- 问题：无法按级别过滤、无法输出到文件、没有统一格式、生产环境无法关闭调试输出

**操作步骤**：

1. **创建日志配置模块**：新建 `core/logger.py`，定义统一的日志配置函数：
   - 创建根 logger，设置级别为 `INFO`
   - 添加 `StreamHandler`（控制台输出）
   - 设置格式：`[%(asctime)s] %(levelname)-8s %(name)s - %(message)s`
   - 提供 `get_logger(name)` 便捷函数

2. **在各层模块顶部创建 logger**：
   - `pipeline.py`：`logger = logging.getLogger(__name__)`
   - `layer1_objective.py`：同上
   - 其他层模块同理

3. **逐个替换 print 为对应的 logger 调用**：
   - 进度信息 → `logger.info()`
   - 调试详情 → `logger.debug()`
   - 错误信息 → `logger.error()` 或 `logger.warning()`
   - 例如 `pipeline.py` 第 27 行 `print(f"未找到照片: {photos_dir}")` → `logger.warning(f"未找到照片: {photos_dir}")`

4. **在 `main.py` 启动时调用日志初始化**：
   - 在 `app = FastAPI(...)` 之前调用 `setup_logging()`

**学习要点**：
- `logging` 的级别体系：`DEBUG`（开发调试）< `INFO`（正常运行）< `WARNING`（异常但可继续）< `ERROR`（功能失败）< `CRITICAL`（系统级故障）
- `logger = logging.getLogger(__name__)` 的好处：自动以模块名为 logger 名称，可以按模块控制日志级别
- `Formatter` 控制输出格式，`Handler` 控制输出目标（控制台/文件/网络）

**验收标准**：
- [ ] 全项目无 `print()` 用于日志输出（`grep -rn "print(" core/ main.py` 应无结果或仅剩极少量合理用途）
- [ ] 控制台输出带时间戳、级别、模块名
- [ ] 可通过修改一个配置项切换日志级别（如改为 DEBUG 看到更多细节）

**参考代码**：
```python
# core/logger.py 关键片段
import logging

def setup_logging(level: str = "INFO"):
    """初始化统一日志配置"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

# 在各模块中使用
# pipeline.py 顶部
from core.logger import get_logger
logger = get_logger(__name__)

# 替换示例：
# 原来: print(f"[Layer 1] 客观指标分析...")
# 改为: logger.info("[Layer 1] 客观指标分析（清晰度/曝光/噪声）")
```

**预估耗时**：2-3 小时

**前置依赖**：无

---

### 1.2 Pipeline 每层添加 try/except 容错 🟢

**目标**：任一层执行失败时，记录错误并跳过该层，继续执行后续层，而不是整个流水线崩溃。

**涉及文件**：`core/pipeline.py`

**当前问题**：
- `pipeline.py` 第 42-98 行，Layer 1 到 Layer 4 的调用完全没有 `try/except` 包裹
- 只有 Layer 5（第 116-130 行）有逐张照片的 `try/except`
- 如果 Layer 2 聚类出错，整个流水线直接崩溃，Layer 3-5 全部无法执行

**操作步骤**：

1. **为 Layer 1-4 各添加 try/except 包裹**：
   - 在 `except` 中 `logger.error()` 记录错误
   - 设置该层结果为默认值（如空列表），让后续层可以继续
   - 例如 Layer 1 失败时，`l1_results = [{"path": str(p), "rejected": False, ...} for p in photos]`（全部保留，不淘汰）

2. **Layer 2 失败时的降级策略**：
   - 聚类失败 → 所有照片都作为独立代表保留（不做去重）

3. **Layer 3 失败时的降级策略**：
   - 审美评分失败 → 所有照片 `overall_score` 设为 0，按 Layer 4 分数排序

4. **Layer 4 失败时的降级策略**：
   - 人脸分析失败 → 所有照片 `has_face = False`，不调整排序

5. **在最终汇总中体现哪些层被跳过**：
   - 添加一个 `skipped_layers` 列表到返回结果中

**学习要点**：
- **容错设计**（Fault Tolerance）：系统部分组件失败时仍能运行，是生产系统的基本要求
- **降级策略**（Degradation）：失败时不是简单地跳过，而是提供合理的默认值，保证后续流程的输入完整
- 注意：容错不是掩盖错误——一定要记录完整的错误信息

**验收标准**：
- [ ] 模拟 Layer 1 异常（如传入不存在的路径），流水线不崩溃
- [ ] 被跳过的层在最终结果中有明确标记
- [ ] 日志中有清晰的错误记录

**参考代码**：
```python
# pipeline.py 中 Layer 1 容错示例
try:
    logger.info("[Layer 1] 客观指标分析...")
    l1_results = batch_analyze(photos)
    kept = [r for r in l1_results if not r["rejected"]]
    rejected = [r for r in l1_results if r["rejected"]]
    logger.info(f"  保留: {len(kept)} 张, 淘汰: {len(rejected)} 张")
except Exception as e:
    logger.error(f"[Layer 1] 客观指标分析失败，跳过该层: {e}")
    # 降级：全部保留，不做淘汰
    kept = [{"path": str(p), "filename": Path(p).name, "rejected": False,
             "sharpness": 0, "reject_reason": "Layer 1 跳过"} for p in photos]
    rejected = []
    skipped_layers.append("Layer 1")
```

**预估耗时**：2-3 小时

**前置依赖**：建议先完成 1.1（logging），这样容错日志可以输出到结构化日志系统

---

### 1.3 Layer 3 GPU 检测 + CPU fallback 🟢

**目标**：启动时自动检测 GPU 是否可用，不可用时自动降级到 CPU，并输出警告。

**涉及文件**：`core/layer3_aesthetic.py`

**当前问题**：
- 第 13 行：`model = pyiqa.create_metric("nima-vgg16-ava", device="cuda")` 硬编码 `device="cuda"`
- 在没有 GPU 的机器上（如 Mac、无显卡的服务器），直接崩溃
- 第 45 行：`torch.cuda.memory_allocated()` 在非 CUDA 环境也会报错

**操作步骤**：

1. **在 `score_photos()` 函数开头添加设备检测**：
   ```python
   if torch.cuda.is_available():
       device = "cuda"
       logger.info("检测到 CUDA，使用 GPU 推理")
   else:
       device = "cpu"
       logger.warning("未检测到 CUDA，降级到 CPU 推理（速度较慢）")
   ```

2. **将第 13 行的 `device="cuda"` 改为 `device=device`**

3. **修复第 45 行的显存统计**：
   - 用 `if device == "cuda":` 条件包裹 `torch.cuda.memory_allocated()` 调用

4. **将设备信息写入返回结果**：
   - 每条结果 dict 中添加 `"device": device` 字段

**学习要点**：
- `torch.cuda.is_available()` 是检测 GPU 的标准方法
- **优雅降级**：不是"有 GPU 就用，没有就崩"，而是"有 GPU 就用，没有就慢一点"
- CPU 推理速度大约是 GPU 的 1/10 ~ 1/20，对于少量照片（< 50 张）可以接受

**验收标准**：
- [ ] 在无 GPU 环境（如 Mac）上能正常运行
- [ ] 控制台有明确的设备选择提示
- [ ] 结果中包含 `device` 字段

**参考代码**：
```python
# layer3_aesthetic.py 修改要点
def score_photos(image_paths: list[str | Path]) -> list[dict]:
    import pyiqa
    import torch

    # 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    model = pyiqa.create_metric("nima-vgg16-ava", device=device)

    # ... 评分循环 ...

    # 显存统计（仅 CUDA）
    if device == "cuda":
        logger.info(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

**预估耗时**：1 小时

**前置依赖**：建议先完成 1.1（logging）

---

### 1.4 Layer 4 初始化失败重试逻辑 🟢

**目标**：修复 MediaPipe 初始化失败后仍标记为已初始化的 bug，并添加重试机制。

**涉及文件**：`core/layer4_face.py`

**当前问题**：
- 第 50 行：`self.initialized = True` 在 `except` 块中也被执行——初始化失败后仍标记为成功
- 后续调用 `_init_mediapipe()` 时第 28 行 `if self.initialized: return` 直接返回，永远不会重试
- 结果：如果第一次初始化因网络问题（模型下载失败）失败，后续所有人脸分析都会静默返回空结果

**操作步骤**：

1. **修复第 50 行的 bug**：
   - 将 `except` 块中的 `self.initialized = True` 改为 `self.initialized = False`
   - 这样下次调用时会重新尝试初始化

2. **添加重试计数器**（防止无限重试）：
   - 添加 `self._init_attempts = 0` 到 `__init__`
   - 在 `_init_mediapipe()` 中检查：如果已尝试 3 次仍失败，直接返回不再重试

3. **在 `analyze_face()` 中改进错误处理**：
   - 第 73 行 `if self.detector is None` 时，记录 warning 而非静默返回

**学习要点**：
- **重试模式**（Retry Pattern）：网络请求、模型下载等外部依赖经常因临时故障失败，自动重试能大幅提高成功率
- **重试上限**：无限重试会导致死循环，必须设置最大重试次数
- **失败标记**：初始化状态必须准确——成功才标记为 `True`，失败必须保持 `False`

**验收标准**：
- [ ] 模拟初始化失败（如断网），不会永久标记为已初始化
- [ ] 最多重试 3 次
- [ ] 日志中有重试记录

**参考代码**：
```python
# layer4_face.py FaceAnalyzer.__init__ 修改
def __init__(self):
    self.detector = None
    self.initialized = False
    self._init_attempts = 0
    self._max_init_attempts = 3

def _init_mediapipe(self):
    if self.initialized:
        return
    if self._init_attempts >= self._max_init_attempts:
        logger.warning(f"MediaPipe 初始化已失败 {self._max_init_attempts} 次，不再重试")
        return

    self._init_attempts += 1
    try:
        # ... 原有初始化代码 ...
        self.initialized = True  # 只在成功时标记
    except Exception as e:
        logger.error(f"MediaPipe 初始化失败 (第 {self._init_attempts} 次): {e}")
        self.initialized = False  # 失败保持 False
```

**预估耗时**：1-2 小时

**前置依赖**：建议先完成 1.1（logging）

---

### 1.5 并发任务控制（队列/互斥） 🟡

**目标**：确保同一时间只有一个分析任务在运行，防止 GPU 显存溢出。

**涉及文件**：`main.py`

**当前问题**：
- `main.py` 第 234-267 行的 `/api/analyze` 端点没有任何并发控制
- 如果用户快速点击两次"分析"按钮，两个 `run_pipeline()` 会同时在线程池中运行
- 两个 NIMA 模型同时加载到 GPU → 显存溢出（OOM）

**操作步骤**：

1. **在 `main.py` 顶部创建全局锁**：
   ```python
   _analysis_lock = asyncio.Lock()
   ```

2. **在 `analyze_photos()` 函数中加锁**：
   - 获取锁时如果已被占用，返回 409 Conflict 或排队等待
   - 推荐方案：尝试获取锁，获取失败则返回 HTTP 409 "分析任务正在运行中"

3. **确保异常时也能释放锁**：
   - 使用 `async with` 语法自动管理

**学习要点**：
- `asyncio.Lock` 是异步互斥锁，`async with` 保证自动释放
- 为什么不用 `threading.Lock`？因为 `analyze_photos()` 是 async 函数，用的是 asyncio 事件循环
- **409 Conflict**：HTTP 状态码，表示请求与服务器当前状态冲突

**验收标准**：
- [ ] 同时发起两个分析请求，第二个返回 409 或排队等待
- [ ] 分析完成后锁被正确释放，可以发起新请求
- [ ] 异常情况下锁也能正确释放

**参考代码**：
```python
# main.py 中添加
_analysis_lock = asyncio.Lock()

@app.post("/api/analyze")
async def analyze_photos(platform: str = "xiaohongshu"):
    if _analysis_lock.locked():
        raise HTTPException(status_code=409, detail="分析任务正在运行中，请等待完成")

    async with _analysis_lock:
        # ... 原有的分析逻辑 ...
```

**预估耗时**：1-2 小时

**前置依赖**：无

---

## 第二阶段：模型升级（约 2-3 天）

> 这一阶段的改进都是"换模型"，核心思路是用更先进的 AI 模型替换当前模型，大幅提升分析精度。

---

### 2.1 Layer 3：NIMA-VGG16 → TOPIQ-IAA ★★★★★ 🟢

**目标**：审美评分 SRCC 从 0.657 提升至 0.791（+20%），收益最大、成本最低。

**涉及文件**：`core/layer3_aesthetic.py`

**当前代码**：第 13 行 `model = pyiqa.create_metric("nima-vgg16-ava", device="cuda")`

**操作步骤**：

1. **确认 pyiqa 版本支持 topiq_iaa**：
   ```bash
   pip show pyiqa  # 确认版本 >= 0.9.0
   python -c "import pyiqa; m = pyiqa.create_metric('topiq_iaa'); print('OK')"
   ```

2. **修改模型名称**（仅改一行）：
   ```python
   # 原来
   model = pyiqa.create_metric("nima-vgg16-ava", device=device)
   # 改为
   model = pyiqa.create_metric("topiq_iaa", device=device)
   ```

3. **更新模块顶部注释**（第 1 行）：
   ```python
   """Layer 3: 审美评分 — TOPIQ-IAA (Swin Transformer)"""
   ```

4. **更新 pipeline.py 第 72 行的打印信息**：
   ```python
   logger.info("[Layer 3] 审美评分（TOPIQ-IAA）...")
   ```

5. **测试验证**：用几张已知照片对比新旧模型的评分差异

**学习要点**：
- **NIMA**（Neural Image Assessment）：Google 2017 年提出，用 VGG16 作为 backbone 提取图像特征，再接全连接层预测美学分数。VGG16 是 2014 年的经典 CNN，特征提取能力有限。
- **TOPIQ**：基于 **Swin Transformer** 的 IQA 模型。Swin Transformer 用"移动窗口"机制实现高效的多尺度特征提取，能同时理解局部细节和全局构图，比 VGG16 更适合评判"一张照片好不好看"。
- **SRCC**（Spearman Rank Correlation Coefficient）：衡量模型打分与人类判断的排序一致性。0.657 → 0.791 意味着模型排名与人类排名的一致性从 65.7% 提升到 79.1%。

**验收标准**：
- [ ] `pyiqa.create_metric('topiq_iaa')` 能正常加载
- [ ] 评分结果在合理范围内（通常 1-10 分）
- [ ] 对比 NIMA 和 TOPIQ 对同一组照片的评分，确认 TOPIQ 排序更合理

**预估耗时**：0.5 天

**前置依赖**：建议先完成 1.3（GPU fallback），这样 TOPIQ 也自动支持 CPU

---

### 2.2 Layer 1：增加 TOPIQ-NR 综合质量评分 ★★★★ 🟡

**目标**：在现有 OpenCV 子维度（清晰度/曝光/噪声）基础上，增加一个 AI 综合质量评分，整体质量判断精度提升 30-50%。

**涉及文件**：`core/layer1_objective.py`

**当前代码**：
- 第 45-48 行：`compute_sharpness()` 用 Laplacian 方差
- 第 51-64 行：`compute_exposure()` 用直方图统计
- 第 67-76 行：`compute_noise()` 用信噪比估计
- 这些是"手工规则"，精度有限

**操作步骤**：

1. **在 `analyze_image()` 函数中增加 TOPIQ-NR 评分**：
   - 在函数开头（加载图片后），调用 `pyiqa.create_metric('topiq_nr')` 获取综合质量分
   - 模型只需加载一次（用模块级变量或函数属性缓存）

2. **将 TOPIQ-NR 分数加入 `ObjectiveScore` dataclass**：
   - 新增字段 `topiq_score: float`

3. **修改综合分计算**（第 95 行）：
   - 原来：`overall = sharpness_normalized * 0.4 + exposure * 0.3 + noise * 0.3`
   - 改为：`overall = topiq_score * 0.5 + sharpness_normalized * 0.2 + exposure * 0.15 + noise * 0.15`
   - TOPIQ-NR 权重最高，因为它是最全面的指标

4. **模型加载优化**：
   - 用模块级变量缓存模型实例，避免每张图片都重新加载
   - 复用 1.3 中的 GPU/CPU 自动检测逻辑

**学习要点**：
- **NR-IQA**（No-Reference IQA）：不需要"完美参考图"，直接评估图片本身的质量。TOPIQ-NR 是这类模型中的佼佼者。
- **为什么保留 OpenCV 子维度？** TOPIQ-NR 给出的是综合分，但不知道具体哪里不好。OpenCV 子维度能告诉你"这张照片是因为模糊还是过曝被淘汰的"，可解释性更好。
- **混合策略**：AI 模型负责"准"，手工规则负责"快"和"可解释"，两者互补。

**验收标准**：
- [ ] 每张图片的评分结果中包含 `topiq_score` 字段
- [ ] 综合排序比纯 OpenCV 规则更合理
- [ ] 模型只加载一次（检查日志中不应出现多次"加载模型"）

**参考代码**：
```python
# layer1_objective.py 中添加模型缓存
_toriq_model = None
_toriq_device = None

def _get_topiq_model():
    global _toriq_model, _toriq_device
    if _toriq_model is None:
        import pyiqa, torch
        _toriq_device = "cuda" if torch.cuda.is_available() else "cpu"
        _toriq_model = pyiqa.create_metric('topiq_nr', device=_toriq_device)
    return _toriq_model, _toriq_device
```

**预估耗时**：1-2 天

**前置依赖**：建议先完成 1.3（GPU fallback）和 2.1（TOPIQ-IAA，熟悉 pyiqa 用法）

---

### 2.3 Layer 5：升级 Qwen3-VL ★★★★ 🟢

**目标**：将视觉语言模型从 Qwen-VL Max 升级到 Qwen3-VL，提升分类、构图分析、文案生成的质量。

**涉及文件**：`config.py`

**当前代码**：`config.py` 第 61 行 `QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-vl-max")`

**操作步骤**：

1. **确认 Qwen3-VL 的模型名称**：
   - 查阅阿里云 DashScope 文档，确认最新模型名称（如 `qwen-vl-max-latest` 或 `qwen3-vl-max`）
   - 确认 API 兼容性（Qwen3-VL 应该兼容现有 API 格式）

2. **修改默认模型名称**：
   ```python
   # config.py 第 61 行
   QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-vl-max")  # 改为最新模型名
   ```

3. **检查 `layer5_analysis.py` 中的 API 调用**：
   - 第 77-93 行的 `call_qwen_vl()` 函数——确认 API 格式兼容
   - 如果 Qwen3-VL 需要不同的参数（如 `temperature`、`max_tokens`），相应调整

4. **测试验证**：用 2-3 张照片测试分类和文案生成质量

**学习要点**：
- **视觉语言模型**（VLM）：同时理解图片和文字的 AI 模型。Qwen-VL 系列能"看懂"照片内容并用文字描述。
- **API 兼容性**：大多数模型升级只是换了"大脑"，"接口"不变。这就是为什么只改模型名就能升级——`layer5_analysis.py` 的 `call_qwen_vl()` 不需要改。
- **环境变量覆盖**：`os.environ.get("QWEN_MODEL", "默认值")` 的设计让你不改代码就能切换模型。

**验收标准**：
- [ ] 分类结果合理（8 类中正确归类）
- [ ] 文案生成质量有提升（更自然、更有深度）
- [ ] 无 API 报错

**预估耗时**：0.5 天

**前置依赖**：无

---

### 2.4 Layer 2：pHash + CLIP 两级聚类（进阶） ★★★ 🟡

**目标**：在现有 pHash 像素级去重基础上，增加 CLIP 语义级聚类，让系统能理解"这两张照片拍的是同一个场景"。

**涉及文件**：`core/layer2_similarity.py`、`requirements.txt`

**当前代码**：第 19-26 行 `compute_phash()` 和第 29-80 行 `cluster_photos()` 仅用感知哈希

**操作步骤**：

1. **添加 CLIP 依赖**：
   ```
   # requirements.txt 中添加
   clip @ git+https://github.com/openai/CLIP.git
   # 或使用 transformers 库
   transformers>=4.30.0
   ```

2. **实现 CLIP 特征提取函数**：
   - 加载 `clip ViT-B/32` 模型
   - 对每张图片提取 512 维特征向量

3. **修改 `cluster_photos()` 为两级聚类**：
   - **第一级**（保留）：pHash 去近似重复（汉明距离 ≤ 10）
   - **第二级**（新增）：对各级代表照片用 CLIP 特征做语义聚类（余弦相似度 > 0.85 视为同一主题）

4. **返回结果中添加聚类类型标记**：
   - `"cluster_type": "pixel_duplicate"` 或 `"semantic_similar"`

**学习要点**：
- **pHash vs CLIP**：pHash 比较像素级相似度（两张照片看起来像不像），CLIP 比较语义相似度（两张照片"表达的内容"像不像）。例如：同一座建筑从不同角度拍，pHash 认为不像，CLIP 认为像。
- **CLIP 模型**：OpenAI 的对比学习模型，将图片和文本映射到同一向量空间。这里只用它的图片编码器。
- **两级策略**：先用 pHash 快速去重（毫秒级），再用 CLIP 精细聚类（百毫秒级），兼顾速度和精度。

**验收标准**：
- [ ] 不同角度的同一场景能被 CLIP 聚类到一起
- [ ] 完全不同的场景不会被错误合并
- [ ] pHash 去重仍然正常工作

**预估耗时**：2-3 天

**前置依赖**：建议先完成 1.3（GPU fallback），CLIP 推理需要 GPU

---

### 2.5 Layer 4：MediaPipe → InsightFace（进阶） ★★★ 🔴

**目标**：人脸检测准确率从 AP 0.743 提升到 0.95+，漏检率降低约 25%。

**涉及文件**：`core/layer4_face.py`、`requirements.txt`

**当前代码**：第 26-50 行 `FaceAnalyzer._init_mediapipe()` 和第 69-129 行 `analyze_face()`

**操作步骤**：

1. **添加 InsightFace 依赖**：
   ```
   # requirements.txt
   insightface>=0.7.0
   onnxruntime>=1.15.0  # InsightFace 的后端
   ```

2. **重写 `FaceAnalyzer` 类**：
   - `_init_mediapipe()` → `_init_insightface()`
   - 使用 `insightface.app.FaceAnalysis` 加载 `buffalo_l` 模型
   - 模型自动下载到 `~/.insightface/models/`

3. **修改 `analyze_face()` 方法**：
   - InsightFace 返回的 `face` 对象包含 `bbox`、`kps`（关键点）、`det_score`（检测置信度）
   - 闭眼检测：改用关键点间距（EAR，Eye Aspect Ratio）
   - 人脸清晰度：类似现有逻辑，裁剪人脸区域算 Laplacian

4. **保持 `FaceQuality` dataclass 和 `batch_analyze()` 接口不变**：
   - 这样 `pipeline.py` 不需要任何修改

**学习要点**：
- **MediaPipe vs InsightFace**：MediaPipe 是 Google 的轻量级方案，速度快但精度有限；InsightFace 是学术界最全面的人脸分析框架，精度更高但更重。
- **AP**（Average Precision）：目标检测的标准指标。0.743 → 0.95+ 意味着每 100 张含人脸的照片，少漏检约 21 张。
- **ONNX Runtime**：InsightFace 使用 ONNX 格式的模型，通过 ONNX Runtime 推理，支持 CPU 和 GPU。

**验收标准**：
- [ ] InsightFace 能正常检测人脸
- [ ] 闭眼检测准确率提升
- [ ] `pipeline.py` 无需修改（接口兼容）

**预估耗时**：2-3 天

**前置依赖**：建议先完成 1.4（初始化重试逻辑），InsightFace 模型下载也可能失败

---

## 第三阶段：架构优化（P2，约 2-4 周）

---

### 3.1 层间数据传递改为 Pydantic model / dataclass 🟡

**目标**：用结构化类型替代 dict 传递层间数据，获得类型检查和自动补全。

**涉及文件**：所有 `core/layer*.py`、`core/pipeline.py`

**当前问题**：
- `pipeline.py` 第 56 行：`sharpness_scores = {r["path"]: r["sharpness"] for r in kept}` 用字符串键访问
- 各层的 `batch_analyze()` 返回 `list[dict]`，字段名无约束
- 已有 `ObjectiveScore`（layer1 第 17 行）、`SimilarityCluster`（layer2 第 11 行）、`FaceQuality`（layer4 第 9 行）等 dataclass，但最终都被序列化为 dict

**操作步骤**：

1. **定义统一的层间数据模型**（建议新建 `core/models.py`）：
   ```python
   from pydantic import BaseModel

   class PhotoResult(BaseModel):
       path: str
       filename: str
       # Layer 1 字段
       sharpness: float = 0
       exposure: float = 0
       # ... 所有层可能添加的字段，给默认值
   ```

2. **各层返回 Pydantic model 实例而非 dict**

3. **pipeline.py 中用属性访问替代字典访问**：
   - `r["path"]` → `r.path`
   - `r["sharpness"]` → `r.sharpness`

**学习要点**：
- **Pydantic vs dataclass**：`dataclass` 只提供 `__init__` 和 `__repr__`；Pydantic 额外提供运行时类型验证、自动类型转换、JSON 序列化。FastAPI 就用 Pydantic。
- **类型安全的好处**：写 `r.sharpness` 时 IDE 能自动补全，写 `r.shapness`（拼写错误）时 IDE 会标红。用 dict 则运行时才会发现 KeyError。

**验收标准**：
- [ ] 定义统一的 `PhotoResult` 模型
- [ ] 各层返回该模型的实例
- [ ] `pipeline.py` 用属性访问

**预估耗时**：1-2 天

**前置依赖**：无

---

### 3.2 Pipeline 断点续传 🟡

**目标**：流水线中途失败后，重启时能从上次完成的层继续，不需要从头开始。

**涉及文件**：`core/pipeline.py`

**当前问题**：
- `run_pipeline()` 每次从头执行所有 5 层
- 如果 Layer 5（最耗时的一步，要逐张调用 API）在第 8/12 张时失败，重启后 Layer 1-4 全部重跑

**操作步骤**：

1. **每层完成后保存中间结果到 JSON**：
   - 路径：`output/{platform}/checkpoint_layer{N}.json`
   - 内容：该层的完整输出

2. **`run_pipeline()` 开头检查已有 checkpoint**：
   - 如果 `checkpoint_layer1.json` 存在且有效，直接加载跳过 Layer 1
   - 以此类推

3. **所有层完成后删除 checkpoint 文件**

4. **添加 `--force` 参数忽略 checkpoint**（可选）

**学习要点**：
- **持久化中间状态**是长流程系统的常见需求。类似数据库事务的"保存点"。
- 注意：中间结果可能因为代码更新而格式不兼容，需要版本号机制。

**验收标准**：
- [ ] Layer 5 中途失败后重启，Layer 1-4 不重新执行
- [ ] `--force` 参数能强制从头开始
- [ ] checkpoint 文件在所有层完成后被清理

**预估耗时**：1-2 天

**前置依赖**：依赖 1.2（Pipeline 容错）先完成

---

### 3.3 WebSocket 逐层进度推送 + 重连机制 🟡

**目标**：前端能展示逐层进度条（如"Layer 3/5: 审美评分中..."），并在 WebSocket 断线后自动重连。

**涉及文件**：`core/pipeline.py`、`main.py`、`templates/index.html`

**当前问题**：
- `main.py` 第 238/249/263 行只推送 `analysis_start`、`analysis_complete`、`analysis_error` 三种事件
- 前端无法知道当前执行到哪一层
- WebSocket 断线后无重连逻辑

**操作步骤**：

1. **pipeline.py 添加进度回调参数**：
   ```python
   def run_pipeline(photos_dir=None, platform="xiaohongshu",
                    progress_callback=None):
       # 每层开始/完成时调用
       if progress_callback:
           progress_callback("layer_start", {"layer": 3, "name": "审美评分"})
   ```

2. **main.py 中通过 WebSocket 转发进度**：
   ```python
   def ws_progress(event, data):
       asyncio.run(manager.broadcast({"type": event, **data}))

   result = await asyncio.to_thread(run_pipeline, platform=platform,
                                     progress_callback=ws_progress)
   ```

3. **前端添加 WebSocket 重连**：
   ```javascript
   function connectWebSocket() {
       const ws = new WebSocket("ws://localhost:8080/ws");
       ws.onclose = () => setTimeout(connectWebSocket, 3000); // 3秒后重连
   }
   ```

**学习要点**：
- **回调函数**（Callback）：一个函数作为参数传给另一个函数，在特定时机被调用。这是解耦的好方法——pipeline 不需要知道 WebSocket 的存在。
- **WebSocket 重连**：网络不稳定时 WebSocket 可能断开，客户端必须有自动重连机制。指数退避（1s → 2s → 4s → 8s）是常见策略。

**验收标准**：
- [ ] 前端显示当前执行到哪一层
- [ ] WebSocket 断线后 3 秒内自动重连
- [ ] 重连后能收到最新状态

**预估耗时**：2-3 天

**前置依赖**：无

---

### 3.4 创建 tests/ 目录 + 核心层单元测试 🟡

**目标**：为 Layer 1-4 的纯函数编写单元测试，确保修改不会引入回归 bug。

**涉及文件**：新建 `tests/` 目录

**当前问题**：项目零测试。任何代码修改都只能靠"跑一遍看看"来验证。

**操作步骤**：

1. **创建目录结构**：
   ```
   tests/
   ├── __init__.py
   ├── test_layer1.py
   ├── test_layer2.py
   ├── test_layer3.py  (需要 GPU，可标记 skip)
   └── test_layer4.py
   ```

2. **安装测试框架**：
   ```
   # requirements.txt 添加
   pytest>=7.0.0
   ```

3. **为 Layer 1 纯函数编写测试**（最简单，不需要模型）：
   ```python
   # tests/test_layer1.py
   from core.layer1_objective import compute_sharpness, compute_exposure, compute_noise

   def test_compute_sharpness_returns_positive():
       # 用 numpy 创建一张测试图片
       import numpy as np
       image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
       result = compute_sharpness(image)
       assert result > 0

   def test_compute_exposure_normal_image():
       # 中等亮度图片
       image = np.full((100, 100, 3), 128, dtype=np.uint8)
       score, is_over, is_under = compute_exposure(image)
       assert 0 <= score <= 1
       assert not is_over
       assert not is_under
   ```

4. **为 Layer 2 编写测试**：
   - 测试 `cluster_photos()` 对完全相同的图片能正确聚类

5. **运行测试**：`pytest tests/ -v`

**学习要点**：
- **单元测试**（Unit Test）：测试最小代码单元（通常是一个函数），验证输入→输出是否符合预期。
- **pytest**：Python 最流行的测试框架。函数名以 `test_` 开头就会被自动发现。
- **纯函数优先测试**：`compute_sharpness()` 这种输入→输出、无外部依赖的函数最容易测试。需要加载模型的函数可以加 `@pytest.mark.skip` 跳过。

**验收标准**：
- [ ] `pytest tests/ -v` 全部通过
- [ ] 至少覆盖 Layer 1 的所有纯函数
- [ ] Layer 2 至少一个聚类测试

**预估耗时**：2-3 天

**前置依赖**：无

---

### 3.5 添加 pyproject.toml + ruff 配置 🟢

**目标**：统一项目配置，添加代码风格检查工具。

**涉及文件**：新建 `pyproject.toml`

**当前问题**：项目无代码风格配置，不同文件风格可能不一致。

**操作步骤**：

1. **创建 `pyproject.toml`**：
   ```toml
   [project]
   name = "photorank"
   version = "0.1.0"
   requires-python = ">=3.11"

   [tool.ruff]
   line-length = 100
   target-version = "py311"

   [tool.ruff.lint]
   select = ["E", "F", "W", "I"]  # 基础规则 + import 排序
   ```

2. **安装 ruff**：`pip install ruff`

3. **运行检查**：`ruff check .`

4. **自动修复**：`ruff check . --fix`

**学习要点**：
- **ruff**：用 Rust 写的超快 Python linter，替代 flake8 + isort + black 的组合。
- **pyproject.toml**：Python 项目的标准配置文件（PEP 518），替代散落的 `setup.py`、`setup.cfg`、`.flake8` 等。

**验收标准**：
- [ ] `ruff check .` 无错误（或只有少量 warning）
- [ ] import 顺序统一

**预估耗时**：1-2 小时

**前置依赖**：无

---

## 第四阶段：部署与运维（P3，可选）

---

### 4.1 GitHub Actions CI 🟡

**目标**：每次 push 自动运行测试和代码检查。

**涉及文件**：新建 `.github/workflows/ci.yml`

**操作步骤**：

1. **创建 workflow 文件**：
   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.11" }
         - run: pip install -r requirements.txt
         - run: pip install pytest ruff
         - run: ruff check .
         - run: pytest tests/ -v
   ```

2. **推送到 GitHub 验证 CI 是否通过**

**学习要点**：
- **CI**（Continuous Integration）：每次代码变更自动运行测试，及早发现问题。
- GitHub Actions 的 YAML 语法：`on` 定义触发条件，`jobs` 定义任务，`steps` 定义步骤。

**验收标准**：
- [ ] push 后 GitHub Actions 自动运行
- [ ] 测试通过显示绿色 ✓，失败显示红色 ✗

**预估耗时**：1-2 小时

**前置依赖**：依赖 3.4（测试）先完成

---

### 4.2 Dockerfile 容器化 🟡

**目标**：一键启动整个应用，不需要手动安装依赖。

**涉及文件**：新建 `Dockerfile`、`.dockerignore`

**操作步骤**：

1. **创建 Dockerfile**：
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8080
   CMD ["python", "main.py"]
   ```

2. **创建 `.dockerignore`**：排除 `uploads/`、`output/`、`__pycache__/`、`.env`

3. **构建和运行**：
   ```bash
   docker build -t photorank .
   docker run -p 8080:8080 --env-file .env photorank
   ```

**学习要点**：
- **Docker**：将应用及其所有依赖打包成一个容器，在任何机器上都能一致运行。
- **多阶段构建**：可以先安装编译依赖，再只拷贝运行需要的文件，减小镜像体积。

**验收标准**：
- [ ] `docker build` 成功
- [ ] `docker run` 后能访问 `http://localhost:8080`

**预估耗时**：1 天

**前置依赖**：无

---

### 4.3 前端拆分（CSS/JS 独立） 🟢

**目标**：将 `templates/index.html` 中的内联 CSS 和 JS 拆分为独立文件。

**涉及文件**：`templates/index.html`、新建 `static/style.css`、`static/app.js`

**当前问题**：`index.html` 是单文件应用，CSS 和 JS 全部内联，难以维护和缓存。

**操作步骤**：

1. **提取 CSS 到 `static/style.css`**

2. **提取 JS 到 `static/app.js`**

3. **在 `main.py` 中挂载静态文件**：
   ```python
   app.mount("/static", StaticFiles(directory="static"), name="static")
   ```

4. **在 HTML 中引用**：
   ```html
   <link rel="stylesheet" href="/static/style.css">
   <script src="/static/app.js"></script>
   ```

**学习要点**：
- **静态文件服务**：FastAPI 的 `StaticFiles` 中间件可以直接提供 CSS/JS/图片等静态文件。
- **浏览器缓存**：独立文件可以被浏览器缓存，用户再次访问时不需要重新下载。

**验收标准**：
- [ ] 前端功能不变
- [ ] CSS 和 JS 为独立文件
- [ ] 浏览器开发者工具中无 404 错误

**预估耗时**：2-3 小时

**前置依赖**：无

---

## 附录

### A. 模型升级对比数据

| 层级 | 当前模型 | 推荐模型 | 关键指标提升 | 实施成本 |
|------|----------|----------|-------------|----------|
| Layer 1 | OpenCV 手工规则 | + TOPIQ-NR | SRCC ~0.5 → ~0.85+ | 1-2 天 |
| Layer 2 | pHash 像素级 | + CLIP 语义级 | 语义能力从 0 到有 | 2-3 天 |
| Layer 3 | NIMA-VGG16 (SRCC 0.657) | TOPIQ-IAA (SRCC 0.791) | **+20% 精度** | **0.5 天** |
| Layer 4 | MediaPipe (AP 0.743) | InsightFace (AP 0.95+) | 漏检率 -25% | 2-3 天 |
| Layer 5 | Qwen-VL Max | Qwen3-VL | +10-15% 质量 | 0.5 天 |

**升级优先级排序**（按收益/成本比）：
1. ★★★★★ Layer 3 TOPIQ-IAA（0.5 天，+20% 精度）
2. ★★★★★ Layer 1 TOPIQ-NR（1-2 天，+30-50% 精度）
3. ★★★★ Layer 5 Qwen3-VL（0.5 天，+10-15% 质量）
4. ★★★ Layer 2 CLIP 聚类（2-3 天，语义能力从 0 到有）
5. ★★★ Layer 4 InsightFace（2-3 天，漏检率 -25%）

### B. 推荐学习资源汇总

| 主题 | 资源 | 链接 |
|------|------|------|
| Python logging | 官方教程 | https://docs.python.org/3/howto/logging.html |
| PyTorch 设备管理 | 官方文档 | https://pytorch.org/docs/stable/ |
| pyiqa 模型库 | GitHub | https://github.com/chaofengc/IQA-PyTorch |
| CLIP | OpenAI GitHub | https://github.com/openai/CLIP |
| InsightFace | GitHub | https://github.com/deepinsight/insightface |
| Pydantic V2 | 官方文档 | https://docs.pydantic.dev/latest/ |
| pytest | 官方文档 | https://docs.pytest.org/ |
| ruff | 官方文档 | https://docs.astral.sh/ruff/ |
| Docker | 入门教程 | https://docs.docker.com/get-started/ |
| GitHub Actions | 官方文档 | https://docs.github.com/en/actions |
| asyncio 同步 | 官方文档 | https://docs.python.org/3/library/asyncio-sync.html |

### C. 各阶段完成后的自检清单

#### 第一阶段完成后
- [ ] `grep -rn "print(" core/ main.py` 结果接近零
- [ ] 模拟 Layer 1 异常，流水线不崩溃
- [ ] 在无 GPU 环境运行 `python -c "from core.layer3_aesthetic import score_photos"` 不报错
- [ ] 模拟 MediaPipe 初始化失败，不会永久标记为已初始化
- [ ] 同时发起两个分析请求，第二个被拒绝

#### 第二阶段完成后
- [ ] `pyiqa.create_metric('topiq_iaa')` 能正常加载并评分
- [ ] Layer 1 结果包含 `topiq_score` 字段
- [ ] Qwen3-VL API 调用正常，文案质量有提升
- [ ] CLIP 能将不同角度的同一场景聚类
- [ ] InsightFace 能检测人脸并保持 pipeline 接口兼容

#### 第三阶段完成后
- [ ] `pytest tests/ -v` 全部通过
- [ ] Layer 5 中途失败后重启，Layer 1-4 不重新执行
- [ ] 前端显示逐层进度
- [ ] `ruff check .` 无错误
- [ ] WebSocket 断线后能自动重连

---

## 任务依赖关系总览

```
1.1 logging ─────┬──→ 1.2 Pipeline 容错 ──→ 3.2 断点续传
                 ├──→ 1.3 GPU fallback ──→ 2.1 TOPIQ-IAA ──→ 2.2 TOPIQ-NR
                 └──→ 1.4 初始化重试
1.5 并发控制（独立）
2.3 Qwen3-VL（独立）
2.4 CLIP 聚类（独立，建议 1.3 后）
2.5 InsightFace（独立，建议 1.4 后）
3.1 Pydantic 模型（独立）
3.3 WebSocket 进度（独立）
3.4 测试（独立）
3.5 pyproject.toml（独立）
4.1 CI（依赖 3.4）
4.2 Docker（独立）
4.3 前端拆分（独立）
```

**建议执行顺序**：
1. 第一周：1.1 → 1.2 → 1.3 → 1.4 → 1.5
2. 第二周：2.1 → 2.3 → 2.2
3. 第三周：2.4 → 2.5（进阶可选）
4. 第四周起：3.1 → 3.4 → 3.2 → 3.3 → 3.5
5. 可选：4.1 → 4.2 → 4.3
