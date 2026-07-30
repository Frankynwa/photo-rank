# 📸 PhotoRank — 旅行照片 AI 智能筛选平台

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> 从大量旅行照片中，自动筛出适合发小红书/朋友圈/抖音的优质照片。

## ✨ 特性

- **5 层本地模型流水线** — 客观指标 → 相似聚类 → 审美评分 → 人脸质量 → 深度分析，逐层筛选可验证
- **RTX 5070 CUDA 加速** — 审美评分 GPU 推理
- **Web UI** — 拖拽上传、WebSocket 实时进度、详情查看
- **平台适配** — 小红书/朋友圈/抖音独立评分
- **深度分析** — Qwen-VL API 分类/构图/改进/情绪/文案
- **安全可靠** — 路径遍历防护、文件类型校验、上传大小限制
- **工程质量** — 44 个单元测试、Pydantic 数据模型、ruff 代码风格检查

## 🏗️ 架构

```
52 张照片
    ↓ Layer 1: 客观指标（OpenCV，CPU）
    ↓   淘汰模糊/过曝/欠曝
36 张保留
    ↓ Layer 2: 相似聚类（pHash，CPU）
    ↓   同类选最佳
26 组代表
    ↓ Layer 3: 审美评分（NIMA-VGG16，RTX 5070）
    ↓   按分数排序
Top 12
    ↓ Layer 4: 人脸质量（MediaPipe，CPU）
    ↓   闭眼/微笑检测
    ↓ Layer 5: 深度分析（Qwen-VL API）
    ↓   分类/构图/改进/情绪/文案
输出：精选照片 + 评分 + 文案
```

<details>
<summary>五层流水线详情</summary>

| 层级 | 功能 | 核心技术 | 说明 |
|------|------|----------|------|
| Layer 1 | 客观指标 | OpenCV（CPU） | 清晰度（拉普拉斯方差）、曝光（直方图分析）、噪声（信噪比） |
| Layer 2 | 相似聚类 | pHash（CPU） | 感知哈希 + 汉明距离聚类，每组选最清晰代表 |
| Layer 3 | 审美评分 | NIMA-VGG16（GPU） | AVA 数据集训练，SRCC 0.71，1-10 分评分 |
| Layer 4 | 人脸质量 | MediaPipe（CPU） | 478 关键点 + 52 blendshape，闭眼/微笑/人脸清晰度检测 |
| Layer 5 | 深度分析 | Qwen-VL Max API | 分类（8 类）、构图解读、改进建议、情绪图谱、平台定制文案 |

</details>

## 🚀 快速开始

### 环境要求

- Python 3.11+
- NVIDIA GPU（RTX 5070 或更高）
- CUDA 12.6+

### 安装

```bash
# 克隆仓库
git clone https://github.com/Frankynwa/photo-rank.git
cd photo-rank

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 PyTorch（CUDA 13.2，支持 RTX 5070）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

### 配置

创建 `.env` 文件：

```env
# Qwen-VL API（用于深度分析）
QWEN_API_KEY=your_api_key_here
QWEN_BASE_URL=https://llm-1r8e612iutxlixav.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-vl-max

# 代理（可选）
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
```

### 运行

```bash
# 启动 Web UI
python main.py

# 打开浏览器访问
# http://127.0.0.1:8000
```

## 📁 项目结构

```
photo-rank/
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── layer1_objective.py      # Layer 1: 客观指标（清晰度/曝光/噪声）
│   ├── layer2_similarity.py     # Layer 2: 相似聚类（pHash）
│   ├── layer3_aesthetic.py      # Layer 3: 审美评分（NIMA-VGG16）
│   ├── layer4_face.py           # Layer 4: 人脸质量（MediaPipe）
│   ├── layer5_analysis.py       # Layer 5: 深度分析（Qwen-VL）
│   ├── models.py                # Pydantic 数据模型
│   ├── logger.py                # 日志模块
│   └── pipeline.py              # 流水线整合
├── main.py                      # Web UI 启动
├── config.py                    # 配置文件
├── templates/
│   └── index.html               # 前端 UI
├── tests/                       # 单元测试（44 个）
│   ├── test_config.py
│   ├── test_layer1.py
│   └── test_layer2.py
├── docs/
│   └── research/                # 研究报告
│       ├── photo_aesthetic_models_research.md
│       ├── photo-aesthetic-research-report.md
│       ├── qwen-vl-photo-analysis-research.md
│       └── web_ui_framework_research.md
├── uploads/                     # 上传的照片
├── output/                      # 分析结果
├── pyproject.toml               # 项目配置（ruff 等）
├── requirements.txt             # 依赖
├── .gitignore
└── README.md
```

## 🔬 技术细节

### Layer 1: 客观指标

- **清晰度**: 拉普拉斯方差（Laplacian Variance）
- **曝光**: 直方图分析（过曝/欠曝检测）
- **噪声**: 信噪比（SNR）
- **阈值**: 清晰度 > 100，曝光 < 30% 过曝

### Layer 2: 相似聚类

- **算法**: 感知哈希（pHash）
- **阈值**: 汉明距离 ≤ 10
- **选择**: 每组选清晰度最高的代表

### Layer 3: 审美评分

- **模型**: NIMA-VGG16（AVA 数据集训练）
- **SRCC**: 0.71（Spearman 秩相关系数）
- **分数范围**: 1-10（5 分是"普通"，6 分以上是"好看"）
- **推理**: RTX 5070 CUDA，0.09GB 显存

### Layer 4: 人脸质量

- **模型**: MediaPipe Face Landmarker v1.0
- **检测**: 478 个关键点 + 52 个 blendshape
- **功能**: 闭眼检测、微笑检测、人脸清晰度

### Layer 5: 深度分析

- **模型**: Qwen-VL Max API
- **功能**:
  - 分类（8 类：风景/人像/人文/美食/夜景/自拍/合照/其他）
  - 构图解读（三分法/引导线/对称等）
  - 改进建议（裁剪/调色/角度）
  - 情绪图谱（快乐/宁静/震撼等）
  - 文案生成（平台定制）

## 📊 研究报告

详见 `docs/research/` 目录：

1. **审美模型研究** — Q-ReAlign/NIMA/TOPIQ 对比
2. **相似度聚类研究** — DINOv2/CLIP/pHash 对比
3. **Qwen-VL API 研究** — prompt 模板/最佳实践
4. **Web UI 框架研究** — FastAPI/React/Vue/Streamlit 对比

## 🛠️ 开发

### 运行测试

```bash
# 运行全部测试（44 个）
python -m pytest tests/ -v

# 代码风格检查
ruff check .

# 自动修复代码风格
ruff check . --fix
```

### 添加新平台

编辑 `config.py`，在 `PLATFORMS` 字典中添加：

```python
"new_platform": {
    "name": "新平台",
    "ratio": "3:4",
    "style": "风格描述",
}
```

### 自定义模型

替换 `core/layer3_aesthetic.py` 中的模型：

```python
# 使用其他 PyIQA 模型
import pyiqa
model = pyiqa.create_metric('your_model_name', device='cuda')
```

## 📝 更新日志

### v1.1.0 (2026-07-30)

- ✅ 安全加固：路径遍历防护、文件类型校验、上传大小限制
- ✅ Pydantic 数据模型集成（`core/models.py`）
- ✅ 44 个单元测试覆盖
- ✅ ruff 代码风格检查
- ✅ WebSocket 实时进度推送

### v1.0.0 (2026-07-29)

- ✅ 5 层流水线完整实现
- ✅ RTX 5070 CUDA 支持
- ✅ Web UI（拖拽上传、实时进度）
- ✅ 平台适配（小红书/朋友圈/抖音）
- ✅ 深度分析（Qwen-VL API）

## 🙏 致谢

- [PyIQA](https://github.com/chaofengc/IQA-PyTorch) — 图像质量评估库
- [MediaPipe](https://github.com/google/mediapipe) — 人脸检测
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 视觉语言模型
- [NIMA](https://arxiv.org/abs/1709.05424) — 神经网络图像美学评估

## 📄 许可证

MIT License

## 📧 联系

- GitHub: [Frankynwa](https://github.com/Frankynwa)
- 项目: [photo-rank](https://github.com/Frankynwa/photo-rank)
