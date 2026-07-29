# 快速开始

本指南帮助你在 5 分钟内上手 PhotoRank。

## 前提条件

- Python 3.11+
- NVIDIA GPU（RTX 5070 或更高）
- CUDA 12.6+

## 步骤

### 1. 克隆项目

```bash
git clone https://github.com/Frankynwa/photo-rank.git
cd photo-rank
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt

# 安装 PyTorch（CUDA 13.2，支持 RTX 5070）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

### 4. 配置环境变量

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

### 5. 启动 Web UI

```bash
python main.py
```

### 6. 打开浏览器

访问 http://127.0.0.1:8000

## 使用方法

1. **上传照片**
   - 拖拽照片到上传区域
   - 或点击上传区域选择文件

2. **选择平台**
   - 小红书（3:4 竖版）
   - 朋友圈（1:1 正方形）
   - 抖音（9:16 竖版）

3. **开始分析**
   - 点击"开始分析"按钮
   - 等待分析完成（通常 2-5 分钟）

4. **查看结果**
   - 照片画廊显示评分
   - 点击查看详情（分类/构图/改进/情绪/文案）

## 下一步

- [安装指南](Installation.md) - 详细安装步骤
- [配置说明](Configuration.md) - 环境变量和配置文件
- [使用教程](Usage.md) - 功能详解和示例
- [架构设计](Architecture.md) - 系统架构和设计理念
