# 更新日志

本文件记录 PhotoRank 项目的所有重要更改。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [1.0.0] - 2026-07-29

### 新增
- 5 层本地模型流水线
  - Layer 1: 客观指标（OpenCV，清晰度/曝光/噪声）
  - Layer 2: 相似聚类（pHash）
  - Layer 3: 审美评分（NIMA-VGG16，RTX 5070 CUDA）
  - Layer 4: 人脸质量（MediaPipe，闭眼/微笑检测）
  - Layer 5: 深度分析（Qwen-VL API，分类/构图/改进/文案）
- Web UI
  - 拖拽上传照片
  - 平台选择（小红书/朋友圈/抖音）
  - 实时进度（WebSocket）
  - 照片画廊 + 评分显示
  - 详情弹窗（分类/构图/改进/情绪/文案）
- 平台适配
  - 小红书：3:4 竖版，氛围感
  - 朋友圈：1:1 正方形，叙事性
  - 抖音：9:16 竖版，视觉冲击
- 深度分析
  - 照片分类（8 类：风景/人像/人文/美食/夜景/自拍/合照/其他）
  - 构图解读（三分法/引导线/对称等）
  - 改进建议（裁剪/调色/角度）
  - 情绪图谱（快乐/宁静/震撼等）
  - 文案生成（平台定制，15-30 字）

### 技术
- RTX 5070 CUDA 支持（PyTorch 2.13.0+cu132）
- MediaPipe Face Landmarker v1.0
- Qwen-VL Max API 集成
- FastAPI + 原生 HTML/JS Web UI

### 文档
- 详细 README.md
- 4 份研究报告
  - 审美模型研究
  - 相似度聚类研究
  - Qwen-VL API 研究
  - Web UI 框架研究

## [未发布]

### 计划
- [ ] 支持更多平台（微博/Instagram）
- [ ] 批量导出（ZIP 包）
- [ ] 照片编辑建议（自动裁剪/调色）
- [ ] 多语言支持（英文/日文）
- [ ] 移动端适配
- [ ] 离线模式（无需 API）
