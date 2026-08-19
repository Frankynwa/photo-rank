# Web界面系统

<cite>
**本文引用的文件**   
- [main.py](file://main.py)
- [config.py](file://config.py)
- [templates/index.html](file://templates/index.html)
- [core/pipeline.py](file://core/pipeline.py)
- [core/layer1_objective.py](file://core/layer1_objective.py)
- [core/layer2_similarity.py](file://core/layer2_similarity.py)
- [core/layer3_aesthetic.py](file://core/layer3_aesthetic.py)
- [core/layer4_face.py](file://core/layer4_face.py)
- [core/video_frames.py](file://core/video_frames.py)
- [requirements.txt](file://requirements.txt)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构总览](#架构总览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考虑](#性能考虑)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介
本文件系统是一个基于FastAPI的Web应用，提供照片/视频上传、评分与可视化展示能力。前端模板index.html负责用户交互（上传、结果展示），后端通过 `/api/*` 接口接收文件并调用核心处理管线进行多维度分析与评分，通过 WebSocket（/ws）实时推送进度，最终返回结构化数据供前端渲染。

## 项目结构
- 入口与配置
  - main.py：FastAPI应用入口，定义路由、请求处理与 WebSocket 进度推送。
  - config.py：应用配置项（如上传路径、平台权重、阈值、服务端口等）。
- 前端模板
  - templates/index.html：主页面模板，包含上传区域与结果展示区。
- 核心处理管线
  - core/pipeline.py：编排层，串联各处理阶段，支持增量模式。
  - core/layer*.py：分阶段实现客观指标、相似聚类、美学评分、人脸质量。
  - core/video_frames.py：视频流式抽帧与本地快筛。
- 其他
  - requirements.txt：Python依赖声明。
  - uploads/：用户上传文件的临时存储目录。
  - output/：处理结果输出目录（可选）。

```mermaid
graph TB
subgraph "前端"
UI["index.html<br/>上传/展示/图表"]
end
subgraph "后端(FastAPI)"
APP["main.py<br/>路由与WebSocket"]
CFG["config.py<br/>配置"]
end
subgraph "核心处理"
PIPE["core/pipeline.py<br/>处理管线"]
L1["layer1_objective.py"]
L2["layer2_similarity.py"]
L3["layer3_aesthetic.py"]
L4["layer4_face.py"]
VF["video_frames.py"]
end
UI --> |HTTP/WebSocket| APP
APP --> CFG
APP --> PIPE
PIPE --> L1
PIPE --> L2
PIPE --> L3
PIPE --> L4
PIPE --> VF
```

**图示来源** 
- [main.py:1-200](file://main.py#L1-L200)
- [config.py:1-120](file://config.py#L1-L120)
- [templates/index.html:1-600](file://templates/index.html#L1-L600)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)
- [core/layer1_objective.py:1-150](file://core/layer1_objective.py#L1-L150)
- [core/layer2_similarity.py:1-150](file://core/layer2_similarity.py#L1-L150)
- [core/layer3_aesthetic.py:1-150](file://core/layer3_aesthetic.py#L1-L150)
- [core/layer4_face.py:1-150](file://core/layer4_face.py#L1-L150)

**章节来源**
- [main.py:1-200](file://main.py#L1-L200)
- [config.py:1-120](file://config.py#L1-L120)
- [templates/index.html:1-600](file://templates/index.html#L1-L600)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

## 核心组件
- FastAPI应用与路由（main.py）
  - 定义上传/查询/删除/分析接口，校验文件类型与大小，调用处理管线，返回统一JSON响应；通过 WebSocket 广播进度。
- 配置管理（config.py）
  - 集中管理上传目录、允许的文件类型、最大文件大小、平台权重（PLATFORMS）、各层阈值等。
- 处理管线（core/pipeline.py）
  - 按顺序执行多阶段分析：客观指标、相似聚类、美学评分（TOPIQ+VLM，含分类标签）、人脸质量，聚合结果并标准化输出；支持 checkpoint 与增量模式。
- 分阶段模块（core/layer*.py）
  - layer1_objective：客观指标检测（清晰度/曝光/噪声）。
  - layer2_similarity：pHash 相似聚类与去重。
  - layer3_aesthetic：TOPIQ-IAA + VLM 四路融合审美评分。
  - layer4_face：人脸质量评估与硬伤钳制。
  - video_frames：视频流式抽帧与本地快筛。
- 前端模板（templates/index.html）
  - 提供拖拽/选择上传（照片/视频）、WebSocket 实时进度、结果画廊与详情查看。

**章节来源**
- [main.py:1-200](file://main.py#L1-L200)
- [config.py:1-120](file://config.py#L1-L120)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)
- [core/layer1_objective.py:1-150](file://core/layer1_objective.py#L1-L150)
- [core/layer2_similarity.py:1-150](file://core/layer2_similarity.py#L1-L150)
- [core/layer3_aesthetic.py:1-150](file://core/layer3_aesthetic.py#L1-L150)
- [core/layer4_face.py:1-150](file://core/layer4_face.py#L1-L150)
- [templates/index.html:1-600](file://templates/index.html#L1-L600)

## 架构总览
系统采用“前端模板 + FastAPI后端 + 核心处理管线”的分层架构。前端通过Fetch发起上传请求，后端校验并持久化到uploads目录，随后调用pipeline进行多阶段分析，分析进度通过 WebSocket 实时推送，结果通过 `/api/results` 接口返回给前端渲染。

```mermaid
sequenceDiagram
participant U as "用户浏览器"
participant F as "FastAPI应用(main.py)"
participant P as "处理管线(core/pipeline.py)"
participant L1 as "客观指标(layer1)"
participant L2 as "相似聚类(layer2)"
participant L3 as "美学评分(layer3)"
participant L4 as "人脸质量(layer4)"
U->>F : "POST /api/upload (multipart/form-data)"
F->>F : "校验文件类型/大小/命名"
F->>P : "提交图片路径与参数"
P->>L1 : "执行客观指标检测"
L1-->>P : "返回清晰度/曝光/噪声"
P->>L2 : "执行相似聚类"
L2-->>P : "返回分组与代表"
P->>L3 : "执行美学评分（含分类标签）"
L3-->>P : "返回融合分与标签"
P->>L4 : "执行人脸质量评估"
L4-->>P : "返回硬伤钳制结果"
P-->>F : "聚合结果(评分/标签/排名)"
F-->>U : "WebSocket推送进度 + JSON结果"
```

**图示来源** 
- [main.py:1-200](file://main.py#L1-L200)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)
- [core/layer1_objective.py:1-150](file://core/layer1_objective.py#L1-L150)
- [core/layer2_similarity.py:1-150](file://core/layer2_similarity.py#L1-L150)
- [core/layer3_aesthetic.py:1-150](file://core/layer3_aesthetic.py#L1-L150)
- [core/layer4_face.py:1-150](file://core/layer4_face.py#L1-L150)

## 详细组件分析

### 前端模板 index.html
- 功能特性
  - 上传界面：支持点击选择与拖拽上传，显示文件名、大小与类型校验提示。
  - 进度与反馈：上传中显示进度条与状态消息；成功/失败时给出明确提示。
  - 结果展示：上传成功后渲染评分卡片、关键指标与说明文本。
  - 可视化图表：使用图表库渲染雷达图/柱状图，展示多维度评分。
- 交互流程
  - 用户选择/拖拽图片 -> 前端校验 -> 异步上传 -> 后端返回JSON -> 前端更新DOM与图表。
- 数据交换格式
  - 请求：multipart/form-data，字段名建议为image/file。
  - 响应：JSON，包含状态码、消息、评分对象、图表数据数组等。
- 实时反馈机制
  - 使用Fetch/AJAX发送请求，监听progress事件（若支持）或轮询状态；在UI上即时更新进度与结果。

```mermaid
flowchart TD
Start(["页面加载"]) --> Init["初始化上传控件与图表容器"]
Init --> UserAction{"用户触发上传?"}
UserAction --> |否| Idle["等待操作"]
UserAction --> |是| Validate["前端校验(类型/大小)"]
Validate --> Valid{"校验通过?"}
Valid --> |否| ShowError["显示错误提示"]
Valid --> |是| Upload["异步上传文件"]
Upload --> Progress["更新进度/状态"]
Progress --> Resp{"收到响应?"}
Resp --> |否| Retry["重试/提示网络错误"]
Resp --> |是| Parse["解析JSON结果"]
Parse --> Render["渲染评分卡片与图表"]
Render --> End(["完成"])
```

**图示来源** 
- [templates/index.html:1-600](file://templates/index.html#L1-L600)

**章节来源**
- [templates/index.html:1-600](file://templates/index.html#L1-L600)

### 后端API接口设计（main.py）
- 上传接口
  - 方法：POST
  - 路径：/api/upload（照片，支持 HEIC）、/api/upload_video（视频）
  - 内容类型：multipart/form-data
  - 响应：JSON，包含状态与消息
- 查询/删除接口
  - GET /api/photos：照片列表；GET/DELETE /api/photos/{filename}：单张查询/删除
  - GET /api/results/{platform}：榜单结果；/api/results/{platform}/videos/{filename}：视频榜单
- 分析接口
  - POST /api/analyze：触发流水线；进度通过 WebSocket /ws 推送（layer_start/layer_end/analysis_complete 等事件）
- 错误处理
  - 非法文件类型、过大文件、上传失败、处理异常均返回统一错误结构（HTTPException），便于前端提示。

```mermaid
classDiagram
class FastAPIApp {
+"/api/upload POST"
+"/api/upload_video POST"
+"/api/analyze POST"
+"/api/results GET"
+"/ws WebSocket"
+secure_filename()
}
class Config {
+UPLOAD_DIR
+ALLOWED_EXTENSIONS
+MAX_UPLOAD_SIZE
+PLATFORMS
}
class Pipeline {
+run_pipeline(photos, incremental)
+checkpoint()
}
FastAPIApp --> Config : "读取配置"
FastAPIApp --> Pipeline : "调用处理管线"
```

**图示来源** 
- [main.py:1-200](file://main.py#L1-L200)
- [config.py:1-120](file://config.py#L1-L120)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

**章节来源**
- [main.py:1-200](file://main.py#L1-L200)
- [config.py:1-120](file://config.py#L1-L120)

### 核心处理管线（core/pipeline.py）
- 职责
  - 编排layer1~layer4的执行顺序，传递中间结果，聚合最终评分与指标；支持 checkpoint 断点续跑与增量模式。
- 输入/输出
  - 输入：图片路径、平台权重、incremental/rerun_layers 等参数。
  - 输出：标准化JSON结构（ranking.json），包含融合分、分项明细与分类标签。
- 错误与回退
  - 单阶段失败不影响整体流程，记录错误并降级（如 VLM 不可用时保留 TOPIQ 路径），保证可用性。

```mermaid
flowchart TD
In["输入: 图片路径, 参数"] --> Stage1["阶段1: 客观指标"]
Stage1 --> Stage2["阶段2: 相似聚类"]
Stage2 --> Stage3["阶段3: 美学评分(TOPIQ+VLM)"]
Stage3 --> Stage4["阶段4: 人脸质量"]
Stage4 --> Out["输出: 融合分, 标签, ranking.json"]
```

**图示来源** 
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

**章节来源**
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

### 分阶段模块（core/layer*.py）
- layer1_objective：客观指标检测，返回清晰度/曝光/噪声结果。
- layer2_similarity：pHash 相似聚类，返回分组与每组 2 张代表。
- layer3_aesthetic：TOPIQ-IAA + VLM 四路融合，返回融合分与分类标签/建议。
- layer4_face：人脸质量评估，返回闭眼/清晰度检测与硬伤钳制。
- video_frames：视频流式抽帧（1s 间隔）+ L1/L2/L4 本地快筛。

```mermaid
classDiagram
class Layer1Objective {
+score_sharpness_exposure_noise(image)
}
class Layer2Similarity {
+cluster_by_phash(images)
}
class Layer3Aesthetic {
+topiq_score(image)
+vlm_score(image)
+fused_score()
}
class Layer4Face {
+analyze_faces(image)
+clamp_if_defect()
}
class VideoFrames {
+sample_frames(video)
+local_prefilter()
}
class Pipeline {
+run_pipeline(incremental)
}
Pipeline --> Layer1Objective
Pipeline --> Layer2Similarity
Pipeline --> Layer3Aesthetic
Pipeline --> Layer4Face
Pipeline --> VideoFrames
```

**图示来源** 
- [core/layer1_objective.py:1-150](file://core/layer1_objective.py#L1-L150)
- [core/layer2_similarity.py:1-150](file://core/layer2_similarity.py#L1-L150)
- [core/layer3_aesthetic.py:1-150](file://core/layer3_aesthetic.py#L1-L150)
- [core/layer4_face.py:1-150](file://core/layer4_face.py#L1-L150)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

**章节来源**
- [core/layer1_objective.py:1-150](file://core/layer1_objective.py#L1-L150)
- [core/layer2_similarity.py:1-150](file://core/layer2_similarity.py#L1-L150)
- [core/layer3_aesthetic.py:1-150](file://core/layer3_aesthetic.py#L1-L150)
- [core/layer4_face.py:1-150](file://core/layer4_face.py#L1-L150)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

## 依赖关系分析
- Python依赖
  - FastAPI/Uvicorn用于Web服务，httpx用于VLM API调用，图像处理库（OpenCV/Pillow/pillow-heif）、机器学习框架（PyTorch/pyiqa/MediaPipe）按模块懒加载。
- 模块耦合
  - main.py依赖config.py与core/pipeline.py；pipeline依赖各layer模块；前端依赖原生HTML/JS与WebSocket。
- 潜在循环依赖
  - 各layer模块应仅被pipeline调用，避免反向依赖导致循环。

```mermaid
graph LR
REQ["requirements.txt"] --> FAST["FastAPI/Uvicorn"]
REQ --> IMG["图像处理库"]
REQ --> ML["PyTorch/pyiqa/MediaPipe"]
MAIN["main.py"] --> CFG["config.py"]
MAIN --> PIPE["core/pipeline.py"]
PIPE --> L1["layer1"]
PIPE --> L2["layer2"]
PIPE --> L3["layer3"]
PIPE --> L4["layer4"]
PIPE --> VF["video_frames"]
```

**图示来源** 
- [requirements.txt:1-200](file://requirements.txt#L1-L200)
- [main.py:1-200](file://main.py#L1-L200)
- [config.py:1-120](file://config.py#L1-L120)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

**章节来源**
- [requirements.txt:1-200](file://requirements.txt#L1-L200)
- [main.py:1-200](file://main.py#L1-L200)
- [config.py:1-120](file://config.py#L1-L120)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

## 性能考虑
- 文件上传
  - 限制最大文件大小与类型，减少无效请求；使用流式写入避免内存峰值。
- 处理管线
  - 对耗时阶段启用缓存（相同哈希的图片复用结果）；并行执行独立阶段（如相似度与美学评分可并发）。
- 前端渲染
  - 图表按需渲染，避免重复绘制；懒加载大图预览。
- 资源管理
  - 定期清理uploads与output目录；合理设置超时与重试策略。

[本节为通用指导，不直接分析具体文件]

## 故障排查指南
- 常见错误
  - 文件类型不支持：检查config.allowed_types与前端校验逻辑。
  - 文件过大：调整max_size与前端提示。
  - 处理阶段失败：查看pipeline日志与各layer异常捕获。
  - 进度不更新：检查 WebSocket /ws 连接与事件订阅。
- 调试建议
  - 开启 uvicorn 详细日志；在前端控制台打印请求/响应。
  - 使用浏览器开发者工具检查网络请求与DOM更新。

**章节来源**
- [config.py:1-120](file://config.py#L1-L120)
- [main.py:1-200](file://main.py#L1-L200)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)
- [templates/index.html:1-600](file://templates/index.html#L1-L600)

## 结论
本Web界面系统以FastAPI为核心，结合模块化处理管线与友好的前端模板，实现了从照片/视频上传到多维评分与可视化的完整流程。通过清晰的API设计、WebSocket 实时进度与稳健的错误处理，系统具备良好的可扩展性与可维护性。

[本节为总结性内容，不直接分析具体文件]

## 附录

### 前后端数据交换格式
- 上传请求
  - 方法：POST
  - 路径：/api/upload（照片）、/api/upload_video（视频）
  - 内容类型：multipart/form-data
  - 字段：file（二进制文件）
- 分析触发与进度
  - POST /api/analyze 触发流水线；WebSocket /ws 推送 layer_start/layer_end/analysis_start/analysis_complete/analysis_error 事件
- 结果查询
  - GET /api/results/{platform} 返回榜单（融合分、分项明细、分类标签与建议）

**章节来源**
- [main.py:1-200](file://main.py#L1-L200)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

### 界面定制指南
- 样式修改
  - 在index.html中调整CSS类与布局，确保移动端适配（媒体查询、弹性布局）。
- 功能扩展
  - 新增阶段：在pipeline中添加新layer，并在API响应中补充对应字段。
  - 新增图表：在前端增加图表容器与渲染逻辑，保持数据格式一致。
- 多语言支持
  - 使用i18n键值替换静态文案；根据浏览器语言或用户设置动态切换。

**章节来源**
- [templates/index.html:1-600](file://templates/index.html#L1-L600)
- [core/pipeline.py:1-200](file://core/pipeline.py#L1-L200)

### 响应式设计与浏览器兼容性
- 响应式设计
  - 使用相对单位与媒体查询，确保在不同屏幕尺寸下布局正常。
  - 图表容器自适应宽度，避免溢出。
- 浏览器兼容
  - 主流现代浏览器（Chrome、Firefox、Safari、Edge）均支持Fetch与Canvas/SVG图表。
  - 如需兼容旧版IE，可使用polyfill或降级方案。

**章节来源**
- [templates/index.html:1-600](file://templates/index.html#L1-L600)