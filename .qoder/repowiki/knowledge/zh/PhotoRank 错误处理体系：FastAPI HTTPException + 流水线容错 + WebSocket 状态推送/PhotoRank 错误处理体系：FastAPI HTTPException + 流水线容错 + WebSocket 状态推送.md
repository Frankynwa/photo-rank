---
kind: error_handling
name: PhotoRank 错误处理体系：FastAPI HTTPException + 流水线容错 + WebSocket 状态推送
category: error_handling
scope:
    - '**'
source_files:
    - main.py
    - core/pipeline.py
    - core/layer5_analysis.py
    - core/layer1_objective.py
    - core/layer2_similarity.py
    - core/layer3_aesthetic.py
    - core/layer4_face.py
---

## 1. 使用的系统与模式

- **Web 层**：基于 FastAPI 的 `HTTPException` 作为统一的 HTTP 错误抛出机制，配合 `WebSocketDisconnect` 处理 WebSocket 连接异常。
- **核心流水线**：采用「逐层 try/except Exception」的容错策略，单层失败不中断整体流程，通过返回结构化结果（含 `error` 字段）向上传递。
- **实时状态**：通过自定义 `ConnectionManager` 的 `broadcast` 方法，将分析进度、完成状态和错误信息以 JSON 消息推送到前端。
- **外部依赖错误**：使用 `raise_for_status()` 处理 HTTP 请求异常，对缺失配置（如 `.env` 中的 API Key）直接 `raise ValueError`。

## 2. 关键文件与位置

- `main.py`：FastAPI 应用入口，集中处理 HTTP 异常、WebSocket 断连、分析任务异常广播
- `core/pipeline.py`：五层流水线的统一编排，每层失败时捕获并记录错误
- `core/layer5_analysis.py`：Qwen-VL API 调用的错误处理（网络异常、JSON 解析失败、API Key 缺失）
- 各 layer 模块（layer1~4）：独立的 try/except 包裹，失败时返回带 error 字段的字典

## 3. 架构与约定

### 分层错误传播策略
- **Layer 1-4**：每个分析函数内部用 `try/except Exception as e` 包裹，失败时返回包含 `"error": str(e)` 的结构化结果，不向上抛出异常
- **Layer 5**：异步深度分析在 pipeline 中单独 try/except，失败时填充默认值（如 `"classification": "未知"`），保证流水线继续执行
- **Pipeline 层**：收集各层结果，即使某层失败也继续后续处理，最终输出包含统计信息的汇总结果

### Web 层错误处理
- **资源不存在**：使用 `HTTPException(status_code=404, detail="...")` 明确区分不同场景（照片不存在、结果未找到）
- **分析异常**：`/api/analyze` 路由捕获所有异常，先通过 WebSocket 广播 `analysis_error` 消息，再抛出 500 错误
- **WebSocket**：`WebSocketDisconnect` 被显式捕获，确保连接清理逻辑正常执行

### 外部服务错误处理
- **HTTP 请求**：`httpx.AsyncClient` 调用后使用 `response.raise_for_status()` 抛出 HTTP 异常
- **配置文件读取**：`.env` 文件缺失或格式错误时直接 `raise ValueError("QWEN_API_KEY not found in .env")`
- **JSON 解析**：Qwen-VL 返回的 JSON 解析失败时，返回预定义的默认结构而非崩溃

## 4. 约定与约束

- **禁止裸 except**：除 WebSocket broadcast 和 JSON 解析等少数场景外，所有异常捕获都指定具体类型或使用 `Exception as e` 保留错误信息
- **错误信息结构化**：所有错误都以包含 `error` 键的字典形式返回，便于前端统一处理
- **非致命错误降级**：Layer 5 深度分析失败不影响其他层结果，提供默认值保证系统可用性
- **用户可见错误**：所有用户可感知的错误都通过 HTTP 状态码或 WebSocket 消息明确传达，避免静默失败
- **日志与调试**：关键错误路径都有 print 语句输出，便于本地调试和问题定位