---
kind: logging_system
name: 日志系统 — 基于 print 的简单输出，无结构化日志框架
category: logging_system
scope:
    - '**'
source_files:
    - main.py
    - config.py
    - core/pipeline.py
    - core/layer3_aesthetic.py
    - core/layer4_face.py
    - core/layer5_analysis.py
---

本仓库未实现专门的日志系统。代码中所有输出均通过 Python 内置的 `print()` 函数直接打印到标准输出，未使用任何日志框架（如 logging、loguru、structlog 等），也没有统一的日志级别管理、格式化器或输出目标配置。

具体表现：
- `core/pipeline.py` 中使用大量 `print()` 打印流水线进度、筛选统计和分隔线；
- `core/layer3_aesthetic.py`、`core/layer4_face.py`、`core/layer5_analysis.py` 等模块在模型加载、初始化、分析进度及异常时均通过 `print()` 输出信息；
- `main.py` 中的 FastAPI Web UI 未集成任何日志中间件，错误通过 HTTPException 返回，WebSocket 广播用于前端状态推送而非日志记录。

约束与约定：
- 所有调试与运行信息均以人类可读的中文文本形式直接输出到控制台；
- 没有日志文件写入、没有结构化字段（如 timestamp、level、module）、没有日志轮转或远程收集机制。

由于缺乏统一的日志抽象层，若需引入结构化日志，需要在各模块中统一替换 `print()` 调用并配置统一的 logger。