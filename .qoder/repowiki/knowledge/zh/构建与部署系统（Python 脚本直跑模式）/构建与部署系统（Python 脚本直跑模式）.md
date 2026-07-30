---
kind: build_system
name: 构建与部署系统（Python 脚本直跑模式）
category: build_system
scope:
    - '**'
source_files:
    - requirements.txt
    - main.py
    - config.py
---

该仓库未采用传统意义上的构建系统，而是以纯 Python 脚本 + 依赖声明文件的方式组织，通过直接运行入口脚本启动服务。具体特点如下：

1. **依赖管理**：使用 `requirements.txt` 声明运行时依赖（FastAPI、Uvicorn、Pillow、PyYAML 等），无 `setup.py`、`pyproject.toml`、`Pipfile` 或虚拟环境管理脚本。
2. **启动方式**：以 `main.py` 为统一入口，通过 `uvicorn.run(app, host="127.0.0.1", port=8000)` 直接启动 FastAPI 服务；`config.py` 提供项目路径、平台配置、各层阈值等运行时参数。
3. **无构建/打包步骤**：仓库中不存在 Makefile、Dockerfile、docker-compose.yml、CI 配置文件（`.github/workflows`）、发布脚本或版本管理脚本。`docs/research/web_ui_framework_research.md` 中出现的 `docker-compose.yml` 仅为研究文档中的示例结构，并非实际实现。
4. **目录约定**：`uploads/` 存放上传照片，`output/` 存放分析结果，`templates/index.html` 为前端模板，均为运行时动态创建或读取，无需预构建。
5. **部署形态**：当前为单进程 Python 应用，通过 `python main.py` 即可本地运行，适合开发调试与轻量部署，尚未容器化或集成 CI/CD。

综上，该项目处于“脚本即部署”的极简形态，缺乏标准化的构建、测试、打包与发布流程。