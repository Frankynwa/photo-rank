---
kind: dependency_management
name: Python 依赖管理（requirements.txt + 运行时按需导入）
category: dependency_management
scope:
    - '**'
source_files:
    - requirements.txt
    - main.py
    - core/pipeline.py
    - core/config.py
---

本项目采用最简化的 Python 依赖管理模式，通过根目录的 `requirements.txt` 声明运行时依赖，无虚拟环境、无锁文件、无私有仓库配置。

**使用的工具与方式**
- 依赖声明：仅使用 `requirements.txt`，未使用 `pyproject.toml`、`setup.py`、`Pipfile` 或 `poetry.lock` 等现代依赖管理工具。
- 版本约束：全部使用 `>=` 宽松下限约束（如 `fastapi>=0.100.0`、`uvicorn[standard]>=0.30.0`），未锁定具体版本号，也未使用 `~=` 或精确版本。
- 可选依赖：仅 `uvicorn[standard]` 使用了 extras 语法。
- 无 vendoring：未发现 `vendor/` 目录或任何第三方库源码内嵌。
- 无私有注册表：未发现 `.pypirc`、`pip.conf` 或 `--index-url` 相关配置。

**核心依赖清单**
- Web 框架层：`fastapi`、`uvicorn[standard]`、`httpx`、`python-multipart`
- 图像处理层：`Pillow`、`pillow-heif`（HEIC 支持）、`opencv-python`（cv2，在 layer1_objective.py 中 import）、`imagehash`（pHash 相似性，layer2_similarity.py 中使用）
- AI/ML 层：`torch`（NIMA-VGG16 审美评分，layer3_aesthetic.py 中条件导入）、`pyiqa`（图像质量评估，layer3_aesthetic.py 中条件导入）、`mediapipe`（人脸检测，layer4_face.py 中条件导入）
- 配置与工具：`PyYAML`（config.py 中未直接使用但已声明）

**关键设计模式：运行时按需导入（Lazy Import）**
项目大量使用函数内部 `import` 语句实现依赖的延迟加载，这是本项目的核心依赖管理策略：
- `core/pipeline.py` 中各 layer 模块在 `run_pipeline()` 函数体内才导入
- `core/layer3_aesthetic.py` 中 `torch`、`pyiqa` 在函数内条件导入
- `core/layer4_face.py` 中 `mediapipe` 及其子模块在函数内导入
- `main.py` 中 `core.pipeline.run_pipeline` 也在 `/api/analyze` 路由内动态导入

这种模式使得重型依赖（如 torch、mediapipe）仅在真正执行对应功能时才被加载，降低启动开销和内存占用。

**约束与约定**
- 所有第三方依赖必须通过 `pip install -r requirements.txt` 安装
- 无依赖版本锁定机制，不同环境可能安装不同版本
- 无依赖冲突检测或解决策略
- 无 CI/CD 中的依赖更新自动化流程