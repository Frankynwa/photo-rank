---
kind: configuration_system
name: PhotoRank 配置系统：硬编码常量 + .env 密钥管理
category: configuration_system
scope:
    - '**'
source_files:
    - config.py
    - core/pipeline.py
    - core/layer5_analysis.py
    - main.py
    - .gitignore
---

## 1. 使用的系统与方式
- 采用 Python 模块级常量作为主配置源（`config.py`），通过 `from config import ...` 直接导入使用。
- 敏感凭据（Qwen-VL API Key）通过本地 `.env` 文件读取，由 `core/layer5_analysis.py` 中的 `get_api_key()` 函数解析。
- Web 服务启动参数（host/port）同时出现在 `config.py` 和 `main.py` 中，存在重复定义。

## 2. 关键文件与包
- `config.py`：集中存放项目路径、照片分类、平台策略、各层阈值、API 地址与代理、Web 服务监听参数等全部运行时配置。
- `core/pipeline.py`：流水线入口，从 `config` 导入 `PHOTOS_DIR, OUTPUT_DIR, CATEGORIES, PLATFORMS, TOP_K_PER_CATEGORY, FINAL_OUTPUT_COUNT` 驱动五层筛选。
- `core/layer5_analysis.py`：唯一从 `.env` 读取 QWEN_API_KEY 的模块，并依赖 `config` 中的 `QWEN_BASE_URL, QWEN_MODEL, PROXY` 调用 Qwen-VL API。
- `main.py`：FastAPI 应用入口，独立定义 `BASE_DIR / uploads, output, templates` 目录，未复用 `config.PHOTOS_DIR/OUTPUT_DIR`。
- `.gitignore`：显式忽略 `.env` 与 `*.env.local`，防止密钥泄露。

## 3. 架构与约定
- **单一配置文件模式**：所有可调参数集中在 `config.py`，以全局变量形式暴露，被 `core/*` 各层直接 import 使用，无配置对象或工厂封装。
- **分层阈值分离**：Layer 1（清晰度/曝光）、Layer 2（相似度）、Layer 3（审美 Top K/Final Count）、Layer 5（Qwen API）各自的阈值均作为独立常量声明在 `config.py`，便于调参。
- **平台策略字典化**：`PLATFORMS` 以键值对形式定义不同社交平台（小红书/朋友圈/抖音）的名称、风格描述与输出数量，pipeline 按 platform 字符串动态选择。
- **路径基于相对位置**：`PROJECT_ROOT = Path(__file__).parent`，所有目录（uploads/output/cache）均相对于 `config.py` 所在根目录构造，保证跨环境一致性。
- **密钥与环境隔离**：`.env` 仅用于存储 `QWEN_API_KEY`，且路径硬编码为 Windows 用户目录；其他非敏感配置不通过环境变量注入。

## 4. 约定与约束
- **配置来源优先级**：当前实现中不存在多环境覆盖机制（如 dev/prod），所有配置均以 `config.py` 为准，`.env` 仅补充 API Key。
- **路径自动创建**：`main.py` 启动时确保 `uploads/output/templates` 目录存在（`mkdir(exist_ok=True)`），但 `cache` 目录未在代码中自动创建。
- **平台枚举约束**：`run_pipeline(platform=...)` 期望传入 `xiaohongshu/wechat/douyin` 之一，若传入未知平台会在 `PLATFORMS[platform]` 处抛出 KeyError。
- **API Key 读取约束**：`get_api_key()` 固定读取 `C:\Users\Administrator\AppData\Local\hermes\.env`，若文件中不存在 `QWEN_API_KEY=` 行则抛出 `ValueError`。
- **Git 安全约束**：`.gitignore` 明确排除 `.env` 和 `*.env.local`，禁止将任何包含密钥的环境文件提交到仓库。
- **配置与代码耦合**：`main.py` 中再次硬编码了 `HOST=127.0.0.1, PORT=8000`，与 `config.py` 中的 `HOST=127.0.0.1, PORT=8080` 不一致，存在潜在冲突。

## 5. 改进建议（非强制）
- 统一 `main.py` 与 `config.py` 中的 Web 服务端口配置，避免歧义。
- 将 `.env` 路径改为可配置或使用标准库 `python-dotenv` 加载，提升跨平台兼容性。
- 考虑引入 Pydantic Settings 或类似框架，实现类型校验、默认值与环境变量覆盖的统一管理。