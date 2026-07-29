# 贡献指南

感谢你对 PhotoRank 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

1. 在 [Issues](https://github.com/Frankynwa/photo-rank/issues) 页面创建新 issue
2. 使用清晰的标题描述问题
3. 提供详细的问题描述：
   - 操作系统和 Python 版本
   - 错误信息和堆栈跟踪
   - 重现步骤
   - 预期行为 vs 实际行为

### 提交代码

1. Fork 项目
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 创建 Pull Request

### 代码规范

- 使用 Python 3.11+ 语法
- 遵循 PEP 8 代码风格
- 添加类型注解
- 编写文档字符串
- 添加单元测试

### 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

类型：
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```
feat(layer3): add Q-ReAlign-Mini model support

- Add Q-ReAlign-Mini-0.8B model for aesthetic scoring
- Support CUDA inference on RTX 5070
- Update model loading logic

Closes #123
```

## 开发环境

### 设置

```bash
# 克隆项目
git clone https://github.com/Frankynwa/photo-rank.git
cd photo-rank

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install pytest black flake8 mypy
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_layer1.py

# 运行带覆盖率的测试
python -m pytest tests/ --cov=core --cov-report=html
```

### 代码检查

```bash
# 格式化代码
black core/ main.py

# 检查代码风格
flake8 core/ main.py

# 类型检查
mypy core/ main.py
```

## 项目结构

```
photo-rank/
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── layer1_objective.py      # Layer 1: 客观指标
│   ├── layer2_similarity.py     # Layer 2: 相似聚类
│   ├── layer3_aesthetic.py      # Layer 3: 审美评分
│   ├── layer4_face.py           # Layer 4: 人脸质量
│   ├── layer5_analysis.py       # Layer 5: 深度分析
│   └── pipeline.py              # 流水线整合
├── main.py                      # Web UI 启动
├── config.py                    # 配置文件
├── templates/                   # 前端模板
├── tests/                       # 测试文件
├── docs/                        # 文档
│   └── research/                # 研究报告
├── uploads/                     # 上传的照片
├── output/                      # 分析结果
├── requirements.txt             # 依赖
├── CHANGELOG.md                 # 更新日志
├── CONTRIBUTING.md              # 贡献指南
└── README.md                    # 项目说明
```

## 添加新功能

### 添加新模型

1. 在 `core/` 目录创建新文件，例如 `layer6_newmodel.py`
2. 实现评分函数：
   ```python
   def score_photos(image_paths: list[str | Path]) -> list[dict]:
       # 实现评分逻辑
       pass
   ```
3. 在 `core/pipeline.py` 中集成
4. 添加测试
5. 更新文档

### 添加新平台

1. 编辑 `config.py`，在 `PLATFORMS` 字典中添加：
   ```python
   "new_platform": {
       "name": "新平台",
       "ratio": "3:4",
       "style": "风格描述",
   }
   ```
2. 更新 `core/layer5_analysis.py` 中的文案生成 prompt
3. 更新 Web UI 中的平台选择器
4. 添加测试

## 发布流程

1. 更新版本号（`config.py` 和 `README.md`）
2. 更新 `CHANGELOG.md`
3. 创建 Git tag：`git tag -a v1.0.0 -m "Release v1.0.0"`
4. 推送 tag：`git push origin v1.0.0`
5. 在 GitHub 创建 Release

## 行为准则

- 尊重所有参与者
- 接受建设性批评
- 专注于对社区最有利的事情
- 对他人表示同理心

## 许可证

本项目采用 MIT 许可证。贡献代码即表示你同意你的贡献将在 MIT 许可证下发布。

## 联系方式

如有问题，请通过以下方式联系：

- GitHub Issues: [photo-rank/issues](https://github.com/Frankynwa/photo-rank/issues)
- GitHub Discussions: [photo-rank/discussions](https://github.com/Frankynwa/photo-rank/discussions)

感谢你的贡献！🎉
