# P0-1 第二路审美分（Q-ReAlign）落地决策 — 2026-11

> 状态：调研+本地核查完成，**待拍板**。未改任何代码。
> 本地实测环境：photo-rank venv（pyiqa 0.1.16 / transformers 5.14.1 / torch 2.13.0+cu132 / RTX 5070 12GB）。

## 结论先行

- **选型：Q-ReAlign Mini（0.8B）**，pyiqa 指标名 `qrealign`（默认=mini），`task='aesthetic'`。"两行代码接入"属实（本地已验证指标名与依赖）。
- **融合：与 TOPIQ-IAA 做秩融合（平均分位 / Borda）**，不做加权平均（两路尺度不一致且现有管线有批次归一化）。
- **成本警示：Q-ReAlign 是 Qwen3.5-VL 系 VLM**（Mini 2.21GB），不是轻量 CNN。API 是 2 行，推理成本是 VLM 量级——Mini 在 4090 上 26.7 img/s@bs4，几千张分钟级，12GB 显存随便跑；但与"TOPIQ 纯 CNN"不是一类东西。

## 1. 本地事实核查（2026-11-03 直接查 venv，非文档转述）

- pyiqa 0.1.16 已内置 QReAlign 架构（`pyiqa/archs/qrealign_arch.py`），0.1.15 起随包提供。
- **指标名**：`qrealign`（= `qrealign-mini`）、`qrealign-lite`、`qrealign-pro`。`metric_mode=NR`，`score_range 0,1`。
- **权重**：`q-future/Q-ReAlign-Mini-0.8B`（2.21GB bf16）/ `Lite-4B`（10.35GB）/ `Pro-9B`（18.84GB）。
- **底座**：Qwen3.5-VL（`AutoModelForImageTextToText`），得分 = 5 个等级词（excellent/good/fair/poor/bad）logits 按 `[1.0,0.75,0.5,0.25,0.0]` 加权折算标量；`task` 可选 `quality`/`aesthetic`。
- **依赖**：transformers ≥ 5.0（≥5.2 原生 qwen3_5；本机 5.14.1 ✅ 免 shim）。无需另装 CLIP/LLM。
- 显存：Mini <4GB；Lite 10.35GB 仅 bs1 勉强进 12GB；Pro 装不下需 4bit。

## 2. 候选模型对比（环境：Win11 + RTX 5070 12GB，批量几百～几千张）

| 维度 | TOPIQ-IAA（现有） | **Q-ReAlign Mini/Lite/Pro** | ArtiMuse (CVPR26) | DeQA-Score | NIMA |
|---|---|---|---|---|---|
| AVA SRCC | 0.791 | **0.797 / 0.814 / 0.832** | 0.827 | 无（纯 IQA，任务不对齐） | 0.713 |
| 权重体积 | 数百 MB | 2.21 / 10.35 / 18.84 GB | ~16GB (InternVL3-8B) | ~14–16GB (mPLUG-Owl2-7B) | 几十 MB |
| 12GB 显存 | ~2–3GB | Mini 轻松 / Lite 勉强 / Pro 需 4bit | 装不下 | 装不下 | 低 |
| 速度 | 快（CNN） | Mini 26.7 img/s@bs4 | 慢（8B+8 维文本生成） | 慢（7B） | 快 |
| 集成 | pyiqa `topiq_iaa` | **pyiqa `qrealign` 两行** | 自建（custom_code+flash-attn） | 自建（锁老 transformers） | pyiqa `nima` |
| 许可证 | PolyForm-NC + CC BY-NC-SA（**非商用**） | **模型 Apache-2.0 / 代码 MIT** | 模型 Apache-2.0（数据集非商用） | MIT | CC BY-NC-SA |
| 分数尺度 | 批次归一化 0-10 | [0,1]，`aesthetic` 任务对齐 IAA | 0–100（Token As Score） | [1,5]+高斯分布 | [0,1] |

关键判断：
- **DeQA-Score 出局**：它是 IQA（KonIQ/SPAQ 失真质量），不是审美分，任务不对齐。"分数分布加置信度"的想法可留到 P2 再评估别的载体。
- **ArtiMuse 精度最高（0.827）但成本/集成双高**，且 8 维属性是专家文本非数值分。留作 P2 增强（若要 8 维文本解读再上）。
- **Q-ReAlign 是唯一"现代 MLLM 审美分 + 原生 pyiqa + 商用干净"的组合**。且许可证比现有 TOPIQ（非商用）更干净——若未来商用，Q-ReAlign 恰是合规资产。

## 3. 融合方案对比（两路客观分 → 现有 0-10 批次归一化管线）

| 方案 | 批次大小变化稳定性 | 尺度不一致处理 | 缺点 | 结论 |
|---|---|---|---|---|
| **秩融合（平均分位）** | **稳**（秩对单调变换不变） | 天然免疫 | 丢弃分差信息 | ✅ 推荐 |
| z-score 平均 | 差（均值/方差随批次漂移，叠加现有批次归一化=双重批次依赖） | 需归一化 | 跨批次不可比 | ❌ |
| min-max 加权平均 | 差（离群点敏感） | 需归一化 | 小批次不稳 | ❌ |

**推荐公式（融合只发生在两路客观分之间）**：
- 第 m 路（TOPIQ / Q-ReAlign）第 i 张的升序秩 r_{m,i} ∈ [1,N]；分位 p_{m,i} = (N − r_{m,i} + 1) / N。
- 客观审美通道分：p_obj = 0.5·p_topiq + 0.5·p_qrealign（或按各自 SRCC 微调权重）。
- 回映 0-10：对 p_obj 做一次批次内 min-max ×10（仅用于展示/下游一致性）。

**集成位置（两个选项，待拍板）**：
- **选项 A（推荐，改动小）**：把现有 FUSION_WEIGHTS 的 `topiq` 通道（0.30）的输入从"TOPIQ 单路"换成"TOPIQ⊕Q-ReAlign 秩融合"，vlm/face/objective 通道不动。
- **选项 B（重构大）**：最终分全面改秩融合（vlm/topiq/qrealign/face/objective 五路 Borda）。更彻底但动 checkpoint/重归一化/测试面大，不建议 MVP 做。

## 4. 风险

1. **pyiqa/TOPIQ 许可证是非商用**（PolyForm-NC / CC BY-NC-SA；PyPI 的 Apache 标签是错标，上游 issue #294）。商用需另谈或全链换件——Q-ReAlign 恰好是第一步合规替代。
2. Q-ReAlign **无同行评审论文**（Q-Future 软件发布，AVA 数字自报）。投产前建议在自有标注批次上做一次小样本 SRCC 对照（TOPIQ vs Q-ReAlign vs 人工序）。
3. transformers 5.x 与 pyiqa 其他指标的潜在版本摩擦——本机 5.14.1 已装且 TOPIQ（纯 CNN）不依赖 transformers，当前无冲突；若未来加 4bit 量化链再评估独立环境。
4. 显存分时：Mini <4GB 可与 TOPIQ 常驻共存，暂不需要动 `release_topiq_model` 分时逻辑。

## 5. 待拍板

1. 选 Mini（推荐）还是 Lite/Pro？
2. 集成走选项 A（推荐）还是 B？
3. 融合权重 0.5/0.5 起步，还是按 SRCC（0.791:0.797）微调？
4. 是否同步补一个"小样本 SRCC 校验"验收脚本（建议要）？

## 来源

- https://github.com/Q-Future/Q-ReAlign （AVA SRCC：Mini 0.797 / Lite 0.814 / Pro 0.832）
- https://iqa-pytorch.readthedocs.io/en/latest/autoapi/pyiqa/archs/qrealign_arch/index.html （指标名/尺度/等级词折算）
- https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html （topiq_iaa 0.7902 / nima 0.7126）
- https://huggingface.co/q-future/Q-ReAlign-Mini-0.8B （权重 2.21GB、<4GB 显存、26.7 img/s）
- https://github.com/thunderbolt215/ArtiMuse / https://arxiv.org/abs/2507.14533
- https://github.com/zhiyuanyou/DeQA-Score / https://arxiv.org/abs/2501.11561
- https://github.com/ncoevoet/facet/blob/master/docs/zh/THIRD_PARTY_MODELS.md （许可证审计）
- 秩融合稳健性：https://www.cse.msu.edu/~rossarun/pubs/NandakumarJainRoss_MissingDataICB2009.pdf
