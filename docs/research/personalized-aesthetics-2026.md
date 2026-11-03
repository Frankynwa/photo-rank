# 大众审美 → 个人审美：个性化审美学习方案（2026 调研）

> 全部为公开论文/开源资源；调研 2026-10 完成，未调用付费 API。

## 一、大众审美基础（审美先验）

1. **AVA**（CVPR 2012）：255K 图、每图 78-549 票 1-10 分评分分布；IAA 校准黄金标准（注意含比赛主题偏差）。
2. **LAION-Aesthetics V2**：**CLIP ViT-L/14 + 简单线性头**（SAC 176k+AVA 250k 评分对训练），官方结论"简单线性排序优于 MLP"→ **冻结 CLIP 特征+线性头就是审美先验的正确形态**，也是个人残差头的模板。
3. 偏好模型证据链：PickScore（50 万偏好对，预测 70.5%>人类专家 68.0%）、HPS v2（798K 偏好选择）、ImageReward（137k 专家对比对）。⚠️ 训练分布是 AI 生成图，对真实照片降权使用。
4. **TOPIQ-IAA 定位**（pyiqa 官方 benchmark）：topiq_iaa 0.791，开源第一梯队，作 s_base 主干合理。

## 二、个人偏好采集 UX 范式

1. **对比式打分**（信息量最高）：两图二选一+平局（Pick-a-Pic 范式）；ImageReward 137k 对即训出强 reward
2. **组内选优**（选片场景免费信号）：连拍/相似组挑 best ≈ N-1 条 pairwise 标签——**PhotoRank 的 L2 组内选优天然是偏好流**
3. 显式评分：星级/1-5 分+维度（对齐现有维度评分）
4. **隐式反馈**：keep/reject 即正负样本（culling 工作流本身=偏好流）；Aftershoot"随使用学习"、FilterPixel memory-based learning 宣称 85-95% 一致率；Aftershoot 个人画像需 2,500-5,000 张历史（工业界样本量参考）
5. 拖拽排序→listwise（LambdaRank）；Elo/TrueSkill 维护组内能力值
6. **主动学习**（建议内置）：Active-PAM（ICCV 2017）按"多样性+大残差"挑图打分，10→100 张收益最大化

## 三、个性化建模路线（按工程成本 d<a<b<c）

| 路线 | 冷启动样本 | 12GB 可行性 | 证据 | 预期 |
|---|---|---|---|---|
| **(d) VLM prompt 风格注入** | 5-20 张 | 需本地 VLM 或一次性画像蒸馏 | ACL 2026 同款 prompt 协议 | 个位数提升，胜在即时可解释 |
| **(a) 冻结特征+线性/MLP 偏好头（推荐默认）** | 10 张可用/100 张显著 | ✅ 特征预提取后 CPU 训，头 KB 级 | Ren ICCV 2017 残差模型（k=10 有效、k=100 +0.039）；ACL 2026 线性探针；ECCV 2024 Task Vector（10-shot 0.561） | 稳定正收益 |
| **(b) Pairwise RankNet/LightGBM** | 50-200 偏好对 | ✅ LightGBM 纯 CPU | RankNet/PickScore pairwise 目标；与 top-k 目标天然一致 | 直接优化"选得对不对" |
| **(c) LoRA 微调小模型** | 500-2,000+ | CLIP-L LoRA 轻松；2-4B VLM 4bit QLoRA 可行 | PRAC（四基准 SOTA） | 上限最高，防过拟合只训残差 |

## 四、开源可抄清单

- **LAION-AI/aesthetic-predictor**：CLIP+线性头全代码（<50 行），残差头直接照抄
- **YeolJ00/Personalized-Aesthetics**（ECCV 2024, MIT）：6 个预训练 IAA 模型+PIAA 训练脚本——最直接可抄
- **ynklab/vlm-latent-piaa**：VLM 隐层 PIAA 全套（linear probe/residual/LoRA 基线）
- lwchen6309/aesthetics_transfer_learning（BMVC 2025）+ pip 包 unified-iaa（8 个 PIAA checkpoint，含人口学/Big5 条件）
- yzc-ippl/PRAC（代码将开）；数据集 PARA / FLICKR-AES / **REAL-CUR**（14 个真实用户相册自评）/ LAPIS
- ⚠️ 无现成端到端"个性化照片选片"开源项目（Aftershoot/FilterPixel 均闭源，文档是 UX/样本量参考）

## 五、隐私与本地化

- 单机路线（首选）：特征提取→偏好头（KB-MB）→LoRA（10-100MB）全在 12GB 本地；偏好数据落本地 SQLite/Parquet，原图不出设备（行业标配卖点）
- 联邦路线：**FedPIAA**（ICME 2023）：全局模型学大众审美+本地学个人，接近集中式 SOTA——"大众先验持续更新"的隐私合规通道
- ⚠️ 云端 VLM 打分会外泄照片内容——须本地量化 VLM 或只传脱敏风格画像文本

## 六、推荐架构：大众先验 + 个人残差（模块化草案）

**score(x,u) = s_base(x) + λ(n_u)·Δ_personal(x,u)**，λ 随个人样本数 n_u 退火（0→1），样本不足自动退回大众分。

| 模块 | 内容 | 冷启动 | 工程量（单人） |
|---|---|---|---|
| M1 特征提取器（冻结） | CLIP ViT-L/14+DINOv2+现有技术分 | — | 1-2 周 |
| M2 大众先验 s_base | TOPIQ-IAA + LAION 线性头 + Qwen3-VL 维度分平台权重融合 | — | 调参 1 周（主体已有） |
| M3 偏好采集 UI | 二选一对比+组内 best-pick+星级+隐式信号 | — | 2-3 周 |
| M4 偏好数据湖 | SQLite 偏好对/评分/隐式信号，版本化 | — | 1 周 |
| M5 个人残差头（默认） | Ridge/MLP on [f_CLIP;f_DINO;s_base;维度分]→Δs | 10-100 张 | 1-2 周 |
| M6 个人排序器（进阶） | LightGBM LambdaRank / RankNet 小 MLP | 50-200 对 | 1-2 周 |
| M7 LoRA 适配器（高阶） | CLIP-L 或 2-4B VLM 残差精调 | 500+ 张 | 2-3 周 |
| M8 VLM 风格注入（零训练旁路） | 风格画像 prompt+few-shot 注入维度评分 | 5-20 张 | 3-5 天 |
| M9 融合与解释 | score+reason 输出 | — | 1 周 |
| M10 评测闭环 | 留出 pairwise accuracy/NDCG，A/B，λ 退火监控 | — | 1 周 |

**MVP = M1-M5+M9 ≈ 4-6 周；全量 2-3 个月。** 冷启动路径：d(5-20 张)→a(10-100 张)→b(50-200 对)→c(500+ 张)，与 λ 退火一一对应。

## 主要来源

AVA(CVPR2012)；laion.ai/blog/laion-aesthetics；Pick-a-Pic(2305.01569)；HPSv2(2306.09341)；ImageReward(NeurIPS2023)；Ren ICCV2017；Task Vector(ECCV2024)；vlm-latent-piaa(ACL2026)；PRAC(2607.15752)；PALATE(2608.18622)；FedPIAA(ICME2023)；PARA(CVPR2022)/LAPIS(CVPRW2025)；pyiqa benchmark；FilterPixel/Aftershoot 公开文档
