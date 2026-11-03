# IAA 与人类偏好模型调研更新（2025H2–2026，补充初版 photo-aesthetic-research-report.md）

> 硬件约束：RTX 5070 12GB / Windows / 本地推理为主。调研截至 2026-10，未发起付费 API。

## 一、IAA 新 SOTA（MLLM 系）

| 模型 | 来源 | 关键数字 | 开源 | 12GB 可跑性 |
|---|---|---|---|---|
| **Q-ReAlign**（Q-Future, 2026-06） | github.com/Q-Future/Q-ReAlign | AVA SRCC：Mini 0.797 / Lite 0.814 / Pro 0.832（七基准均值 0.879-0.896） | 权重+训练+CLI 全开，**已入 pyiqa（`qrealign`）** | Mini<4GB、Lite~8GB 轻松；Pro 9B 需 int4 |
| **DeQA-Score**（CVPR 2025） | github.com/zhiyuanyou/DeQA-Score | IQA 多基准 SRCC ~0.95 量级；输出**分数分布**而非点估计 | 代码+权重开（LLaVA 7B） | 需 4bit 量化 |
| **EvalMuse-40K**（AAAI 2026） | arXiv:2412.18150 | 细粒度图文对齐 SRCC 0.68；Seedream 2.0 采用 | 30K 训练集+代码 | BLIP2 级可跑 |
| **Venus**（CVPR 2026 北大） | arXiv:2602.23980 | 美学问答+CoT 美学裁剪双 SOTA；AesGuide 10,748 张 | 代码开 | 7B 级量化可推理 |
| **ArtiMuse**（CVPR 2026） | arXiv:2507.14533 | **AVA SRCC 0.827**、PARA 0.936；8 维美学属性分解 | 模型+数据集（HF） | 7B 档量化待验证 |
| **Next Token Is Enough**（阿里高德） | arXiv:2503.06141 | AVA 0.828；数字 token 直接回归+CoT 属性推理 | 代码+RealQA 数据集 | 方法可移植到现有 Qwen3-VL |

## 二、人类偏好奖励模型（"大众审美"底座）

| 模型 | 来源 | 数字 | 备注 |
|---|---|---|---|
| **HPSv3**（ICCV 2025） | HF: MizzenAI/HPSv3 | HPDv3 偏好预测 76.9%（PickScore 65.6 / HPS 63.8 / ImageReward 58.6） | 需 prompt（可空/生成 caption）；权重+pip 开源 |
| **PickScore / ImageReward** | arXiv:2305.01569 / NeurIPS 2023 | PickScore 偏好预测 70.5% > 人类专家 68.0% | 经典基线；<4GB；注意训练分布是 AI 生成图 |
| **UnifiedReward-2.0-qwen3vl-2b** | HF: CodeGoat24/... | 2B，Alignment/Coherence/Style 1-5 分 pointwise | 12GB 轻松，可直接当第三路偏好分 |
| **UnifiedReward-Flex** | arXiv:2602.02380 | 个性化奖励：动态层级评估维度+SFT+DPO | 维度动态生成思路可借 |
| **Personalize at Test Time** | arXiv:2610.09015 | 50 个属性锚定偏好维度×用户线性权重（Bradley-Terry），held-out ~77% | ★"大众维度+个人权重"直接可抄骨架 |

## 三、个性化美学（PIAA）最新

1. **VLM 内部表征+线性偏好头**（arXiv:2604.11374, ACL 2026 Findings）：VLM 隐层已编码美学属性，10-shot 线性探针 SRCC 0.32-0.41、100-shot LoRA 0.49-0.58（纯 prompt 仅 0.1-0.26）→ **冻结特征+浅层偏好头数据效率最高**。
2. **PRAC**（arXiv:2607.15752, ACM MM 2026）：偏好富集采样（Collective Controversy 群体争议 + Personalized Deviation 个人偏离挑最有信息量的对）+ 相似用户 cohort 模型合并，四基准 SOTA。
3. **P-MLLM**（arXiv:2604.17233）：零样本 PIAA，用户画像文本做上下文（冷启动）。
4. **PreferThinker**（ICLR 2026）：先预测用户偏好画像再按画像多维评分（可解释两阶段）。
5. **LLM 访谈收集偏好**（arXiv:2605.14761）：主动提问冷启动，超过人类预测者。
6. 数据集：PARA（31,220 图/438 标注者）、FLICKR-AES、**REAL-CUR**（14 个真实用户相册自评——最贴近选片）、LAPIS（CVPRW 2025）。

## 四、"大众先验 + 个人残差"混合方法

- **PALATE**（arXiv:2608.18622, 2026-08）★最直接命中：三层奖励分解 = global backbone（全体共享）+ category residual（审美相近用户群）+ **512 字节 user adapter**（毫秒级打分）；PPR10K 偏好预测 72.83%（PickScore 仅 58.06%）。
- Personalize at Test Time（2610.09015）：线性版同思想。
- 历史脉络：Ren 2017 全局分+个人偏移 SVR、residual adapters（1907.03802）——新工作是这些思想在基础模型特征上的复活，**都收敛到"低维个人参数 + 共享高维先验"**。

## 五、可落地借鉴清单（按优先级）

1. **P0 Q-ReAlign-Mini/Lite 作第二路 IAA 分**：pyiqa 两行接入，与 TOPIQ-IAA（SRCC 0.791）做秩融合/加权；免训练当天可上。
2. **P0 冻结特征+线性偏好头个人残差**（2604.11374 路线）：冻结 DINOv2/Qwen-VL 特征，每用户线性 Bradley-Terry 头，几十~几百对偏好；CPU 秒级训练。
3. **P1 PRAC 式偏好富集标注**：优先展示"群体争议大+与个人历史偏离大"的照片对二选一。
4. **P1 PALATE 三层分解**：global(现有 IAA)→类别残差→512B user adapter，平滑升级路径。
5. **P1 50 维偏好维度线性组合**（2610.09015）：可解释、数据高效（维度可借 ArtiMuse 8 维/RealQA 10 维扩展）。
6. **P2 HPSv3 / UnifiedReward-2.0-2B 第三路偏好分**。
7. **P2 ArtiMuse 8 维属性 / 数字 token 打分法**升级现有 Qwen3-VL 维度评分。
8. P3 Venus 美学批评/裁剪建议；P-MLLM 冷启动画像注入。

## 12GB 可跑性总表

- 轻松（<8GB）：Q-ReAlign Mini/Lite、HPSv3、PickScore、ImageReward、UnifiedReward-2.0-2B、一切线性/MLP 偏好头
- 量化可跑（int4 6-10GB）：DeQA 7B、ArtiMuse、Venus、Q-ReAlign Pro 9B
- 不建议：7B+ 全参微调（QLoRA 2-4B 档可行）
