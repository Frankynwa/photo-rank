# P0-3 定额出片 + Score&Reason — 落地决策 2026-11

> 状态：调研完成，**待拍板**。未改任何代码。
> 基础设施：现有 0-10 批次归一化综合分、VLM 多维评分（构图/色彩/光影/总体+硬伤+题材分类）、L2 相似度分组（连拍组）。

## 结论先行

- **定额语义（硬承诺）**：输出恰好 `min(N, M′)` 张，M′=无硬伤候选数；不足 N 时全选+明示原因，**绝不凑数放行硬伤**。这比 Aftershoot（"极端比例可能不准"）和 FilterPixel（"你自己调整"）都干净。
- **算法**：硬伤淘汰 → 连拍组代表 → 分段配额（sqrt 平滑+余量回收）→ 段内 MMR 多样性选优 → 全局补足。对齐 Aftershoot 公开语义（quality+coverage+variety、强制去重），配额维度是我们的差异化（Aftershoot 只给全局数）。
- **评分桶**：精选/可用/淘汰 三桶 = 批次分位数（P85/P20）与固定底线（7.0/4.0）**取严**，硬伤一票否决。
- **Score&Reason**：评分同调用顺带生成（schema 顺序 **(reason, score)** 先推理后分数），确定性模板兜底 100% 覆盖，入选张可选批量润色。理由 ≤30 汉字、事实短语、锚定可核对的测量值。

## 1. 商业工具定额语义对比

| 维度 | Aftershoot Cull to Target | FilterPixel Magic Number | Lightroom Assisted Culling |
|---|---|---|---|
| 定额 | 精确 N（滑杆 10–60% 或精确张数；极端比例官方承认会不准） | 精确 N（"calibrated to your target"，事后可调） | **无定额**，阈值过滤 Select/Reject |
| 选法 | **质量+覆盖+多样性联合构建**，官方明确反对"排序截断" | 相似度分组校准 + DeepCull 每组至少 1 张 | 各信号独立阈值（Subject/Eye Focus、闭眼、误拍单据） |
| 连拍组 | 定额时强制开启去重组（不可关）；无 Highlights 桶，用 Maybe=组内次优 | 分组+组内选优 | 无 |
| 配额粒度 | 全局 1 个数（官方明确不做分人/分段配额） | 全局 1 个数 | 不适用 |
| 解释 | 星标/色标/桶（无逐张理由） | 每张分数+一句理由 | 分项分数+判定依据（无自然语言理由） |

设计原则摘录（Adobe 官方设计文）：信号独立评分不给黑盒总分、"Assist, don't decide"、拒绝刻意保守优化 keeper 召回。

## 2. 恰好 N 张算法对比

| 方案 | 优点 | 致命退化情形 |
|---|---|---|
| a) 全局 top-N | 最简 | 连拍 12 张全收变"同瞬间 12 连张"；高分段霸榜、低分段清空 |
| b) 组代表优先+补足 | 天然去重 | 组数>N 时退化为 a；组数≪N 时补足张重复感回升 |
| c) 分段配额+组内选优 | 覆盖最好（Aftershoot "coverage" 的可落地版） | 小段被迫放水→需配额回收；分段错误错配 |
| d) MMR 多样性 top-N | 自动抑制重复 | 可解释性差；λ 过大牺牲质量 |
| **推荐 b+c+d 混合** | 兼顾质量/覆盖/多样性/可解释 | 实现量中等 |

**推荐伪代码**：
```text
输入: photos(score 0-10, hard_flags, group_id, segment_id, emb), N
1. 硬伤淘汰: 候选 = 无 hard_flags（闭眼/糊/误触）
2. 组代表: 每组 argmax score → reps；组内其余 → alts
3. 分段配额: q[t] = N·sqrt(|t|)/Σsqrt(|t'|)；q[t]=min(q[t], |reps∩t|)；差额回收给未满段
4. 段内 MMR: S={段内最高分}; while |S|<q[t]: 取 argmax[s(i) − λ·max_{j∈S} cos(emb_i,emb_j)]
5. 全局补足: Σ|S|<N 时从 reps−S ∪ alts 按分降序补（同组仅在代表全用尽后放第 2 张）
6. 退化: M′<N → 全选+warning（不放行硬伤）；组极多而小 → 代表池上按分+多样性截断
复杂度: O(M) 分组 + O(|reps|²·d) 段内贪心（向量化后可忽略）
```
分段建议：以**时间**为主（拍摄段）、题材分类为辅；UI 允许手动调段配额。

## 3. 评分桶设计

- **淘汰**：硬伤一票否决 或 score < max(批次 P20, 4.0)
- **精选**：score ≥ max(批次 P85, 7.0) 且无硬伤
- **可用（Maybe）**：其余；保留"组内次优"标记（Aftershoot 语义），是定额补足第一来源
- 分位数吸收批次难度，固定底线保住绝对语义；随题材平移（PixCull verticals 思路）
- 每张同时存 `score_abs`（模型原始分）+ `percentile`，跨批次可比
- 学术支撑：绝对分需校准锚、纯分位数失去绝对语义→混合最好（PortraitAes 高斯校准、ScenePercept 三目标校准）

## 4. Score&Reason 对比与推荐

业界形态：
- FilterPixel DeepCull："Score: 94. Reason: Keynote standing ovation, sponsor banner visible, audience engaged."——一句 8–15 词、并列事实短语
- framejury：**VLM 当证人不当法官**——VLM 只出结构化观测，200 行规则拼理由（含清晰度数值）；组内前 4 清晰张进模型控成本；前两名差 ≤5 分标 Review；同组双向各跑一遍抑制位置偏置
- PixCull：6 轴评分带 rationale；测量值作为证据交裁判；输出 XMP sidecar + IPTC Caption

| 维度 | 同调用顺带生成 | 独立调用 |
|---|---|---|
| 成本 | 只多输出 token（一句中文 30–60），不重复图像编码 | 图像 token 翻倍，万张级不可行 |
| 一致性 | 分与理由天然对齐 | 两次调用可能矛盾 |
| 质量 | 有"给分辩护"风险 | 上限更高、可引用上游信号 |

**推荐两段式**：
1. **主调用（每张必跑）**：现有 VLM prompt 加 `observations`（客观事实短语）与 `reason_draft` 字段，schema 顺序 (reason, score)（Structure Tax：先答案后推理在多步任务严重退化）。
2. **模板兜底**：reason_draft 缺失/超长/矛盾时用确定性模板拼装（"连拍 5 张中最清晰｜表情自然｜构图完整"）——100% 覆盖零成本。
3. **可选精修**：只对入选 N 张一次批量润色（带 token 预算台账）。
4. **可信度**：组内险胜加"险胜"标记；理由引用的硬伤须与 hard_flags 一一对应（防 VLM confabulate）；理由锚定测量值。
- 句式：中文一句 ≤30 字、不用"这张照片"开头、3 种句式轮换（入选夸亮点/补足夸覆盖/险胜标差异）。
- 输出载体：**JSON 为真源**（score/score_abs/percentile/bucket/dims/hard_flags/group/segment/reason/reason_source/evidence）→ CSV 报告 → XMP Description（写前读回合并，防覆盖用户已有元数据——PixCull 教训）。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| "恰好 N"被误解为凑数 | 硬承诺只对 M′ 有效；不足时 warning，不放行硬伤 |
| 截断线附近随机性 | 保留双分；险胜标记；一键"换入换出"（swap 不 add） |
| 连拍组自动降级误杀好片（PixCull 盲测 87% 会保留被迫回滚） | 组内排序"只报告不执行"，定额选中才生效 |
| VLM 理由幻觉/位置偏置（4B 模型 49% 选第一张） | 模板兜底+锚定测量值；组内比较双向跑 |
| 理由与分数矛盾 | (reason, score) schema 顺序 + 一致性校验 |
| 分段配额错配 | 余量回收；时间为主题材为辅；UI 手动调 |

## 6. 待拍板

1. 定额算法按推荐混合（b+c+d）实现，还是先做简版（b+全局补足）再演进？
2. 桶阈值 P85/P20 + 底线 7.0/4.0 起步值认可？
3. Score&Reason 走"同调用+模板兜底"（推荐），还是纯模板先省 VLM 改动？
4. 精修润色（入选张批量 LLM）本期要不要？

## 来源

- https://support.aftershoot.com/en/articles/11533911-cull-to-target-with-aftershoot
- https://aftershoot.com/blog/how-to-control-final-photo-count-with-cull-to-target （quality+coverage+variety）
- https://filterpixel.com/blog/how-to-use-filterpixel-the-complete-feature-walkthrough / https://filterpixel.com/deepcull
- https://helpx.adobe.com/lightroom-classic/desktop/organize-photos-in-lightroom-classic/assisted-culling.html
- https://adobe.design/ideas/behind-the-design-assisted-culling-in-adobe-lightroom
- https://github.com/ChrisChen667788/pixcull （三桶阈值+XMP 教训）
- https://github.com/carlosjcastro/framejury / https://dev.to/carlosjcastrog/framejury-a-culling-assistant-that-works-offline-and-explains-every-photo-it-rejects-14jm （证人模式+close-call）
- 结构化输出：https://arxiv.org/pdf/2610.12056 （Structure Tax）、https://arxiv.org/pdf/2609.29496
- VLM 偏置/幻觉：https://arxiv.org/abs/2610.01243 、https://arxiv.org/pdf/2610.35247
- 校准：https://arxiv.org/pdf/2610.05010 、https://arxiv.org/pdf/2610.17661v1
- MMR/次模：https://arxiv.org/pdf/1906.11285v1.pdf 、https://research.google/blog/introducing-gist-the-next-stage-in-smart-sampling/
