# AI 照片筛选/选片（Photo Culling）全景调研（2025–2026）

> 调研基准：以下载量/用户量/活跃度论主流，不以 star 数论英雄。价格为官网口径（2026-09/10 核查）。找不到用户量证据的均明说。

## 一、商业选片软件

| 软件 | 核心选片能力 | 定价 | 用户量证据 |
|---|---|---|---|
| **Aftershoot** | 30+ 因素打分；重复分组/闭眼模糊旗标；AI-Automated 全自动 + AI-Assisted 人拍板双模式；**Cull-to-target 定额出片**；Key Face Detection 主体人脸优先；AI 学习个人选片风格；100% 离线；集成 LrC/C1/Photo Mechanic | Select $10/月；Complete $45/月 | **248,000+ 摄影师、8.8B+ 张**（官方 2026 路线图） |
| **FilterPixel** | Basic Cull（清晰/曝光/睁眼/近重复不限量）；**DeepCull 26+ 题材模型**，10 维评分含表情/peak action/叙事性/瞬间时机，每张 **Score + 文字 Reason**；best-of-burst；Magic Number 定额出片 | Standard $14.99 / Pro $34.99 / Studio $64.99 月 | 14,000+ 摄影师（官网） |
| **Narrative Select** | Scenes View 相似帧聚组+组内排名；逐脸 Eye/Focus Assessment；**Close-ups Panel 人脸裁切并排对比**、主角人脸锁定；Smart Face Zoom | Free / Pro $15/月 | 用户数未公布；种子轮 $2.58M |
| **Lightroom Assisted Culling** | 2025-10 v15.0 上线：Subject/Eye Focus、Eyes Open（含 Can't Tell 不确定档）、Exposure issues、**误拍单据识别**；逐脸得分面板；Auto Stack 时间+视觉相似堆叠 | CC 订阅内 | 功能级用户数未公布 |
| **Imagen AI** | Cull to Exact Number 定额出片、带编辑预览选片、个人风格 profile | 选片附加 $12/月起 | 1.5 亿+张/年、数万客户（2022） |
| Picktorial | ⚠️ 不是 AI 选片工具（Mac 编辑器+DAM） | — | 找不到 |

## 二、自托管/开源

| 项目 | ML 能力 | curation | 用量证据 |
|---|---|---|---|
| **PhotoPrism** | ONNX 标签/人脸（YuNet+SFace）、Ollama/OpenAI 图说 | Quality Filter 低质图剔除、Stacks、Moments | **Docker Hub 拉取 84,205,322 次**（实查） |
| **Immich** | CLIP Smart Search（VectorChord）、人脸、OCR、标签 | 无自动选片；收藏/星级/Archive/Memories | 未公布（star 仅参考） |
| **digiKam** | OpenCV DNN 人脸/红眼/Auto-Tags/**Aesthetic Detection→Pick Label** | 美学分级即精选；相似指纹找重复 | 无权威总数 |
| LibrePhotos | 人脸/物体/场景/CLIP 语义搜索 | 时间线聚合 | 找不到 |

**2026 开源选片新势力**（直接对标 PhotoRank）：
- **picturesque**（2026-10, GPL-3）：连拍归系列、清晰/曝光/噪点打分、最佳张推荐、XMP sidecar 互通 LrC/digiKam
- **best-photo-picker**（MIT）：五套人脸方案、dHash+CLIP 双重去重、**按人加权**（选片偏向娃）、时间多样性（每日上限）
- **facet**（2026-02, 1933 commits）：9 维评分、Web 画廊精选、推送数码相框/Home Assistant
- **framejury**（2026-10）：近重复分组→组内清晰度筛→**本地 VLM 当裁判**（Gemma via Ollama）判闭眼/表情并自然语言解释，**双向输入消位置偏差**
- fixxer（Mac）：按题材评分、XMP sidecar 回流 LrC/C1

## 三、手机厂商系统能力

| 厂商 | 机制 | 可借鉴 |
|---|---|---|
| Google **Top Shot**（Pixel 3） | 快门前后 ±1.5s 取 90 帧，端上打分：光流+曝光+陀螺仪+语义（微笑/睁眼/表情），留 2 张备选跑 HDR+ | 移动端连拍选优完整特征清单 |
| Google **Best Take**（Pixel 8） | 相似照片→人脸姿态+表情信号→每人换脸合成"都在笑"合照 | 合照表情问题终极解法；华为已跟进 |
| Apple Photos | 端上策展（Days/Months 偏好构图、Liveliness 混排）、知识图谱、Memories | "策展式时间线"：控制呈现节奏与多样性 |
| 小米 | 动态照片"保存优选照片"（连拍帧挑精彩另存）；AI 相册搜索 | Live 图帧级选优轻交互 |
| 华为 | nova 13 AI 最佳表情（2 张+连续拍摄换表情）；AI 云增强合影 | 合照表情替换已落地国产旗舰 |

## 四、"最佳张"技术线索

1. 分组：时间戳聚类（Best Take）；时间+视觉相似（Lr Auto Stack）；CLIP/深度 embedding 85-90% 阈值近重复；感知哈希剔精确重复
2. 组内打分：清晰度/曝光/噪点/睁眼 EAR/微笑/构图美学（Aftershoot 30+ 因素、FilterPixel 10 维、Facet 9 维、Lr 逐脸）
3. **"时刻"维度（2025-26 差异化点）**：peak action、瞬间时机、叙事性（FilterPixel DeepCull、Aftershoot 2026 Key Subjects）
4. **VLM 裁判（新趋势）**：本地 VLM 判组内闭眼/表情+可读理由；双向输入消位置偏差（framejury）
5. 学术：Burst IQA（arXiv:2511.07958, AAAI 连拍质量统一框架）、Light-Head Adversarial（1803.07212 端上实时选帧）、相册级美学选片（HAL tel-02345620）

## 五、PhotoRank 可抄清单（13 条）

1. **评分桶+双模式**（Aftershoot）：Selected/Highlights/Closed Eyes/Blur/Duplicates 分桶 + "AI 自动选/AI 辅助人拍板"两档
2. **定额出片**（Aftershoot Cull-to-target / FilterPixel Magic Number）：输入"要 100 张"按质量+多样性配额出片——社交场景刚需
3. **每张 Score + 一句话 Reason**（FilterPixel/framejury）：可解释性换信任
4. **题材化模型**（FilterPixel 26+）：旅行/人像/宠物/美食/娃不同权重
5. **相似组视图**（Narrative Scenes/Lr Auto Stack）：组内排名、人脸裁切并排对比
6. **逐脸评估**（Narrative/Lr）：每人脸睁眼/清晰度红黄绿，合照"谁不行"一目了然
7. 表情替换兜底（Best Take/华为）
8. **去重双保险+时间多样性**（best-photo-picker）：pHash+CLIP + 每日/场景上限
9. 移动端帧级信号（Top Shot：光流/陀螺仪/曝光）
10. **XMP/星级互通**（picturesque/fixxer）：结果回流 LrC
11. 质量过滤前置（PhotoPrism/Lr）：截图/单据/纯色图先剔
12. 本地/隐私卖点（全行业标配叙事）
13. 技术筛不限量+审美精选计费分层（FilterPixel）

## 六、来源

商业：aftershoot.com/selects、aftershoot.com/roadmap-2026、filterpixel.com/deepcull、narrative.so/select/*、helpx.adobe.com/lightroom-classic（assisted-culling）、petapixel/techcrunch 2022 Imagen 融资
开源：immich.app/docs/features/searching、docs.photoprism.app/user-guide/ai、hub.docker.com/v2/repositories/photoprism/photoprism、digikam.org/about/features、github.com/{denyszorinets/picturesque,Arkalogy/best-photo-picker,ncoevoet/facet,a45674567/fixxer,carlosjcastro/framejury}
手机：research.google/blog/top-shot-on-pixel-3、blog.google Best Take、apple.com Photos Tech Brief、consumer.huawei.com、mi.com
学术：arXiv:1803.07212、arXiv:2511.07958、HAL tel-02345620
