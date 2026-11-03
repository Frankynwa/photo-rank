# P0-2 Live 图（实况照片）解析与原生输出 — 落地决策 2026-11

> 状态：调研完成，**待拍板**（MVP 范围）。未改任何代码。
> 约束：Windows 11、全离线、Python；已有 opencv-headless / pillow-heif；ffmpeg 可装。

## 结论先行

- **MVP = 安卓 MotionPhoto 输出**（JPEG+MP4 字节拼接 + 手写 XMP），低风险、纯离线、1–2 周量级。换封面帧 = 换 JPEG + 改 PTS/长度字段，**不重编码视频**。
- **Apple 输出放阶段 2**（风险高、必须真机验证）：关键难点是 MOV 的 mebx still-image-time 定时轨——ffmpeg 写不了也 copy 不了，需手写 box / Bento4 移植 / 模板法。
- 最佳帧公式：`score(t) = w1·清晰度 + w2·运动显著性 + w3·人脸加成 + w4·曝光门 − λ·|t−t封面|`，纯 OpenCV，按 **PTS** 索引（iPhone MOV 是 VFR）。
- pymobilephoto3 **查无此库**（PyPI/网页均无），原方案"exiftool + 自研 XMP 解析"路线成立，Immich 公式可直接抄。

## 1. 格式解析字段对照（精确到字段名）

| 格式 | 配对/定位字段 | 视频定位 | 封面帧字段 |
|---|---|---|---|
| Apple HEIC/JPG+MOV | 静态图 EXIF MakerNote **Apple:ContentIdentifier (0x0011)**；MOV moov/meta 的 **com.apple.quicktime.content.identifier**（[Keys]/ilst） | ContentIdentifier/UUID 配对（大写带横线！） | **com.apple.quicktime.still-image-time** mebx 定时轨：样本值恒 0xFF，封面时刻=样本 PTS（timescale 600） |
| MotionPhoto **MicroVideo v1**（Pixel/旧小米） | XMP GCamera:MicroVideo=1、MicroVideoVersion=1、MicroVideoOffset、MicroVideoPresentationTimestampUs | JPEG 尾附 MP4；**MicroVideoOffset=从文件末尾倒数的字节数** | PresentationTimestampUs（µs） |
| MotionPhoto **1.0/2.0**（Pixel/HyperOS/现代三星OPPOvivo） | XMP Camera:MotionPhoto=1、MotionPhotoVersion=1、MotionPhotoPresentationTimestampUs + **Container:Directory**（rdf:Seq：Item:Mime/Semantic/Length/Padding） | 尾附 MP4；Item:Length 从末尾倒数，MotionPhoto 项必须在文件最末尾 | MotionPhotoPresentationTimestampUs（−1 时读者取视频中点） |
| 三星 SEF | JPEG 尾块 [0x0A30 u16] MotionPhoto_Data + SEFH/SEFT 索引（version 101→107），文件以 ASCII **SEFT** 结尾 | 走 box 树确定视频结尾，**不能 rfind FFD9** | XMP MotionPhotoPresentationTimestampUs |
| vivo | 旧：双文件 JPG+MP4（MediaGroupUUID 配对）；新：MotionPhoto v2 + VCamera ns + 私有尾块（只读） | 双文件或尾附 | PresentationTimestampUs |
| OPPO/华为 | MotionPhoto v2 + OpCamera ns / 尾附 MP4（H.264/H.265） | 尾附 | 同 v2 |

Immich 可直接抄的公式：`position = fileSize − length − padding`（MicroVideo v1 则 length=MicroVideoOffset）；检测 = tags.MotionPhoto || tags.MicroVideo；三星走 `exiftool -b MotionPhotoVideo`；双文件配对键 = ContentIdentifier || MediaGroupUUID。

## 2. 解析库选型（Windows 离线）

| 组件 | 用途 | 结论 |
|---|---|---|
| **exiftool.exe** | 万能读（-G -ee / -XMP:all -struct / -b MotionPhotoVideo） | ✅ 读路径首选 |
| **手写 XMP（APP1 包/regex）** | 写 Container:Directory rdf:Seq struct | ✅ 唯一稳妥写法（pyexiv2/exiftool 对 struct 都别扭） |
| **pillow-heif** | HEIC 读写（`convert_hdr_to_8bit=False` 保 HDR） | ✅（已在 requirements） |
| **pymp4** | 读 MOV box 树、辅助拼 mebx | ✅ 纯 Python |
| **doodspav/motionphoto**（pip） | 一行生成 Google+三星 MotionPhoto（含 SEF 尾块逆向） | ✅ 安卓输出直接用/抄 |
| python-xmp-toolkit | — | ❌ Windows 上 exempi.dll "基本不工作" |
| pymobilephoto3 | — | ❌ 查无此库 |
| makelive | Apple 配对参考 | ❌ macOS-only |
| live-photo-box（C#/C++） | Windows 上最全协议参考（含 mebx/MakerNote 写手） | 参考实现 |

## 3. 输出写路径

### 3.1 安卓 MotionPhoto（MVP，低风险）
1. 最佳帧 → JPEG（EXIF/ICC 用 exiftool -TagsFromFile 拷贝）；
2. 手写 XMP APP1：GCamera:MotionPhoto=1 + MotionPhotoVersion=1 + MotionPhotoPresentationTimestampUs=新帧 PTS + Container:Directory 两项（Primary: image/jpeg; MotionPhoto: video/mp4, Length=视频字节数）；兼容老读者双写 GCamera:MicroVideo 全套；
3. 拼接 JPEG(EOI)+MP4。Length/Offset 是末尾倒数长度，**视频没变就不用改**，只改 PTS；
4. 三星可选：追加 SEF 尾块（doodspav 已实现），必须是最后一个 trailer；
5. 顺序纪律：XMP 写完再拼视频；JPEG 结尾走 marker 树找 EOI，不能 rfind(FFD9)。

### 3.2 Apple（阶段 2，必须真机验证）
1. 静态图 MakerNote 0x0011：exiftool **不能从零建 MakerNote**。可行：a) 真机 donor 图 `-TagsFromFile -makernotes` 整块拷贝后改单条；b) 按 Apple iOS(NUL)+0x0001+MM+IFD 字节手写 patch（已验证与真机逐字节同构）。
2. MOV ContentIdentifier：**exiftool -Keys:ContentIdentifier**（写 moov/meta，实测可配对）；⚠️ ffmpeg `-movflags use_metadata_tags` 写进 moov/udta/meta，**iOS 不认**（社区"最贵的 bug"）。同时写 live-photo.auto=1 与 full-frame-rate-playback-intent=1。
3. still-image-time mebx 轨：ffmpeg 写不了（remux 即丢）。三条路：**手写 box**（stsd(mebx)+stts+elst 空 edit；样本 [9][1][FF]；moov 后置于 mdat 追加 trak 不动 stco）/ **Bento4** mp4extract 真机 trak + mp4edit 插入改时刻 / **模板 MOV**。主推手写，字节级对齐真机。
4. 导入 iOS：爱思助手"导入实况照片"（真机 iPhone 15/iOS 27 验证过）；**扩展名必须大写 .JPG/.MOV**；两文件同批导入。

## 4. 最佳帧算法（便宜、离线、CPU）

```
候选帧: 全片 10–30fps 采样, 灰度缩到 ~320px
score(t) = w1·sharp_norm(Laplacian/Tenengrad)      # 每帧 1–5ms
         + w2·motion_norm(帧差, 光流幅值惩罚抖动)
         + w3·face_bonus(OpenCV 人脸 ×1.3)
         + w4·exposure_gate(均值 20–233 + 熵门)
         − λ·|t − t_cover|                            # 贴封面 vs 换更好的帧
```
- 按 **PTS** 不按帧号（VFR）；写回的 PTS 必须对应真实可解码帧，安卓侧尽量靠近关键帧。
- 显著性检测（cv2.saliency）在 opencv-contrib，headless 没有——不依赖它。
- 先例：安卓官方视频轨里就有 MotionPhotoFrameScoreDescr（score+PTS），"帧评分+PTS"就是官方做法。

## 5. 兼容性与坑（验收必查）

1. moov/meta ≠ moov/udta/meta（ffmpeg 写错位置 = iOS 不认）。
2. **HDR/10bit**：iPhone MOV 多为 10-bit HEVC + bt2020；直接重编码会灰片。ffprobe 看 color_transfer，zscale/libplacebo tone map 并显式输出 bt709 标签——只改标签不转像素无效。
3. HEIC XMP 多 item：挑 iref cdsc 指向 pitm 的那个；HDR 图主图可能没 XMP item。
4. 旋转：MOV tkhd display matrix；封面必须处理旋转否则"实况照片倒置"（朋友圈有此 bug）。
5. XMP 残留假阳性：MotionPhoto=1 可能残留但视频已剥离，读侧必须验证视频真实存在。
6. UUID 大写带横线（去横线是致命 bug）。
7. 平台发布是**逐厂商/逐 App 适配**：小红书 iOS v7.60+ / 朋友圈 iOS 8.0.51+ / 抖音 iOS 支持；安卓要厂商相册先认成实况（小红书 ≥8.50.1 且 MagicOS 8.0+ 等）。"相册认出"≠"平台能发"，需真机真账号端到端验收。

## 6. MVP 范围（待拍板）

- **阶段 1（建议本期）**：安卓 MotionPhoto 1.0 输出（双写 GCamera v1 + Container:Directory + PTS），可选三星 SEF；解析覆盖 MotionPhoto/MicroVideo/三星，vivo/OPPO/华为按"尾附 MP4 + ftyp 扫描"兜底识别不承诺全解析。
- **阶段 2**：Apple JPG+MOV 输出（手写 MakerNote + exiftool Keys + 手写 mebx），大写扩展名，自动校验（CID 相等/box 位置/mebx PTS/宽高比 Δ<0.05）后真机验收。
- **明确不做**：HEIC 主图 MotionPhoto、伪造遥测、.livp、HDR 增益图重建。

## 7. 风险

| 风险 | 对策 |
|---|---|
| 平台可能对合成实况做来源校验 | 阶段 1 只承诺"相册识别为实况"；发布链路真机验收后再对外说 |
| Apple 隐性要求随 iOS 变化（mebx "必须"vs"别加"两派实测） | 锁一台真机做回归基准 |
| 三星 SEF/vivo 尾块是逆向文档，版本会变 | 解析失败自动降级为普通照片，不影响主链路 |
| 10bit/HDR、旋转处理不当出灰片/倒置 | ffprobe 分支 + tone map + 人工抽检 |

## 来源

- https://developer.android.com/media/platform/motion-photo-format （官方格式规范）
- https://github.com/immich-app/immich/blob/main/server/src/services/metadata.service.ts （提取公式）
- https://git.xszz.org/reviventura/LivePhotoTools/blob/main/docs/FORMATS.md （mebx/MakerNote 逆向）
- https://github.com/mufeng510/ai-image-process/blob/main/docs/live-photo-testing.md （真机逐字节验证）
- https://github.com/doodspav/motionphoto （安卓+SEF 写出实现）
- https://github.com/LengxiQwQ/live-photo-box （Windows 全协议参考）
- https://discussions.apple.com/thread/255509516 / https://github.com/RhetTbull/makelive/issues/28 （iOS 配对实证）
- https://www.hackerfactor.com/blog/index.php?%2Farchives%2F1039-Reversing-Samsung-Metadata.html （SEF 布局）
- 坑：https://stackoverflow.com/questions/65376581/write-live-photo-metadata-to-video-using-ffmpeg / https://issues.apache.org/jira/browse/TIKA-4777
