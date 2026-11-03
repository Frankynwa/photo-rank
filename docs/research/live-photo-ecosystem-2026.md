# Live Photo（实况照片/Motion Photo）技术生态调研（截至 2026-11）

> 格式细节经查证：Android 官方规范、exiftool Tag 文档、Immich/PhotoPrism 源码、GitHub API 实测。

## 一、格式全景

| 维度 | Apple Live Photo | Google MicroVideo v1（废弃） | Google MotionPhoto 1.0 | Samsung v2/v3 | vivo/OPPO 双文件 |
|---|---|---|---|---|---|
| 结构 | HEIC/JPG + MOV 两文件 | JPEG 尾附 MP4 | JPEG/HEIC/AVIF 尾附 MP4 | JPEG(SEF trailer)/HEIC(mpvd+sefd) | JPG + MP4 两文件 |
| 配对/定位 | UUID ContentIdentifier 双写（图侧 Maker Apple 0x0011；MOV com.apple.quicktime.content.identifier） | MicroVideoOffset（末尾倒数） | XMP Container:Directory 的 Item:Length+Padding | SEF MotionPhoto_Data / MotionPhotoVideo 标签 | 同名 stem + 私有 ID |
| 封面帧时间 | MOV timed metadata（still-image-time）PTS / MakerNote 0x0017 LivePhotoVideoIndex | MicroVideoPresentationTimestampUs | MotionPhotoPresentationTimestampUs | 同左 | 无标准 |
| 现状 | iOS 全系；视频约 3s（前 1.5+后 1.5） | **官方定为 MUST ignore** | Android 现行（支持 Ultra HDR gainmap；文件名主名以 MP 结尾） | 三星现行 | 厂商私有，Immich/Google 均不识别 |

2025-26 变化：MicroVideo 废弃；三星 mpv3/HEIC 普及（exiftool 12.85+ 补齐）；小米 HyperOS 3 切新标准；vivo/OPPO 双文件暴露缺陷（motion-photo-bridge 补救）；iOS 18 HEIC 用 ISO 21496-1 gainmap（Google Photos 才认 HDR）。

## 二、处理库清单（star 为 2026-11 实测）

| 库 | 语言/活跃度 | 用途 |
|---|---|---|
| exiftool | Perl，事实标准 | ContentIdentifier/MotionPhoto*/Apple MakerNotes 全覆盖；⚠️ Container:Directory 未实现（与 depth-map 标签冲突），需自写 XMP 解析 |
| **wszqkzqk/live-photo-conv** | Vala，235★，2026-10 活跃，MSYS2/AUR 有 | 提取/合成/修复/逐帧导出/**长曝光生成**；ftyp 兜底；liblivephototools 可被 Python 调 |
| **PetrVys/MotionPhoto2** | Python，319★，2026-09 活跃 | 合成 Motion Photo v2/v3；--exif-match 配散对（Takeout/iCloud 导出场景） |
| mihir-io/MotionPhotoMuxer | 175★ | Apple Live → Google MicroVideo |
| doodspav/motionphoto | pip 包，32★ | Google+三星合成；**三星 SEF 逆向最佳文档** |
| LengxiQwQ/live-photo-box | 60★，2026-11 极活跃 | Windows 跨品牌工具箱（合并/拆分/转换/修 HDR） |
| duruonanni/motion-photo-bridge | PyPI 2026 | vivo/OPPO 双文件→Motion Photo+Immich 联动（含 yuvj420p 黑帧修复） |
| LimitPoint/LivePhoto | Swift，478★ | iOS 端分解/合成 |
| mindeng/nom-exif | Rust v3.1.0 | Motion Photo 探测（三种 XMP 布局） |

**Immich 实现要点**（server/src/services/metadata.service.ts）：Apple 按 ContentIdentifier/MediaGroupUUID 配对建 hidden 视频资产；Motion Photo 判定条件 = MotionPhoto / MicroVideo+Offset / MotionPhotoVideo（三星 HEIC）/ EmbeddedVideoType=MotionPhoto_Data（三星 JPEG SEF）/ ContainerDirectory；position = 文件大小 - Length - Padding 切视频。已知坑：S24 FE 部分 SEF 无标记；vivo 双文件不识别。

## 三、最佳帧选择（类 Top Shot）

- **Google Top Shot**：±1.5s 最多 90 帧，端上 MobileNet-SSD 人脸属性（睁眼/微笑）+ 光流运动显著性 + 陀螺仪/OIS 全局模糊 + 3A 状态 → GAM 融合打分，留 2 张备选跑 HDR+
- **学术**：Ren et al.《Best Frame Selection in a Short Video》WACV 2020（11,000 短视频标注 best frame，CNN+ranking）
- 现状：无成熟开源"live 图最佳帧"库，需自研；组件齐全（Laplacian/直方图/人脸眼睛 ONNX/光流/距封面时间衰减）
- 防抖：ffmpeg vidstab；Apple MOV 已含稳定，勿二次
- 衍生：Loop/Bounce；**长曝光**（多帧对齐混合，live-photo-conv 有参考实现）
- **换封面帧必须写回元数据**（Apple 改 LivePhotoVideoIndex/still-image-time；Google 改 PresentationTimestampUs）——小红书/抖音/OPPO 都支持"封面编辑"，这是核心卖点

## 四、社交平台发布链路

- 小红书：iPhone 早已支持；安卓机型白名单适配（OPPO Reno12 首发→vivo/荣耀跟进）；左上"实况"标识
- 微信朋友圈：iOS 8.0.51（2024-09）正式支持；安卓 8.0.58（2025-04）上线；聊天收发 2025-09 起内测推进（按厂商白名单放行）
- 抖音：OPPO/荣耀机型 2024-10/11 起支持发布 LivePhoto（抖音 32.0.0+）
- **结论：三平台统一走"从系统相册选取原生实况"链路，无发布 API——工具应输出相册可识别的原生 Live Photo，让用户直接发布**

## 五、PhotoRank 技术方案建议

1. **解析层**：统一 `LivePhotoAsset` 抽象。探测优先级：Apple 双文件（ContentIdentifier，次选同名 stem）→ Android MotionPhoto 1.0（XMP Container:Directory 自解析）→ 三星 SEF（exiftool 二进制标签）→ vivo/OPPO 同名双文件 → ftyp 扫描兜底。字段：still_file/video_file/cover_pts/duration/vendor
2. **抽帧层**：复用现有 cv2 grab/retrieve 管线；封面 PTS ±0.5s 加密采样；MOV 全范围 yuvj420p 在 Windows 硬解黑帧 → 统一转 yuv420p 或软解
3. **评分层**：L1/L2/L4 快筛照旧跑抽帧；新增 motion 维度（帧间光流显著性、闭眼/笑容，本地 ONNX）；**最佳帧 = argmax(L4 分 + motion 加分 − ε·|t−t_cover|)**（贴近封面优先）；输出 Top-K 备选
4. **输出层**：iOS 用户给 HEIC+MOV 对（保持同一 ContentIdentifier）；安卓给 Motion Photo JPEG（尾附 MP4+XMP，文件名 MP 结尾）；**换封面帧同步改元数据**；兜底 GIF+H.264 MP4；可选一键双格式兼容包、长曝光/Loop 衍生
5. **依赖**：exiftool-vendored + 自研 XMP Container 解析 + FFmpeg（抽帧/转码/vidstab）+ 现有 cv2 管线；全程离线

## 六、主要来源

- Android Motion Photo format 1.0：developer.android.com/media/platform/motion-photo-format
- exiftool.org/TagNames/Apple.html；exiftool/exiftool#254（三星标签）
- Immich：immich-app/immich metadata.service.ts；Discussion #32598（vivo）；Issue #30654（三星）
- Google Top Shot：research.google/blog/top-shot-on-pixel-3；Best Frame WACV 2020（openaccess.thecvf.com）
- 微信/小红书/抖音适配：ithome 系列报道（2024-05 ~ 2025-11）
- 格式解析：wszqkzqk.github.io《解析Android的动态照片》、doodspav/motionphoto README、LLLLLayer/Live-Photos
