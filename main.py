"""PhotoRank Web UI — FastAPI + 原生 HTML/JS"""
import asyncio
import json
import re
from pathlib import Path

import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse

# ─────────────────────────────────────────────────────────
# 统一从 config.py 导入配置，消除重复定义
# ─────────────────────────────────────────────────────────
from config import (
    HOST,
    OUTPUT_DIR,
    PORT,
    PROJECT_ROOT,
    VIDEO_EXTS,
    VIDEOS_DIR,
)
from config import (
    PHOTOS_DIR as UPLOADS_DIR,  # config.py 中命名为 PHOTOS_DIR
)
from core.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="PhotoRank")

# 并发控制
_analysis_lock = asyncio.Lock()

# config.py 未定义 TEMPLATES_DIR，仅本地定义
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# ─────────────────────────────────────────────────────────
# P0 安全常量 — 不可删除
# ─────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
MAX_VIDEO_UPLOAD_SIZE = 200 * 1024 * 1024  # 视频上限 200MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif"}
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/bmp",
    "image/webp", "image/heic", "image/heif",
}
VIDEO_MIME_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/webm",
}


def secure_filename(filename: str) -> str:
    """防路径遍历 + 净化 Windows 非法字符，返回安全文件名"""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    parts = [p for p in name.split("/") if p and p != ".."]
    safe = "_".join(parts) if parts else "unnamed"
    # Windows 保留字符：避免保存时 OSError（项目运行在 Windows）
    safe = re.sub(r'[\\/:*?"<>|]', "_", safe).strip()
    return safe or "unnamed"


UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# WebSocket 连接
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"[WebSocket] 广播消息失败: {e}")

manager = ConnectionManager()

# ─────────────────────────────────────────────────────────
# API 路由
# ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>PhotoRank</h1><p>请上传照片开始分析</p>")


@app.post("/api/upload")
async def upload_photos(files: list[UploadFile] = File(...)):
    """上传照片（含安全校验）"""
    uploaded = []
    for file in files:
        if not file.filename:
            continue

        safe_name = secure_filename(file.filename)
        ext = Path(safe_name).suffix.lower()

        # 扩展名白名单
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

        # MIME 类型白名单
        # 注意：浏览器上传 HEIC/HEIF 时 Content-Type 通常为 application/octet-stream
        #（浏览器不识别该格式的通病），扩展名已过白名单，故放行 octet-stream，
        # 文件真实性由下游 PIL 解码验证。
        if (
            file.content_type
            and file.content_type not in ALLOWED_MIME_TYPES
            and file.content_type != "application/octet-stream"
        ):
            raise HTTPException(status_code=400, detail=f"不支持的 MIME 类型: {file.content_type}")

        content = await file.read()

        # 文件大小限制
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"文件过大，最大允许 {MAX_UPLOAD_SIZE // (1024 * 1024)} MB")

        # 用内容哈希生成唯一文件名
        import hashlib
        filename = hashlib.md5(content).hexdigest()[:8] + ext

        filepath = UPLOADS_DIR / filename
        with open(filepath, "wb") as f:
            f.write(content)

        uploaded.append({
            "filename": filename,
            "original_name": file.filename,
            "size": len(content),
        })

    await manager.broadcast({
        "type": "upload_complete",
        "count": len(uploaded),
    })

    return {"uploaded": uploaded, "total": len(uploaded)}


@app.post("/api/upload_video")
async def upload_video(request: Request, file: UploadFile = File(...)):
    """上传视频并流式抽帧：本地快筛（L1/L2/L4）后，合格帧写入 uploads 顶层"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    safe_name = secure_filename(file.filename)
    ext = Path(safe_name).suffix.lower()

    if ext not in VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的视频类型: {ext}")

    # MIME 类型白名单（与照片接口行为一致）
    if file.content_type and file.content_type not in VIDEO_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的 MIME 类型: {file.content_type}")

    # Content-Length 预检：超限直接拒绝，不读取请求体（防内存 DoS）
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() \
            and int(content_length) > MAX_VIDEO_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"视频过大，最大允许 {MAX_VIDEO_UPLOAD_SIZE // (1024 * 1024)} MB",
        )

    # 保存视频（分块写盘，累计超限即中断并清理半成品）
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    video_path = VIDEOS_DIR / safe_name
    try:
        written = 0
        with open(video_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_VIDEO_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"视频过大，最大允许 {MAX_VIDEO_UPLOAD_SIZE // (1024 * 1024)} MB",
                    )
                f.write(chunk)
    except HTTPException:
        video_path.unlink(missing_ok=True)
        raise
    except OSError as e:
        video_path.unlink(missing_ok=True)
        logger.error(f"[视频保存] 写入失败 {safe_name}: {e}")
        raise HTTPException(status_code=500, detail=f"视频保存失败: {e}")

    # 流式抽帧 + 本地快筛（耗 CPU，放线程池避免阻塞事件循环）
    from core.video_frames import VideoExtractError, extract_quality_frames

    try:
        stats = await asyncio.to_thread(
            extract_quality_frames, video_path, UPLOADS_DIR,
        )
    except VideoExtractError as e:
        video_path.unlink(missing_ok=True)  # 失败不残留垃圾视频文件
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        video_path.unlink(missing_ok=True)
        logger.error(f"[视频抽帧] 失败 {safe_name}: {e}")
        raise HTTPException(status_code=500, detail=f"视频抽帧失败: {e}")

    await manager.broadcast({
        "type": "video_frames_ready",
        "video": safe_name,
        "stats": {k: v for k, v in stats.items() if k != "saved"},
    })

    return {
        "video": safe_name,
        "stats": stats,
        "message": f"抽帧完成：候选 {stats['candidates']} → 保留 {stats['kept']} 帧",
    }


@app.get("/api/photos")
async def list_photos():
    """列出所有照片"""
    photos = []
    for f in UPLOADS_DIR.iterdir():
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.heic', '.heif'):
            photos.append({
                "filename": f.name,
                "path": str(f),
                "size": f.stat().st_size,
            })
    return {"photos": photos, "total": len(photos)}


@app.get("/api/photos/{filename}")
async def get_photo(filename: str):
    """获取照片（含路径遍历防护）"""
    safe_name = secure_filename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    filepath = UPLOADS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="照片不存在")
    return FileResponse(filepath)


@app.delete("/api/photos/{filename}")
async def delete_photo(filename: str):
    """删除照片（含路径遍历防护）"""
    safe_name = secure_filename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    filepath = UPLOADS_DIR / filename
    if filepath.exists():
        filepath.unlink()
    return {"deleted": filename}


@app.get("/api/results/{platform}")
async def get_results(platform: str):
    """获取分析结果（含路径遍历防护）"""
    safe_platform = secure_filename(platform)
    if safe_platform != platform:
        raise HTTPException(status_code=400, detail="非法平台名")
    ranking_file = OUTPUT_DIR / platform / "ranking.json"
    if not ranking_file.exists():
        raise HTTPException(status_code=404, detail="未找到分析结果")

    with open(ranking_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {"results": data, "total": len(data)}


@app.get("/api/results/{platform}/{filename}")
async def get_result_photo(platform: str, filename: str):
    """获取结果照片（含路径遍历防护）"""
    safe_platform = secure_filename(platform)
    if safe_platform != platform:
        raise HTTPException(status_code=400, detail="非法平台名")
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    filepath = OUTPUT_DIR / platform / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="照片不存在")
    return FileResponse(filepath)


@app.post("/api/analyze")
async def analyze_photos(platform: str = "xiaohongshu"):
    """运行分析"""
    if _analysis_lock.locked():
        raise HTTPException(status_code=409, detail="分析任务正在运行中")

    async with _analysis_lock:
        await manager.broadcast({"type": "analysis_start", "platform": platform})
        from core.pipeline import run_pipeline

        try:
            loop = asyncio.get_running_loop()

            def progress_callback(event: str, data: dict):
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast({"type": event, **data}),
                    loop,
                )

            result = await asyncio.to_thread(
                run_pipeline, platform=platform, progress_callback=progress_callback,
            )

            await manager.broadcast({
                "type": "analysis_complete", "platform": platform,
                "result": {
                    "total": result.get("total", 0),
                    "rejected": result.get("rejected", 0),
                    "output_count": result.get("output_count", 0),
                },
            })
            return result
        except Exception as e:
            logger.error(f"[Analyze] 分析流水线异常: {e}")
            await manager.broadcast({"type": "analysis_error", "error": "分析过程出现内部错误，请稍后重试"})
            raise HTTPException(status_code=500, detail="服务器内部错误")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 连接"""
    await manager.connect(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
