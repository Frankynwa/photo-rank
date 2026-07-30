"""PhotoRank Web UI — FastAPI + 原生 HTML/JS"""
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
# ─────────────────────────────────────────────────────────
# 统一从 config.py 导入配置，消除重复定义
# ─────────────────────────────────────────────────────────
from config import (
    PROJECT_ROOT,
    PHOTOS_DIR as UPLOADS_DIR,   # config.py 中命名为 PHOTOS_DIR
    OUTPUT_DIR,
    HOST,
    PORT,
)
from core.logger import setup_logging, get_logger

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
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif"}
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/bmp",
    "image/webp", "image/heic", "image/heif",
}


def secure_filename(filename: str) -> str:
    """防路径遍历：去除目录片段，返回安全文件名"""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    parts = [p for p in name.split("/") if p and p != ".."]
    safe = "_".join(parts) if parts else "unnamed"
    return safe


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
        if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
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
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
