"""PhotoRank Web UI — FastAPI + 原生 HTML/JS"""
import os
import json
import shutil
import asyncio
import hashlib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

# ─────────────────────────────────────────────────────────
# [修复 #1] 统一从 config.py 导入配置，消除重复定义
# ─────────────────────────────────────────────────────────
from config import (
    PROJECT_ROOT,
    PHOTOS_DIR as UPLOADS_DIR,   # config.py 中命名为 PHOTOS_DIR
    OUTPUT_DIR,
    HOST,
    PORT,
)

app = FastAPI(title="PhotoRank")

# 使用 config.py 中的路径，确保目录存在
TEMPLATES_DIR = PROJECT_ROOT / "templates"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────
# [修复 #2] 文件上传安全校验相关常量
# ─────────────────────────────────────────────────────────

# 最大上传文件大小：50 MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# 允许的图片扩展名
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif"}

# 允许的图片 MIME 类型
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif",
    "image/bmp", "image/webp", "image/heic", "image/heif",
}


def secure_filename(filename: str) -> str:
    """
    [修复 #2] 安全化文件名，防止路径遍历攻击。
    自行实现以避免引入 werkzeug 依赖。
    - 只取最后一级路径分量（去除 / 和 \\ 前缀）
    - 去除 '..' 等危险路径片段
    """
    # 取文件名部分（兼容 Windows 反斜杠和 Unix 正斜杠）
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    # 去除所有 '..' 组件
    parts = [p for p in name.split("/") if p and p != ".."]
    safe = "_".join(parts) if parts else "unnamed"
    return safe


# WebSocket 连接管理
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
                # [修复 #3] 裸 except → 捕获 Exception 并记录日志
                print(f"[WebSocket] 广播消息失败: {e}")

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
    """上传照片"""
    uploaded = []
    for file in files:
        if not file.filename:
            continue

        # ── [修复 #2] 安全校验 ──────────────────────────

        # 1) 文件名安全化：防止路径遍历攻击（如 ../../etc/passwd）
        safe_name = secure_filename(file.filename)

        # 2) 扩展名校验：只允许常见图片格式
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {ext}，仅允许 {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        # 3) MIME 类型校验
        if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的 MIME 类型: {file.content_type}",
            )

        # 4) 读取内容并校验文件大小
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大，最大允许 {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
            )

        # 生成唯一文件名（基于内容哈希，避免重名覆盖）
        filename = hashlib.md5(content).hexdigest()[:8] + ext

        # 保存文件
        filepath = UPLOADS_DIR / filename
        with open(filepath, "wb") as f:
            f.write(content)

        uploaded.append({
            "filename": filename,
            "original_name": file.filename,
            "size": len(content),
        })

    # 广播上传完成
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
    """获取照片"""
    # [修复 #2] 对路径参数也做安全校验，防止目录遍历
    safe_name = secure_filename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="非法文件名")

    filepath = UPLOADS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="照片不存在")
    return FileResponse(filepath)


@app.delete("/api/photos/{filename}")
async def delete_photo(filename: str):
    """删除照片"""
    # [修复 #2] 对路径参数做安全校验
    safe_name = secure_filename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="非法文件名")

    filepath = UPLOADS_DIR / filename
    if filepath.exists():
        filepath.unlink()
    return {"deleted": filename}


@app.get("/api/results/{platform}")
async def get_results(platform: str):
    """获取分析结果"""
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
    """获取结果照片"""
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
    # 广播开始分析
    await manager.broadcast({"type": "analysis_start", "platform": platform})

    # 运行流水线
    from core.pipeline import run_pipeline

    try:
        # [修复 #5] 使用 asyncio.to_thread 将同步的 run_pipeline 放到线程池执行，
        # 避免阻塞 asyncio 事件循环（Python 3.9+）
        result = await asyncio.to_thread(run_pipeline, platform=platform)

        # 广播分析完成
        await manager.broadcast({
            "type": "analysis_complete",
            "platform": platform,
            "result": {
                "total": result.get("total", 0),
                "rejected": result.get("rejected", 0),
                "output_count": result.get("output_count", 0),
            },
        })

        return result
    except Exception as e:
        # [修复 #4] 详细错误只记录到服务端日志，不返回给客户端
        print(f"[Analyze] 分析流水线异常: {e}")
        await manager.broadcast({
            "type": "analysis_error",
            "error": "分析过程出现内部错误，请稍后重试",
        })
        raise HTTPException(status_code=500, detail="服务器内部错误，请稍后重试")


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
    # [修复 #1] 端口和地址统一使用 config.py 中的配置
    uvicorn.run(app, host=HOST, port=PORT)
