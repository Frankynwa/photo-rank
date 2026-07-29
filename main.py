"""PhotoRank Web UI — FastAPI + 原生 HTML/JS"""
import os
import json
import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

app = FastAPI(title="PhotoRank")

# 配置
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
TEMPLATES_DIR = BASE_DIR / "templates"

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
            except:
                pass

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
        
        # 生成唯一文件名
        import hashlib
        ext = Path(file.filename).suffix
        content = await file.read()
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
    filepath = UPLOADS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="照片不存在")
    return FileResponse(filepath)


@app.delete("/api/photos/{filename}")
async def delete_photo(filename: str):
    """删除照片"""
    filepath = UPLOADS_DIR / filename
    if filepath.exists():
        filepath.unlink()
    return {"deleted": filename}


@app.get("/api/results/{platform}")
async def get_results(platform: str):
    """获取分析结果"""
    ranking_file = OUTPUT_DIR / platform / "ranking.json"
    if not ranking_file.exists():
        raise HTTPException(status_code=404, detail="未找到分析结果")
    
    with open(ranking_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return {"results": data, "total": len(data)}


@app.get("/api/results/{platform}/{filename}")
async def get_result_photo(platform: str, filename: str):
    """获取结果照片"""
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
        result = run_pipeline(platform=platform)
        
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
        await manager.broadcast({
            "type": "analysis_error",
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail=str(e))


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
    uvicorn.run(app, host="127.0.0.1", port=8000)
