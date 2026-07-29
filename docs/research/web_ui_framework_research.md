# Photo Gallery + AI Analysis Web UI - Framework Comparison

## Executive Summary

For a photo gallery with AI analysis, drag-drop upload, real-time progress, and platform selector, I recommend **FastAPI + Vanilla HTML/JS** for the following reasons:

1. **Already installed** - FastAPI 0.133.1 + uvicorn are ready
2. **Simplest deployment** - No build step, no Node.js required
3. **Best real-time support** - Native WebSocket support in FastAPI
4. **Perfect for AI integration** - Direct Python to AI model calls
5. **No framework overhead** - Fast loading, minimal dependencies

---

## Framework Comparison

### 1. FastAPI + Vanilla HTML/JS ⭐ RECOMMENDED

**Pros:**
- ✅ Zero build step - just run `python main.py`
- ✅ Native WebSocket for real-time AI progress
- ✅ Jinja2 templates for server-side rendering
- ✅ Static files serving built-in
- ✅ No JavaScript framework learning curve
- ✅ Perfect for Python developers
- ✅ Fastest development time for small-medium apps

**Cons:**
- ⚠️ Less interactive than React/Vue for complex UIs
- ⚠️ Manual DOM manipulation needed
- ⚠️ No component reuse system

**Best For:** Solo developers, AI/ML engineers, rapid prototyping

**Example Structure:**
```
photo-gallery/
├── main.py                  # FastAPI backend
├── static/
│   ├── css/style.css        # Styles
│   ├── js/app.js            # Frontend logic
│   └── uploads/             # Photo storage
└── templates/
    └── index.html           # Main page
```

**Code Example:**

```python
# main.py - FastAPI Backend
from fastapi import FastAPI, UploadFile, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import asyncio
import json

app = FastAPI(title="Photo Gallery with AI Analysis")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Photo storage
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory photo database (use SQLite/PostgreSQL in production)
photos_db = {}

@app.get("/")
async def home(request: Request):
    """Main gallery page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "photos": photos_db
    })

@app.post("/api/upload")
async def upload_photo(file: UploadFile):
    """Upload single photo"""
    file_path = UPLOAD_DIR / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    photo_id = len(photos_db) + 1
    photos_db[photo_id] = {
        "id": photo_id,
        "filename": file.filename,
        "path": f"/static/uploads/{file.filename}",
        "scores": {},
        "analysis": None
    }
    
    return {"id": photo_id, "filename": file.filename}

@app.post("/api/analyze/{photo_id}")
async def analyze_photo(photo_id: int):
    """Start AI analysis for a photo"""
    if photo_id not in photos_db:
        return JSONResponse(status_code=404, content={"error": "Photo not found"})
    
    # Simulate AI analysis (replace with real AI call)
    photo = photos_db[photo_id]
    photo["analysis"] = {
        "aesthetic_score": 8.5,
        "composition": "Good rule of thirds",
        "lighting": "Natural light, well balanced",
        "platforms": {
            "小红书": {"score": 9.0, "reason": "Perfect for lifestyle content"},
            "朋友圈": {"score": 8.0, "reason": "Good for personal sharing"},
            "抖音": {"score": 7.5, "reason": "May need more dynamic angle"}
        }
    }
    
    return photo["analysis"]

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket for real-time analysis progress"""
    await websocket.accept()
    
    try:
        while True:
            # Receive analysis request
            data = await websocket.receive_text()
            request = json.loads(data)
            photo_id = request.get("photo_id")
            
            # Send progress updates
            await websocket.send_json({"status": "analyzing", "progress": 0})
            await asyncio.sleep(0.5)
            
            await websocket.send_json({"status": "analyzing", "progress": 30})
            await asyncio.sleep(0.5)
            
            await websocket.send_json({"status": "analyzing", "progress": 60})
            await asyncio.sleep(0.5)
            
            # Send final result
            await websocket.send_json({
                "status": "complete",
                "progress": 100,
                "result": photos_db.get(photo_id, {}).get("analysis")
            })
            
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/api/photos")
async def list_photos():
    """Get all photos"""
    return list(photos_db.values())

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: int):
    """Delete a photo"""
    if photo_id in photos_db:
        photo = photos_db.pop(photo_id)
        file_path = UPLOAD_DIR / photo["filename"]
        if file_path.exists():
            file_path.unlink()
        return {"status": "deleted"}
    return JSONResponse(status_code=404, content={"error": "Photo not found"})
```

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Photo Gallery</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #2c3e50;
        }
        
        /* Upload Area */
        .upload-area {
            border: 3px dashed #3498db;
            border-radius: 20px;
            padding: 60px;
            text-align: center;
            background: white;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover,
        .upload-area.dragover {
            background: #e8f4fc;
            border-color: #2980b9;
        }
        
        .upload-area p {
            font-size: 18px;
            color: #666;
        }
        
        .upload-area .icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        /* Platform Selector */
        .platform-selector {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 30px;
        }
        
        .platform-btn {
            padding: 12px 24px;
            border: 2px solid #ddd;
            border-radius: 25px;
            background: white;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .platform-btn:hover,
        .platform-btn.active {
            border-color: #3498db;
            background: #3498db;
            color: white;
        }
        
        /* Photo Grid */
        .photo-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }
        
        .photo-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .photo-card:hover {
            transform: translateY(-5px);
        }
        
        .photo-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        
        .photo-info {
            padding: 15px;
        }
        
        .photo-score {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
        }
        
        .score-badge {
            background: #27ae60;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .analyze-btn {
            width: 100%;
            padding: 10px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        
        .analyze-btn:hover {
            background: #2980b9;
        }
        
        .analyze-btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
        }
        
        /* Progress Bar */
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
            display: none;
        }
        
        .progress-bar.active {
            display: block;
        }
        
        .progress-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 4px;
            transition: width 0.3s ease;
            width: 0%;
        }
        
        /* Modal for Photo Details */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        
        .modal.active {
            display: flex;
        }
        
        .modal-content {
            background: white;
            border-radius: 20px;
            max-width: 800px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            padding: 30px;
        }
        
        .modal-close {
            float: right;
            font-size: 24px;
            cursor: pointer;
            color: #666;
        }
        
        /* Export Button */
        .export-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 15px 30px;
            background: #27ae60;
            color: white;
            border: none;
            border-radius: 30px;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(39,174,96,0.4);
            transition: all 0.3s ease;
        }
        
        .export-btn:hover {
            background: #229954;
            transform: translateY(-2px);
        }
        
        /* Checkbox for selection */
        .photo-select {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 24px;
            height: 24px;
            cursor: pointer;
        }
        
        .photo-card {
            position: relative;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 AI Photo Gallery</h1>
        
        <!-- Platform Selector -->
        <div class="platform-selector">
            <button class="platform-btn active" data-platform="all">全部</button>
            <button class="platform-btn" data-platform="小红书">小红书</button>
            <button class="platform-btn" data-platform="朋友圈">朋友圈</button>
            <button class="platform-btn" data-platform="抖音">抖音</button>
        </div>
        
        <!-- Upload Area -->
        <div class="upload-area" id="dropZone">
            <div class="icon">📁</div>
            <p>拖拽照片到这里，或点击选择文件</p>
            <input type="file" id="fileInput" multiple accept="image/*" style="display: none">
        </div>
        
        <!-- Photo Grid -->
        <div class="photo-grid" id="photoGrid">
            <!-- Photos will be loaded here -->
        </div>
    </div>
    
    <!-- Export Button -->
    <button class="export-btn" id="exportBtn" style="display: none">
        📥 导出选中照片 (0)
    </button>
    
    <!-- Photo Detail Modal -->
    <div class="modal" id="photoModal">
        <div class="modal-content">
            <span class="modal-close" id="closeModal">&times;</span>
            <div id="modalBody">
                <!-- Photo details will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        // State management
        const state = {
            photos: [],
            selectedPhotos: new Set(),
            currentPlatform: 'all',
            ws: null
        };
        
        // DOM Elements
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const photoGrid = document.getElementById('photoGrid');
        const exportBtn = document.getElementById('exportBtn');
        const modal = document.getElementById('photoModal');
        const modalBody = document.getElementById('modalBody');
        
        // Initialize WebSocket for real-time updates
        function initWebSocket() {
            state.ws = new WebSocket(`ws://${window.location.host}/ws/progress`);
            
            state.ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateProgress(data);
            };
            
            state.ws.onclose = function() {
                setTimeout(initWebSocket, 3000); // Reconnect
            };
        }
        
        // Drag and Drop handlers
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });
        
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
        
        // Upload files
        async function handleFiles(files) {
            for (const file of files) {
                if (!file.type.startsWith('image/')) continue;
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const photo = await response.json();
                        state.photos.push(photo);
                        renderPhoto(photo);
                    }
                } catch (error) {
                    console.error('Upload failed:', error);
                }
            }
        }
        
        // Render photo card
        function renderPhoto(photo) {
            const card = document.createElement('div');
            card.className = 'photo-card';
            card.dataset.id = photo.id;
            
            card.innerHTML = `
                <input type="checkbox" class="photo-select" 
                       onchange="toggleSelect(${photo.id})">
                <img src="${photo.path}" alt="${photo.filename}"
                     onclick="showDetail(${photo.id})">
                <div class="photo-info">
                    <h3>${photo.filename}</h3>
                    <div class="photo-score" id="score-${photo.id}">
                        <span>未分析</span>
                    </div>
                    <button class="analyze-btn" onclick="analyzePhoto(${photo.id})">
                        🔍 AI 分析
                    </button>
                    <div class="progress-bar" id="progress-${photo.id}">
                        <div class="fill"></div>
                    </div>
                </div>
            `;
            
            photoGrid.appendChild(card);
        }
        
        // Analyze photo with AI
        async function analyzePhoto(photoId) {
            const btn = document.querySelector(`[data-id="${photoId}"] .analyze-btn`);
            const progressBar = document.getElementById(`progress-${photoId}`);
            
            btn.disabled = true;
            btn.textContent = '分析中...';
            progressBar.classList.add('active');
            
            // Send via WebSocket for real-time progress
            state.ws.send(JSON.stringify({ photo_id: photoId }));
        }
        
        // Update progress bar
        function updateProgress(data) {
            const photoId = data.photo_id || state.currentAnalyzing;
            const progressBar = document.getElementById(`progress-${photoId}`);
            
            if (!progressBar) return;
            
            const fill = progressBar.querySelector('.fill');
            fill.style.width = `${data.progress}%`;
            
            if (data.status === 'complete') {
                setTimeout(() => {
                    progressBar.classList.remove('active');
                    updatePhotoScore(photoId, data.result);
                }, 500);
            }
        }
        
        // Update photo score display
        function updatePhotoScore(photoId, analysis) {
            const scoreDiv = document.getElementById(`score-${photoId}`);
            const platform = state.currentPlatform;
            
            let score = analysis.aesthetic_score;
            let reason = analysis.composition;
            
            if (platform !== 'all' && analysis.platforms[platform]) {
                score = analysis.platforms[platform].score;
                reason = analysis.platforms[platform].reason;
            }
            
            scoreDiv.innerHTML = `
                <span class="score-badge">${score.toFixed(1)}</span>
                <span>${reason}</span>
            `;
            
            const btn = document.querySelector(`[data-id="${photoId}"] .analyze-btn`);
            btn.disabled = false;
            btn.textContent = '🔍 重新分析';
        }
        
        // Toggle photo selection
        function toggleSelect(photoId) {
            if (state.selectedPhotos.has(photoId)) {
                state.selectedPhotos.delete(photoId);
            } else {
                state.selectedPhotos.add(photoId);
            }
            
            exportBtn.style.display = state.selectedPhotos.size > 0 ? 'block' : 'none';
            exportBtn.textContent = `📥 导出选中照片 (${state.selectedPhotos.size})`;
        }
        
        // Show photo detail modal
        async function showDetail(photoId) {
            const photo = state.photos.find(p => p.id === photoId);
            if (!photo) return;
            
            let analysisHtml = '<p>未分析</p>';
            if (photo.analysis) {
                const a = photo.analysis;
                analysisHtml = `
                    <h3>AI 分析结果</h3>
                    <p><strong>美学评分:</strong> ${a.aesthetic_score}/10</p>
                    <p><strong>构图:</strong> ${a.composition}</p>
                    <p><strong>光线:</strong> ${a.lighting}</p>
                    <h4>平台适配度</h4>
                    <ul>
                        <li>小红书: ${a.platforms['小红书'].score}/10 - ${a.platforms['小红书'].reason}</li>
                        <li>朋友圈: ${a.platforms['朋友圈'].score}/10 - ${a.platforms['朋友圈'].reason}</li>
                        <li>抖音: ${a.platforms['抖音'].score}/10 - ${a.platforms['抖音'].reason}</li>
                    </ul>
                `;
            }
            
            modalBody.innerHTML = `
                <h2>${photo.filename}</h2>
                <img src="${photo.path}" style="width: 100%; border-radius: 10px; margin: 20px 0;">
                ${analysisHtml}
            `;
            
            modal.classList.add('active');
        }
        
        // Close modal
        document.getElementById('closeModal').onclick = () => {
            modal.classList.remove('active');
        };
        
        // Platform selector
        document.querySelectorAll('.platform-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.platform-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                state.currentPlatform = e.target.dataset.platform;
                // Re-render photos with platform-specific scores
                renderAllPhotos();
            });
        });
        
        // Export selected photos
        exportBtn.addEventListener('click', async () => {
            const selected = Array.from(state.selectedPhotos);
            // Implement export logic (zip download, copy to folder, etc.)
            alert(`导出 ${selected.length} 张照片`);
        });
        
        // Load existing photos
        async function loadPhotos() {
            try {
                const response = await fetch('/api/photos');
                const photos = await response.json();
                state.photos = photos;
                photos.forEach(renderPhoto);
            } catch (error) {
                console.error('Failed to load photos:', error);
            }
        }
        
        // Initialize
        initWebSocket();
        loadPhotos();
    </script>
</body>
</html>
```

---

### 2. FastAPI + React (More Features)

**Pros:**
- ✅ Component-based architecture
- ✅ Rich ecosystem (React Query, Zustand, etc.)
- ✅ Excellent for complex UIs
- ✅ Hot module replacement in dev
- ✅ Large community and resources

**Cons:**
- ❌ Requires Node.js and npm/yarn
- ❌ Build step adds complexity
- ❌ Steeper learning curve
- ❌ More boilerplate code
- ❌ Overkill for simple galleries

**Best For:** Large teams, complex UI requirements, long-term maintenance

**Structure:**
```
photo-gallery/
├── backend/
│   └── main.py              # FastAPI
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── PhotoGrid.tsx
│   │   │   ├── UploadZone.tsx
│   │   │   └── AnalysisPanel.tsx
│   │   └── hooks/
│   │       └── useWebSocket.ts
│   └── public/
└── docker-compose.yml
```

**Key Dependencies:**
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-dropzone": "^14.2.0",
    "@tanstack/react-query": "^5.0.0",
    "framer-motion": "^10.0.0"
  }
}
```

**Code Example (React component):**
```tsx
// UploadZone.tsx
import { useDropzone } from 'react-dropzone';
import { useMutation, useQueryClient } from '@tanstack/react-query';

export function UploadZone() {
  const queryClient = useQueryClient();
  
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['photos'] });
    }
  });
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/*': [] },
    onDrop: (files) => files.forEach(f => uploadMutation.mutate(f))
  });
  
  return (
    <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
      <input {...getInputProps()} />
      <p>拖拽照片到这里</p>
    </div>
  );
}
```

---

### 3. FastAPI + Vue.js (Balance)

**Pros:**
- ✅ Easier learning curve than React
- ✅ Built-in state management (Pinia)
- ✅ Single File Components (SFC)
- ✅ Good TypeScript support
- ✅ Growing ecosystem

**Cons:**
- ❌ Requires Node.js
- ❌ Smaller community than React
- ❌ Fewer third-party components

**Best For:** Solo developers wanting more structure, Vue enthusiasts

**Code Example:**
```vue
<!-- PhotoCard.vue -->
<template>
  <div class="photo-card" :class="{ selected }">
    <img :src="photo.path" @click="$emit('detail', photo.id)">
    <div class="score" v-if="photo.analysis">
      {{ photo.analysis.aesthetic_score }}
    </div>
    <button @click="$emit('analyze', photo.id)" :disabled="analyzing">
      {{ analyzing ? '分析中...' : 'AI 分析' }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const props = defineProps<{
  photo: Photo;
  selected?: boolean;
}>();

const emit = defineEmits<{
  analyze: [id: number];
  detail: [id: number];
}>();

const analyzing = ref(false);
</script>
```

---

### 4. Streamlit (Python-Only)

**Pros:**
- ✅ 100% Python - no HTML/JS/CSS
- ✅ Fastest prototyping
- ✅ Built-in widgets
- ✅ Auto-refresh on code changes
- ✅ Great for data apps

**Cons:**
- ❌ Limited customization
- ❌ Reruns entire script on interaction
- ❌ No real WebSocket support (uses polling)
- ❌ Performance issues with large galleries
- ❌ Not production-ready for complex UIs

**Best For:** Quick demos, internal tools, data exploration

**Code Example:**
```python
# app.py - Streamlit Photo Gallery
import streamlit as st
from PIL import Image
import asyncio

st.title("📸 AI Photo Gallery")

# Platform selector
platform = st.selectbox("选择平台", ["全部", "小红书", "朋友圈", "抖音"])

# File uploader
uploaded_files = st.file_uploader(
    "上传照片",
    type=['jpg', 'jpeg', 'png', 'webp'],
    accept_multiple_files=True
)

if uploaded_files:
    cols = st.columns(3)
    
    for idx, file in enumerate(uploaded_files):
        with cols[idx % 3]:
            image = Image.open(file)
            st.image(image, caption=file.name)
            
            if st.button(f"AI 分析", key=f"analyze_{idx}"):
                with st.spinner("分析中..."):
                    # AI analysis here
                    score = 8.5
                    st.metric("美学评分", f"{score}/10")
                    
                    st.write("**平台适配度:**")
                    if platform == "小红书" or platform == "全部":
                        st.progress(0.9, text="小红书: 9.0/10")
                    if platform == "朋友圈" or platform == "全部":
                        st.progress(0.8, text="朋友圈: 8.0/10")
```

**Install & Run:**
```bash
pip install streamlit
streamlit run app.py
```

---

### 5. Gradio (AI-Focused)

**Pros:**
- ✅ Built for ML/AI demos
- ✅ Auto-generates UI from Python functions
- ✅ Built-in image handling
- ✅ Shareable via public links
- ✅ HuggingFace integration

**Cons:**
- ❌ Very limited customization
- ❌ Not suitable for complex UIs
- ❌ No drag-drop upload (file picker only)
- ❌ Limited layout options
- ❌ Not for production use

**Best For:** Quick AI model demos, HuggingFace Spaces

**Code Example:**
```python
# app.py - Gradio Photo Analyzer
import gradio as gr

def analyze_photo(image, platform):
    """AI analysis function"""
    # Simulated analysis
    scores = {
        "小红书": 9.0,
        "朋友圈": 8.0,
        "抖音": 7.5
    }
    
    score = scores.get(platform, 8.5)
    
    return {
        "美学评分": score,
        "构图": "三分法构图，主体突出",
        "光线": "自然光，柔和均匀"
    }

# Build UI
with gr.Blocks(title="AI Photo Analyzer") as demo:
    gr.Markdown("# 📸 AI Photo Analyzer")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传照片")
            platform = gr.Radio(
                ["小红书", "朋友圈", "抖音"],
                label="目标平台",
                value="小红书"
            )
            analyze_btn = gr.Button("🔍 AI 分析")
        
        with gr.Column():
            output = gr.JSON(label="分析结果")
    
    analyze_btn.click(
        analyze_photo,
        inputs=[image_input, platform],
        outputs=output
    )

demo.launch()
```

---

## Feature Comparison Matrix

| Feature | FastAPI+HTML/JS | FastAPI+React | FastAPI+Vue | Streamlit | Gradio |
|---------|----------------|---------------|-------------|-----------|--------|
| **Drag-Drop Upload** | ✅ Native | ✅ react-dropzone | ✅ vue-dropzone | ❌ | ❌ |
| **Real-time Progress** | ✅ WebSocket | ✅ WebSocket | ✅ WebSocket | ⚠️ Polling | ❌ |
| **Photo Gallery Grid** | ✅ CSS Grid | ✅ Masonry | ✅ Grid | ⚠️ Basic | ❌ |
| **Platform Selector** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Photo Detail Popup** | ✅ Modal | ✅ Modal | ✅ Modal | ⚠️ Expander | ❌ |
| **Export Selected** | ✅ | ✅ | ✅ | ⚠️ | ❌ |
| **AI Integration** | ✅ Direct | ✅ API | ✅ API | ✅ Direct | ✅ Direct |
| **Learning Curve** | 🟢 Low | 🔴 High | 🟡 Medium | 🟢 Low | 🟢 Low |
| **Deployment** | 🟢 Easy | 🟡 Medium | 🟡 Medium | 🟢 Easy | 🟢 Easy |
| **Customization** | 🟢 Full | 🟢 Full | 🟢 Full | 🔴 Limited | 🔴 Limited |
| **Production Ready** | ✅ | ✅ | ✅ | ⚠️ | ❌ |

---

## Recommendation: FastAPI + Vanilla HTML/JS

### Why This is Best for Your Use Case:

1. **Already Installed** - No additional dependencies needed
2. **Perfect Feature Match** - Supports all 5 requirements natively
3. **Fastest Development** - Can be built in 1-2 days
4. **AI-Friendly** - Direct Python-to-model calls without API layer
5. **Real-time Native** - WebSocket built into FastAPI
6. **No Build Step** - Edit and refresh, no compilation
7. **Easy Deployment** - Single `uvicorn main:app` command

### Quick Start Commands:

```bash
# Create project structure
mkdir -p photo-gallery/static/uploads photo-gallery/templates

# Save the code examples above to:
# - photo-gallery/main.py
# - photo-gallery/templates/index.html

# Run the server
cd photo-gallery
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Open in browser
# http://localhost:8000
```

---

## When to Choose Each Framework:

- **FastAPI + HTML/JS**: Solo developer, rapid prototype, AI/ML focus ⭐
- **FastAPI + React**: Large team, complex UI, long-term project
- **FastAPI + Vue**: Want React benefits with easier learning curve
- **Streamlit**: Internal demo, data exploration, quick prototype
- **Gradio**: HuggingFace Space, simple model demo, not for production
