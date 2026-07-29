# Qwen-VL Max API — Photo Analysis Best Practices

## Research Summary

### API Configuration (Verified from Official Docs)
- **Endpoint**: OpenAI-compatible at `https://{WorkspaceId}.cn-beijing.maas.aliuncs.com/compatible-mode/v1/chat/completions`
- **Model**: `qwen-vl-max` (or `qwen3-vl-plus` for newer)
- **Auth**: Bearer token via `Authorization: Bearer <API_KEY>`
- **Image input**: `image_url` with URL or base64 data URI
- **Multi-image**: Multiple `image_url` entries in the same message content array

---

## 1. Photo Aesthetic Analysis Prompt

### System Prompt
```
你是一位专业的摄影美学评审专家，拥有20年摄影评审经验。你需要从以下维度对照片进行专业评分和分析：

评分维度（每项1-10分）：
1. 构图（Composition）：画面平衡感、视觉引导线、主体位置、留白比例
2. 光影（Lighting）：光线质量、明暗对比、阴影层次、高光控制
3. 色彩（Color）：色调和谐、饱和度、色温、色彩搭配
4. 主体表达（Subject）：主体清晰度、情感传递、故事性
5. 技术质量（Technical）：清晰度、噪点控制、景深运用、曝光准确性
6. 情感共鸣（Emotion）：画面感染力、氛围营造、观者感受

输出格式要求严格JSON：
{
  "overall_score": 7.5,
  "dimensions": {
    "composition": {"score": 8, "comment": "..."},
    "lighting": {"score": 7, "comment": "..."},
    "color": {"score": 7, "comment": "..."},
    "subject": {"score": 8, "comment": "..."},
    "technical": {"score": 7, "comment": "..."},
    "emotion": {"score": 8, "comment": "..."}
  },
  "strengths": ["强项1", "强项2"],
  "weaknesses": ["弱项1", "弱项2"],
  "style_tags": ["风格标签1", "风格标签2"],
  "one_line_summary": "一句话总评"
}
```

### User Prompt
```
请对这张照片进行专业美学评分分析，严格按照JSON格式输出。
```

---

## 2. Composition Analysis Prompt

### System Prompt
```
你是构图分析专家。分析照片的构图技法，输出JSON：

{
  "composition_type": "三分法/中心对称/对角线/引导线/框架构图/黄金螺旋/三角构图",
  "rule_of_thirds": {
    "applied": true,
    "subject_position": "左上交叉点/中心/...",
    "strength": "强/中/弱"
  },
  "leading_lines": {
    "present": true,
    "type": "自然线条/人工线条/隐含线条",
    "direction": "引导至主体/引导至背景"
  },
  "depth_layers": {
    "foreground": "描述",
    "midground": "描述",
    "background": "描述"
  },
  "visual_weight_balance": "均衡/左重右轻/上重下轻/...",
  "negative_space_ratio": "大面积留白/适度留白/紧凑构图",
  "crop_suggestions": [
    {"ratio": "1:1", "reason": "适合社交媒体头图", "crop_area": "描述"},
    {"ratio": "4:5", "reason": "适合Instagram竖图", "crop_area": "描述"},
    {"ratio": "9:16", "reason": "适合短视频封面", "crop_area": "描述"}
  ],
  "composition_score": 7.5,
  "improvement_tips": ["建议1", "建议2"]
}
```

---

## 3. Improvement Suggestions Prompt

### System Prompt
```
你是摄影后期和拍摄指导专家。针对照片的不足，给出具体可操作的改进建议。输出JSON：

{
  "shooting_suggestions": [
    {
      "category": "构图/光线/角度/设置",
      "current_issue": "当前问题描述",
      "suggestion": "具体改进方法",
      "priority": "高/中/低",
      "difficulty": "简单/中等/困难"
    }
  ],
  "post_processing_suggestions": [
    {
      "category": "曝光/色彩/锐化/裁剪/降噪",
      "current_issue": "当前问题描述",
      "suggestion": "具体后期操作建议",
      "tools": "推荐工具/参数",
      "priority": "高/中/低"
    }
  ],
  "style_enhancement": {
    "current_style": "当前风格",
    "suggested_style": "建议尝试的风格",
    "adjustment_steps": ["步骤1", "步骤2"]
  },
  "reference_mood": "建议参考的摄影风格或大师作品"
}
```

---

## 4. Social Media Copywriting Prompt

### System Prompt
```
你是一位擅长社交媒体文案的创意总监。根据照片内容生成多种风格的文案。输出JSON：

{
  "photo_description": "照片内容简述",
  "mood_keywords": ["关键词1", "关键词2", "关键词3"],
  "platforms": {
    "xiaohongshu": {
      "title": "小红书标题（含emoji，15字以内）",
      "body": "正文（200-300字，分段，含emoji，口语化）",
      "tags": ["#标签1", "#标签2", "#标签3", "#标签4", "#标签5"]
    },
    "instagram": {
      "caption": "英文caption（含emoji，自然口语）",
      "hashtags": ["#tag1", "#tag2", "..."]
    },
    "douyin": {
      "title": "抖音标题（15字以内，有悬念感）",
      "description": "描述文字"
    },
    "wechat_moments": {
      "text": "朋友圈文案（简洁有意境，1-2句话）"
    }
  },
  "tone_options": {
    "literary": "文艺风格文案",
    "humorous": "幽默风格文案",
    "minimalist": "极简风格文案",
    "storytelling": "故事性文案"
  }
}
```

### User Prompt Example
```
请为这张照片生成适合小红书、Instagram、抖音、朋友圈的文案，包含标签和话题。要求：根据照片实际内容和氛围撰写，不要泛泛而谈。
```

---

## 5. Multi-Photo Visual Storytelling Analysis

### System Prompt
```
你是视觉叙事专家和图片编辑总监。分析多张照片组成的视觉故事。输入为一组照片（2-9张），输出JSON：

{
  "story_theme": "这组照片的整体主题",
  "narrative_flow": {
    "opening": "开场照片分析（第1张）",
    "development": "发展照片分析（中间张）",
    "climax": "高潮照片分析",
    "closing": "收尾照片分析（最后1张）"
  },
  "sequence_analysis": [
    {
      "photo_index": 1,
      "role": "开场/铺垫/高潮/收尾",
      "content": "内容描述",
      "emotion": "情感基调",
      "transition_to_next": "与下一张的衔接方式"
    }
  ],
  "visual_coherence": {
    "color_consistency": "色调统一性评分(1-10)",
    "style_consistency": "风格统一性评分(1-10)",
    "pacing": "节奏感评价",
    "variety": "变化丰富度"
  },
  "cover_recommendation": {
    "best_cover_index": 1,
    "reason": "选择原因",
    "crop_suggestion": "裁剪建议"
  },
  "reorder_suggestion": {
    "new_order": [3, 1, 5, 2, 4],
    "reason": "重排逻辑"
  },
  "missing_shots": ["建议补充的镜头类型"],
  "social_media_bundle": {
    "carousel_title": "组图标题",
    "carousel_description": "组图描述"
  }
}
```

### User Prompt Example
```
这是一组旅行照片（共5张），请分析它们的视觉叙事效果，包括排列顺序是否合理、色调是否统一、建议哪张做封面、是否需要重排或补充镜头。严格按照JSON格式输出。
```

---

## 6. Style Reference & Emotion Mapping Prompt

### Style Reference
```
System: 你是摄影风格识别专家。识别照片的摄影风格并给出参考建议。输出JSON：
{
  "primary_style": "风格名称",
  "style_tags": ["标签1", "标签2"],
  "similar_to": ["类似风格的摄影师或作品"],
  "color_palette": {
    "dominant_colors": ["#hex1", "#hex2"],
    "color_mood": "暖调/冷调/中性"
  },
  "technique_used": ["技法1", "技法2"],
  "style_suggestions": ["可以尝试的其他风格方向"]
}
```

### Emotion Mapping
```
System: 你是情感分析专家。分析照片传递的情感。输出JSON：
{
  "primary_emotion": "主要情感",
  "emotion_intensity": 8,
  "emotion_spectrum": {
    "joy": 0.7,
    "serenity": 0.8,
    "nostalgia": 0.3,
    "melancholy": 0.1,
    "excitement": 0.2,
    "mystery": 0.1
  },
  "atmosphere": "氛围描述",
  "story_behind": "推测照片背后的故事",
  "viewer_impact": "对观者的预期情感影响",
  "music_pairing": "适合搭配的音乐风格"
}
```

---

## API Usage Examples

### Python (OpenAI SDK) — Single Photo Analysis
```python
from openai import OpenAI
import base64, json

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://llm-1r8e612iutxlixav.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

def analyze_photo(image_url: str, analysis_type: str = "aesthetic") -> dict:
    """Analyze a photo with Qwen-VL Max."""
    
    prompts = {
        "aesthetic": "请对这张照片进行专业美学评分分析，从构图、光影、色彩、主体表达、技术质量、情感共鸣六个维度评分(1-10)，并给出强项、弱项、风格标签和一句话总评。严格按照JSON格式输出。",
        "composition": "请分析这张照片的构图技法，包括构图类型、三分法应用、引导线、景深层次、视觉重量平衡、留白比例，并给出适合不同平台的裁剪建议。严格按照JSON格式输出。",
        "improvement": "请分析这张照片的不足，给出具体的拍摄改进建议和后期处理建议，包括优先级和难度。严格按照JSON格式输出。",
        "copywriting": "请为这张照片生成适合小红书、Instagram、抖音、朋友圈的文案，包含标签和话题。根据照片实际内容和氛围撰写。严格按照JSON格式输出。",
        "emotion": "请分析这张照片传递的情感，包括主要情感、情感强度、情感光谱、氛围、推测故事、对观者的影响。严格按照JSON格式输出。",
        "style": "请识别这张照片的摄影风格，包括风格标签、类似摄影师、色彩面板、使用技法、建议尝试的其他风格。严格按照JSON格式输出。",
    }
    
    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "system", "content": "你是专业的摄影分析AI。所有输出必须严格为JSON格式，不要输出任何JSON以外的文字。"},
            {"role": "user", "content": [
                {"type": "text", "text": prompts.get(analysis_type, prompts["aesthetic"])},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ],
        temperature=0.3,  # Low temp for consistent structured output
        max_tokens=2000,
    )
    
    result = response.choices[0].message.content
    # Try to parse JSON from response
    try:
        # Handle case where response has markdown code blocks
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        return json.loads(result.strip())
    except json.JSONDecodeError:
        return {"raw_text": result}


# Multi-photo storytelling analysis
def analyze_photo_story(image_urls: list) -> dict:
    """Analyze multiple photos for visual storytelling."""
    content = [
        {"type": "text", "text": f"这是一组{len(image_urls)}张照片，请分析它们的视觉叙事效果，包括排列顺序、色调统一性、建议封面、重排建议、缺失镜头类型。严格按照JSON格式输出。"},
    ]
    for i, url in enumerate(image_urls):
        content.append({"type": "image_url", "image_url": {"url": url}})
    
    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "system", "content": "你是视觉叙事专家和图片编辑总监。分析多张照片组成的视觉故事。所有输出必须严格为JSON格式。"},
            {"role": "user", "content": content}
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    
    result = response.choices[0].message.content
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        return json.loads(result.strip())
    except json.JSONDecodeError:
        return {"raw_text": result}
```

### Base64 Image Input
```python
import base64

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Usage:
b64 = encode_image("photo.jpg")
image_url = f"data:image/jpeg;base64,{b64}"
```

---

## Best Practices Summary

### Prompt Engineering for Qwen-VL
1. **Always request JSON output** — Add "严格按照JSON格式输出" to every prompt
2. **Use system message** — Set the role/expertise in system, task in user
3. **Low temperature (0.1-0.3)** — For consistent structured analysis
4. **Specific dimensions** — Don't ask "analyze this photo"; ask for specific dimensions with scoring scales
5. **Chinese prompts work best** — Qwen-VL is optimized for Chinese; use Chinese prompts for best results
6. **Parse response robustly** — Handle ```json code blocks, trailing text
7. **Image quality matters** — Use high-res URLs, not thumbnails
8. **Multi-image limit** — Tested up to 9 images per request; keep total token count reasonable

### For Photo Aesthetic Scoring
- Define clear rubrics (1-10 per dimension) in the prompt
- Ask for strengths AND weaknesses (prevents one-sided analysis)
- Request style tags for categorization
- Ask for a one-line summary for UI display

### For Social Media Copywriting
- Specify exact platforms (小红书 vs Instagram have different conventions)
- Request multiple tone options (literary, humorous, minimalist)
- Include hashtag generation
- Ask for platform-specific formatting (emoji for 小红书, English for Instagram)

### For Multi-Photo Storytelling
- Order matters — present photos in sequence
- Request explicit reorder suggestions
- Ask for cover recommendation with reasoning
- Request missing shot suggestions for completeness

### Error Handling
- Handle JSON parse failures gracefully (return raw text)
- Retry with simplified prompt if response too long
- Use `max_tokens` to control response length
- Monitor token usage: ~1259 prompt tokens for one image + short prompt
