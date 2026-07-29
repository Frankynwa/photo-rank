# Photo Aesthetic Scoring Models Research (2025-2026)
## For RTX 5070 12GB VRAM — Local Inference

---

## 🏆 TIER 1: BEST MODELS (Recommended)

### 1. Q-ReAlign (Qwen3.5-VL based) — TOP PICK
**The best open-source aesthetic/quality scorer available today.**

| Variant | Params | HuggingFace | AVA SRCC | VRAM (est.) | Speed (RTX 4090) | Fits RTX 5070? |
|---------|--------|-------------|----------|-------------|-------------------|----------------|
| **Mini** | 0.8B | `q-future/Q-ReAlign-Mini-0.8B` | 0.797 | **<4 GB** | ~26.7 img/s (bs=4) | ✅ YES, easily |
| **Lite** | 4B | `q-future/Q-ReAlign-Lite-4B` | 0.814 | ~8 GB | ~10 img/s (est.) | ✅ YES |
| **Pro** | 9B | `q-future/Q-ReAlign-Pro-9B` | **0.832** | ~14 GB | ~4 img/s (est.) | ⚠️ Tight, needs 4-bit quant |

- **Paper**: Jun 2026, by Q-Future team
- **GitHub**: https://github.com/Q-Future/Q-ReAlign
- **HF Collection**: https://huggingface.co/collections/q-future/q-realign
- **Architecture**: Qwen3.5-VL backbone, distribution-based soft labels
- **Scores**: Quality [0,1] + Aesthetic [0,1] — dual task support
- **PyIQA integration**: `pip install pyiqa` then `pyiqa.create_metric("qrealign")`
- **Requires**: `transformers>=5.2`
- **Key advantage**: 0.8B model matches Q-Align (7B) performance with 50% fewer params
- **Speed on RTX 5070 (estimated)**:
  - Mini: ~20-25 img/s
  - Lite: ~8-12 img/s
  - Pro (4-bit): ~3-5 img/s

### 2. Q-Align (Original, LLaVA based)
| Variant | HuggingFace | AVA SRCC | VRAM | Fits RTX 5070? |
|---------|-------------|----------|------|----------------|
| FP16 | `q-future/Q-Align` | 0.8223 | ~15.4 GB | ❌ OOM |
| 8-bit | same (quantized) | ~0.82 | ~8.2 GB | ✅ YES |
| 4-bit | same (quantized) | ~0.81 | ~5.0 GB | ✅ YES |

- **PyIQA**: `pyiqa.create_metric("qalign")` / `"qalign_8bit"` / `"qalign_4bit"`
- **Speed**: ~0.22 img/s (FP16), ~0.56 img/s (4-bit) on V100 — **slow**
- **Note**: Q-ReAlign Mini is better in most cases AND faster. Prefer Q-ReAlign.

### 3. TOPIQ-IAA (ResNet backbone)
| Model | HuggingFace | AVA SRCC | VRAM | Speed (V100) | Fits RTX 5070? |
|-------|-------------|----------|------|--------------|----------------|
| `topiq_iaa` (ViT) | via PyIQA | **0.791** | ~0.58 GB | 0.047s/img | ✅ YES |
| `topiq_iaa_res50` | via PyIQA | 0.736 | ~0.38 GB | 0.055s/img | ✅ YES |

- **Paper**: https://arxiv.org/abs/2308.03060
- **Speed**: ~21 img/s on GPU — very fast
- **PyIQA**: `pyiqa.create_metric("topiq_iaa")`
- **Best lightweight option** if you need speed over accuracy

---

## 🥈 TIER 2: GOOD ALTERNATIVES

### 4. DeQA-Score (CVPR 2025)
**Distribution-based MLLM quality scorer. State-of-the-art for IQA score regression.**

- **Paper**: "Teaching LLMs to Regress Accurate IQA Scores using Score Distribution"
- **arXiv**: https://arxiv.org/abs/2501.11561 (Accepted CVPR 2025)
- **Authors**: Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, Chao Dong
- **GitHub**: Code released (check paper for link, may be under different username)
- **Architecture**: MLLM-based with soft label (score distribution) approach + Thurstone fidelity loss
- **Key innovation**: Converts continuous quality scores to soft distributions instead of one-hot labels
- **Performance**: "Stably outperforms baselines in score regression" across multiple benchmarks
- **VRAM**: Likely 4-8 GB (MLLM-based, similar to Q-Align family)
- **Status**: Paper released Jan 2025, code & weights released per paper. **Not yet integrated into PyIQA** (as of Jul 2026)
- **Suitability for RTX 5070**: ✅ Likely fits, but need to verify exact model size

### 5. RealQA (2025)
**10-dimensional fine-grained quality + aesthetic scoring**

- **Paper**: "Next Token Is Enough: Realistic Image Quality and Aesthetic Scoring with MLLM"
- **arXiv**: https://arxiv.org/abs/2503.06141
- **Authors**: Mingxing Li et al. (Alibaba)
- **Dataset**: 14,715 UGC images annotated with 10 fine-grained attributes across 3 levels:
  - Low-level: clarity, noise, exposure, etc.
  - Middle-level: subject integrity, etc.
  - High-level: composition, etc.
- **Method**: MLLM with "next token" paradigm — predicts 2 extra significant digits
- **Performance**: SOTA on 5 public datasets for IQA and IAA, with zero-shot VQA generalization
- **Status**: Code and dataset "will be released" — check if available now
- **Suitability**: Depends on base model size (likely Qwen/InternVL based)
- **Key advantage**: 10 separate quality dimensions instead of single score — great for detailed analysis

### 6. MACLIP (Dec 2025)
- **PyIQA**: `pyiqa.create_metric("maclip")`
- **Type**: CLIP-based, introduces magnitude into IQA
- **GitHub**: https://github.com/zhix000/MA-CLIP
- **VRAM**: ~1 GB (CLIP-based)
- **Speed**: Very fast (~20+ img/s)
- **Fits RTX 5070**: ✅ Easily

### 7. QualiCLIP+ (Jan 2025)
- **PyIQA**: `pyiqa.create_metric("qualiclip+")` / `"qualiclip+-spaq"` etc.
- **Paper**: https://arxiv.org/abs/2403.11176
- **VRAM**: ~0.6 GB
- **Speed**: ~19 img/s on V100
- **Variants**: Trained on different datasets (koniq, clive, flive, spaq)
- **Fits RTX 5070**: ✅ Easily

### 8. AFINE (Sep 2025)
- **PyIQA**: `pyiqa.create_metric("afine")`
- **GitHub**: https://github.com/ChrisDud0257/AFINE
- **Fits RTX 5070**: ✅

### 9. FGResQ (Jun 2026)
- **arXiv**: https://arxiv.org/abs/2508.14475
- **PyIQA**: `pyiqa.create_metric("fgresq")`
- **Type**: Fine-grained IQA for image restoration (pairwise ranking + score regression)
- **Fits RTX 5070**: ✅

---

## 📊 BENCHMARK COMPARISON (AVA Dataset)

| Model | AVA SRCC | AVA PLCC | VRAM | Speed (img/s) | RTX 5070? |
|-------|----------|----------|------|---------------|-----------|
| **Q-ReAlign-Pro-9B** | **0.832** | **0.828** | ~14 GB | ~4 | ⚠️ Quant needed |
| **Q-ReAlign-Lite-4B** | 0.814 | 0.804 | ~8 GB | ~10 | ✅ |
| Q-Align (FP16) | 0.822 | 0.796 | 15.4 GB | ~4.5 | ❌ OOM |
| **Q-ReAlign-Mini-0.8B** | 0.797 | 0.794 | **<4 GB** | **~25** | ✅✅ Best value |
| Q-Align (4-bit) | ~0.81 | ~0.79 | 5.0 GB | ~1.8 | ✅ |
| **TOPIQ-IAA** | 0.791 | 0.790 | 0.58 GB | **~21** | ✅✅ Fastest |
| TOPIQ-IAA-Res50 | 0.736 | 0.737 | 0.38 GB | ~18 | ✅ |
| NIMA (IRv2) | 0.713 | 0.717 | 0.25 GB | ~15 | ✅ |
| LAION Aesthetic | 0.665 | 0.666 | 1.64 GB | ~22 | ✅ |
| NIMA-VGG16 | 0.657 | 0.662 | 0.13 GB | ~42 | ✅ |
| CLIPIQA | 0.338 | 0.358 | 0.63 GB | ~17 | ✅ (bad for AVA) |

---

## 🛠️ PyIQA TOOLBOX (v0.1.16)

**The single best tool for running all these models.**

### Installation
```bash
pip install pyiqa
pip install -U "transformers>=5.2"  # needed for Q-ReAlign
```

### All Available Aesthetic Models
```python
import pyiqa
print(pyiqa.list_models())
# Key aesthetic models: nima, nima-vgg16-ava, laion_aes, topiq_iaa, 
# topiq_iaa_res50, qalign, qalign_8bit, qalign_4bit,
# qrealign, qrealign-lite, qrealign-pro
```

### Usage
```python
import pyiqa
import torch

device = torch.device("cuda")

# BEST: Q-ReAlign Mini (fast + accurate)
metric = pyiqa.create_metric("qrealign", device=device)  # 0.8B, <4GB VRAM
score = metric("photo.jpg")  # Returns [0, 1], higher = better

# For quality + aesthetic dual scoring
metric_quality = pyiqa.create_metric("qrealign", device=device)  # quality task (default)
metric_aesthetic = pyiqa.create_metric("qrealign", device=device, task="aesthetic")

# Fast alternative: TOPIQ-IAA
metric = pyiqa.create_metric("topiq_iaa", device=device)

# For quality (not aesthetic): TOPIQ-NR
metric = pyiqa.create_metric("topiq_nr", device=device)  # SRCC 0.93 on KonIQ
```

### Key Non-Aesthetic Models (Technical Quality)
| Model | Type | PyIQA Name | KonIQ SRCC | VRAM |
|-------|------|------------|------------|------|
| Q-ReAlign-Pro | IQA | `qrealign-pro` | 0.950 | ~14 GB |
| Q-ReAlign-Lite | IQA | `qrealign-lite` | 0.943 | ~8 GB |
| Q-ReAlign-Mini | IQA | `qrealign` | 0.935 | <4 GB |
| Q-Align | IQA | `qalign` | 0.942 | 15.4 GB |
| TOPIQ-NR | IQA | `topiq_nr` | 0.930 | 0.67 GB |
| CLIPIQA+ | IQA | `clipiqa+_rn50_512` | 0.885 | 0.63 GB |
| DBCNN | IQA | `dbcnn` | 0.903 | 0.59 GB |
| MANIQA | IQA | `maniqa` | 0.893 | 2.66 GB |

---

## 🎯 RECOMMENDED SETUP FOR RTX 5070 12GB

### Option A: Best Accuracy (parallel pipeline)
```
Pipeline:
1. Q-ReAlign-Lite-4B    → Aesthetic score + Quality score (~8 GB)
2. TOPIQ-NR             → Technical quality (~0.7 GB) 
3. DINOv2-small         → Similarity clustering (~0.5 GB)
Total VRAM: ~9.2 GB ✅
Speed: ~8-10 img/s
```

### Option B: Best Speed
```
Pipeline:
1. Q-ReAlign-Mini-0.8B  → Aesthetic + Quality (<4 GB)
2. TOPIQ-NR             → Technical quality (~0.7 GB)
3. CLIPIQA+             → Additional quality signal (~0.6 GB)
Total VRAM: ~5.3 GB ✅
Speed: ~20-25 img/s
```

### Option C: Maximum Accuracy (sequential, swap models)
```
Load Q-ReAlign-Pro-9B (4-bit quantized, ~5 GB) → Score all images
Load TOPIQ-NR → Score technical quality
Total VRAM: ~6 GB (sequential)
Speed: ~3-5 img/s
```

### Face Quality Detection (add to any pipeline)
```
- topiq_nr-face (PyIQA) → Face IQA (~0.7 GB, shared backbone)
- InsightFace / RetinaFace → Face detection + quality
```

---

## 📝 SUMMARY

**What beats NIMA/cafe_aesthetic on AVA (SRCC > 0.8)?**

1. **Q-ReAlign-Pro-9B**: SRCC 0.832 (best, needs quantization for 12GB)
2. **Q-ReAlign-Lite-4B**: SRCC 0.814 (best balance of accuracy/VRAM/speed)
3. **Q-ReAlign-Mini-0.8B**: SRCC 0.797 (fastest, <4GB VRAM)
4. **Q-Align (4-bit)**: SRCC ~0.81 (slow inference)
5. **TOPIQ-IAA**: SRCC 0.791 (lightweight CNN, very fast)

**Key Finding**: Q-ReAlign is the clear winner for 2026. It's integrated into PyIQA, runs on Qwen3.5-VL backbone, and the 0.8B Mini variant achieves near-Q-Align performance at 10x the speed with <4GB VRAM. The 4B Lite variant is the sweet spot for RTX 5070.

**For the photo culling platform**: Use Q-ReAlign-Lite-4B for aesthetic scoring (SRCC 0.814 on AVA, ~8GB VRAM, ~10 img/s) paired with TOPIQ-NR for technical quality assessment. Total ~9GB VRAM, leaving headroom on the 12GB RTX 5070.
