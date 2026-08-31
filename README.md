# 🛰️ SatQuery AI

**Interactive Vision-Language Assistant for Remote Sensing Image Analysis**

ISRO Problem Statement SIH26167 — Smart India Hackathon 2026

## What It Does

SatQuery AI lets users ask natural-language questions about satellite imagery and get expert-level remote sensing analysis. Upload an optical or SAR satellite image and ask questions like:

- *"Describe this satellite image"* → detailed captioning
- *"Are there any buildings visible?"* → visual question answering
- *"Classify the land cover"* → scene classification
- *"Locate the main features"* → object detection with bounding boxes
- *"Detect ships in this SAR image"* → SAR vessel detection with annotated output

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Input                        │
│              (Image + Natural Language Query)         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Intent Router (< 1ms)                   │
│    Classifies: caption, vqa, detect, grounding,     │
│    classification, sar, change (unsupported)        │
└──────────┬──────────────────────────┬───────────────┘
           │                          │
     Optical Path               SAR Path
           │                          │
           ▼                          ▼
┌──────────────────┐    ┌──────────────────────────┐
│  EarthDial 4B    │    │  YOLOv8 SAR Detector     │
│  (InternVL +     │    │  (via isolated subprocess)│
│   Phi-3, 4-bit)  │    │  Ship detection, ~50ms   │
│  ~50-260s/query  │    │                          │
└────────┬─────────┘    └────────────┬─────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────┐
│            Visualization Engine                      │
│    Draws bounding boxes, labels, confidence scores   │
│    on annotated output image                         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            Gradio Web UI (port 7860)                 │
│    Analysis result + annotated image + query history │
└─────────────────────────────────────────────────────┘
```

## Supported Features

| Feature | Model | Status | Latency |
|---------|-------|--------|---------|
| Image captioning | EarthDial 4B | ✅ Working | ~50-240s |
| Visual QA | EarthDial 4B | ✅ Working | ~50-60s |
| Object detection + grounding | EarthDial 4B | ✅ Working | ~70-260s |
| Scene classification | EarthDial 4B | ✅ Working | ~50-60s |
| SAR vessel detection | YOLOv8 | ✅ Working | ~50ms |
| Annotated visual evidence | visualize.py | ✅ Working | <100ms |
| Pre-computed demo scenarios | 6 demos | ✅ Working | Instant |
| Change detection | — | ⏳ Future work | — |

## Quick Start

### Prerequisites

- Python 3.10-3.12
- NVIDIA GPU with 4+ GB VRAM (tested on RTX 3050)
- CUDA 12.1+ (via PyTorch)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd SatQuery-AI

# Install main dependencies
pip install -r requirements.txt

# Download EarthDial weights (7.8 GB, one-time)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('akshaydudhane/EarthDial_4B_RGB', local_dir='checkpoints/EarthDial_4B_RGB')
"

# Set up isolated SAR environment (one-time)
python -m venv sar_venv
sar_venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
sar_venv/Scripts/pip install ultralytics opencv-python

# Download SAR model weights (6.3 MB)
mkdir -p checkpoints/sar_vessel/unquantized
# Download from: https://huggingface.co/MeWan2808/yolov8n-sar-vessel-detection
# Place best.pt in checkpoints/sar_vessel/unquantized/
```

### Run the Web UI

```bash
python -X utf8 -m satquery.app
# Opens at http://127.0.0.1:7860
```

## Project Structure

```
SatQuery-AI/
├── satquery/                    # Core package (1,884 lines)
│   ├── __init__.py
│   ├── app.py                   # Gradio web UI (461 lines)
│   ├── demos.py                 # 6 pre-computed demo scenarios (198 lines)
│   ├── pipeline.py              # Query → Route → VLM/SAR → Result (183 lines)
│   ├── router.py                # Intent classifier, 7 intents (202 lines)
│   ├── sar_infer.py             # SAR inference (runs in sar_venv) (134 lines)
│   ├── sar_tool.py              # SAR subprocess bridge (195 lines)
│   ├── visualize.py             # Bbox drawing + image annotation (260 lines)
│   ├── vlm.py                   # EarthDial VLM wrapper (142 lines)
│   └── test_pipeline.py         # Pipeline tests (107 lines)
├── checkpoints/
│   ├── EarthDial_4B_RGB/        # VLM weights (7.8 GB)
│   └── sar_vessel/              # SAR detector weights (6.3 MB)
├── test_images/                 # Sample satellite images
├── EarthDial/                   # EarthDial source (for sys.path)
├── sar_venv/                    # Isolated Python env for SAR
├── .gitignore
├── requirements.txt
└── README.md
```

## Technical Details

| Component | Details |
|-----------|---------|
| **VLM** | EarthDial 4B RGB (InternVL + Phi-3, 4.1B params, 4-bit quantized) |
| **SAR** | YOLOv8n, vessel detection (ship class only) |
| **Router** | Keyword-based, 7 intents, <1ms latency, zero VRAM |
| **VRAM** | EarthDial: ~2.85 GB, SAR: ~21 MB, Headroom: ~1.2 GB |
| **GPU** | NVIDIA RTX 3050 Laptop (4 GB VRAM) |
| **Visualization** | PIL-based bbox rendering, color-coded labels + confidence |

## Known Limitations

- **Inference speed**: EarthDial runs slowly (~50-260s) due to CPU offloading on 4 GB VRAM
- **SAR scope**: Only ship/vessel detection — no general SAR scene understanding
- **Change detection**: Not implemented (requires bi-temporal registered imagery + trained model)
- **EarthDial import**: Must use sys.path (Windows cmake issue prevents pip install)
- **Single SAR class**: Only detects "ship" — no other maritime objects

## What We Did NOT Build

We want to be transparent about what SatQuery AI is **not**:

- ❌ We did **not** train any models — all models are pretrained
- ❌ We did **not** integrate SARDet-100K (incompatible with Windows + Python 3.12)
- ❌ We did **not** build SAR VQA or general SAR understanding
- ❌ We did **not** build SAR+optical learned fusion
- ❌ We did **not** implement change detection

Our contribution is the **system-level integration**: routing, orchestration, visualization, and UI that makes multiple pretrained models work as a unified assistant.

## License

- EarthDial: Apache 2.0
- YOLOv8 SAR: HuggingFace model (MeWan2808/yolov8n-sar-vessel-detection)
- SatQuery AI: SIH 2026 prototype

## Team

Smart India Hackathon 2026 — ISRO Problem Statement SIH26167
