# 📋 SatQuery AI — Claims & Limitations

Transparent documentation of what we built, what we reused, and what we did not build.

---

## ✅ Verified Capabilities (Tested and Working)

| Capability | Model/Component | Test Status |
|------------|----------------|-------------|
| Optical image captioning | EarthDial 4B | ✅ Verified |
| Visual question answering | EarthDial 4B | ✅ Verified |
| Object detection with bounding boxes | EarthDial 4B | ✅ Verified |
| Scene classification | EarthDial 4B | ✅ Verified |
| Visual grounding (bbox coordinates) | EarthDial 4B | ✅ Verified |
| SAR vessel detection (ships) | YOLOv8 | ✅ Verified |
| SAR bounding box visualization | visualize.py | ✅ Verified |
| Optical grounding visualization | visualize.py | ✅ Verified |
| Deterministic intent routing (7 intents) | router.py | ✅ Verified |
| Pre-computed demo scenarios (6) | demos.py | ✅ Verified |
| Gradio web UI | app.py | ✅ Verified |
| Edge-case handling (8 scenarios) | pipeline.py + app.py | ✅ Verified |
| Query history (last 5) | pipeline.py | ✅ Verified |

---

## 🔄 Reused Pretrained Components

| Component | Source | License | What It Does |
|-----------|--------|---------|--------------|
| EarthDial 4B RGB | MBZUAI/EarthDial (CVPR 2025) | Apache 2.0 | RS image captioning, VQA, grounding, classification |
| YOLOv8n SAR vessel detection | MeWan2808/yolov8n-sar-vessel-detection | Not specified | Ship detection in SAR imagery |
| Gradio | Gradio team | Apache 2.0 | Web UI framework |
| PyTorch | Meta AI | BSD | ML framework |
| HuggingFace Transformers | HuggingFace | Apache 2.0 | Model loading and inference |

**We did not train, fine-tune, or modify any model weights.**

---

## 🏗️ Our System-Level Contribution

| Contribution | Description |
|--------------|-------------|
| **Intent Router** | Keyword-based classifier mapping natural-language RS queries to 7 intent categories in <1ms |
| **Orchestration Pipeline** | Routes queries to the right model (EarthDial for optical, YOLOv8 for SAR), handles errors, tracks history |
| **Visualization Engine** | Parses model output (both EarthDial grounding format and SAR detection tables) → draws color-coded bounding boxes with labels and confidence scores on images |
| **SAR Integration** | Isolated subprocess architecture allowing incompatible model environments to coexist on one machine |
| **Unified UI** | Gradio interface showing analysis results, annotated images, query history, and model metadata |
| **Demo Scenarios** | 6 pre-computed demonstrations covering all supported capabilities |
| **Honest Documentation** | Clear separation of what works vs. what doesn't |

---

## ❌ Unsupported Capabilities (We Did NOT Build These)

| Capability | Status | Why |
|------------|--------|-----|
| General SAR scene understanding | ❌ Not supported | Requires SAR-specific VLM |
| SAR VQA | ❌ Not supported | No SAR VLM available |
| SAR+optical learned fusion | ❌ Not supported | Requires trained fusion model |
| Change detection | ❌ Not supported | Requires bi-temporal model + registered imagery |
| SARDet-100K integration | ❌ Not possible | Incompatible with Windows + Python 3.12 |
| Model training/fine-tuning | ❌ Not done | All models are pretrained |
| Multi-image comparison | ❌ Not supported | Single-image pipeline only |
| Real-time streaming | ❌ Not supported | Batch inference only |
| Edge deployment | ❌ Not supported | Requires GPU + Python stack |

---

## ⚠️ Honest Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| EarthDial inference is slow (50-260s) | Long wait for live queries | Pre-computed demos for instant results |
| SAR is ship-only | Cannot detect other objects | Documented; honest in UI |
| No change detection | Cannot compare images | Listed as future work |
| Keyword router is simple | May miss nuanced queries | Works well for common RS queries |
| EarthDial bbox is approximate | May not align perfectly | Acceptable for prototype |
| SAR subprocess has ~2s overhead | Added to pipeline time | Acceptable for demo |
| Single GPU, no scaling | Cannot serve multiple users | Prototype only |

---

## 🎯 What This Prototype Demonstrates

SatQuery AI demonstrates that **system-level integration** of pretrained models can create a useful remote sensing assistant. The key engineering contributions are:

1. **Multi-model orchestration**: Optical VLM + SAR detector working through one interface
2. **Visual evidence**: Not just text answers — annotated images with bounding boxes
3. **Honest design**: Each component does what it actually can, nothing is faked
4. **Practical deployment**: Runs on a laptop GPU (4 GB VRAM), not cloud infrastructure
5. **Deterministic behavior**: Router gives consistent, explainable results

---

## 📝 Summary for Judges

> SatQuery AI is a working prototype that integrates EarthDial 4B (for optical RS understanding) and YOLOv8 (for SAR vessel detection) into a unified web interface with visual evidence output. We did not train any models — our contribution is the system-level engineering that makes pretrained models work together reliably. We are honest about limitations: SAR is ship-only, change detection is future work, and inference is slow due to hardware constraints.
