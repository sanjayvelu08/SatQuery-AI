# 🎬 SatQuery AI — Demo Script (3-5 Minutes)

## Recommended Evaluator Walkthrough

### Opening (30 seconds)

> "SatQuery AI is an interactive vision-language assistant for remote sensing image analysis. It lets users ask natural-language questions about satellite imagery and get expert-level analysis — combining an optical VLM for scene understanding with a SAR vessel detector, all through a unified interface."

**Action**: Show the Gradio UI at `http://127.0.0.1:7860` — point out the clean layout with image upload, query input, and demo selector.

---

### Demo 1: Optical Captioning (60 seconds)

**Select**: 🌾 Agricultural Landscape Analysis

> "Let's start with a Sentinel-2 agricultural scene. Watch — I'll select the pre-computed demo for instant results."

**Action**: Click "Analyze Image" → show the analysis result.

**Point at**: The detailed caption describing cropland, fallow fields, vegetation corridors, and farm structures.

**Say**: "Notice the system correctly identifies agricultural patterns — active cropland, fallow fields, and even small structures. The EarthDial model produces RS-expert-level analysis."

**Technical note**: "This uses EarthDial 4B, a 4.1B parameter vision-language model quantized to run on our 4 GB GPU."

---

### Demo 2: SAR Vessel Detection (60 seconds)

**Select**: 🛰️ SAR Maritime Vessel Detection

> "Now let's try something different — SAR imagery. SAR penetrates clouds and works day/night, making it critical for maritime surveillance."

**Action**: Click "Analyze Image" → show both the text result AND the annotated image.

**Point at**: The annotated SAR image with colored bounding boxes around detected ships.

**Say**: "The system detected 3 ships with bounding boxes and confidence scores. Look at the visual evidence panel — colored boxes with ship labels and confidence percentages."

**Technical note**: "This uses a YOLOv8 model running in an isolated environment, producing results in 50 milliseconds with only 21 MB of GPU memory."

**Honest note**: "This detector identifies ships only — it doesn't provide general SAR scene understanding. But ship detection is a critical capability for coastal surveillance."

---

### Demo 3: Object Detection + Grounding (60 seconds)

**Select**: 🔍 Infrastructure Detection

> "Let's look at feature detection with bounding boxes."

**Action**: Click "Analyze Image" → show the detection results and annotated image.

**Point at**: The bounding box coordinates in the text AND the visual evidence with the drawn box around the building cluster.

**Say**: "The system not only describes what it sees — buildings, agricultural plots, vegetation corridors — but also provides approximate bounding box coordinates. The visual evidence shows where each feature is located."

**Technical note**: "This demonstrates the grounding capability of EarthDial — it can locate objects in the image, not just describe them."

---

### Demo 4: Visual QA + Scene Classification (45 seconds)

**Select**: 🏙️ Urban Area Assessment

> "Let's ask a specific question about an urban scene."

**Action**: Click "Analyze Image"

**Point at**: The land use breakdown table (Residential ~60%, Commercial ~15%, etc.)

**Say**: "The system classifies this as peri-urban and provides a detailed land use breakdown. Notice the table format — the model structured its analysis to be directly useful for urban planners."

**Optional**: Switch to 🗺️ Scene Classification to show the classification-focused response.

---

### Closing (30 seconds)

> "SatQuery AI demonstrates that a multimodal remote sensing assistant can be built by integrating pretrained models — EarthDial for optical understanding and YOLOv8 for SAR detection — with a deterministic intent router and visualization engine. The key insight is that system-level integration matters as much as individual model performance."

**Show**: The About section with the architecture description.

**Mention**: "We're transparent about limitations — SAR is ship-only, change detection is future work, and all models are pretrained, not trained by us."

---

## Key Talking Points

1. **Dual-modality**: Optical VLM + SAR detector working together
2. **Visual evidence**: Not just text — annotated images with bounding boxes
3. **Efficient architecture**: Runs on a laptop GPU (4 GB VRAM)
4. **Honest design**: Each component does what it actually can, no false claims
5. **System-level contribution**: Routing, orchestration, visualization, UI

## What to Avoid Saying

- ❌ "We trained a model" → Say "We integrated pretrained models"
- ❌ "General SAR understanding" → Say "SAR vessel detection"
- ❌ "SARDet-100K" → Say "YOLOv8 SAR detector"
- ❌ "Change detection" → Say "Future work"
- ❌ "Our fusion model" → Say "Multi-model orchestration"
