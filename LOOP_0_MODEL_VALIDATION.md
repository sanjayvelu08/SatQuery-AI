# LOOP 0 — Model & Hardware Validation

> **Date**: August 31, 2026
> **Project**: SatQuery AI (SIH26167 — ISRO)
> **Purpose**: Feasibility validation before writing any application code
> **Status**: COMPLETE

---

## 1. Machine Hardware

| Component | Value | Notes |
|-----------|-------|-------|
| **CPU** | AMD Ryzen 7 7445HS | 8 cores, good |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop | **4 GB VRAM only** |
| **GPU Compute** | CUDA 8.6 | Supports FP16/BF16 |
| **GPU Driver** | 610.88 | Recent |
| **System RAM** | 16 GB | Usable ~15.3 GB |
| **Disk** | 448 GB total, 253 GB free | Sufficient |
| **OS** | Windows 11 (MINGW64/Git Bash) | |

### ⚠️ CRITICAL CONSTRAINT: 4 GB VRAM

This is the single most important fact for architecture decisions. With 4 GB VRAM:

| What FITS (≤3 GB model weights) | What DOES NOT FIT |
|---|---|
| EarthDial 4B at4-bit (~2 GB) | EarthDial 4B at FP16 (~8 GB) |
| GroundingDINO Tiny (~1.3 GB FP32) | GeoChat 7B at4-bit (~3.5 GB) ❌ |
| MobileSAM (~40 MB) | GeoChat 7B at FP16 (~14 GB) ❌ |
| SARDet-100K R50 (~0.5 GB) | SARDet-100K ConvNext-B (~1.1 GB) ⚠️ tight |
| Qwen2-0.5B (~1 GB4-bit) | DeltaVLM / Vicuna-7B (~3.5 GB4-bit) ❌ |

**You can load AT MOST one VLM at a time. All other models must be unloaded before loading the next.**

---

## 2. Environment

| Component | Installed Version | Required by Models |
|-----------|------------------|--------------------|
| **Python** | 3.12.10 | ✅ Compatible with all models |
| **PyTorch** | 2.13.0+cpu (just installed) | Need CUDA version for GPU |
| **transformers** | 5.16.1 (just installed) | GeoChat needs custom code |
| **huggingface_hub** | 1.29.0 | ✅ |
| **accelerate** | 1.14.0 | ✅ for device_map |
| **Conda** | NOT installed | EarthDial recommends conda |
| **CUDA toolkit** | NOT installed (driver only) | Need for GPU inference |
| **flash-attn** | NOT installed | EarthDial training needs it (inference optional) |

### What Needs Installing Before Loop 1

1. **CUDA-enabled PyTorch** (replace CPU-only version): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
2. **EarthDial repo** (custom code): `git clone https://github.com/hiyamdebary/EarthDial.git`
3. **bitsandbytes** (for4-bit quantization): `pip install bitsandbytes`
4. **mmdet stack** (for SARDet-100K): `mim install mmengine mmcv mmdet` — **WARNING: heavy dependency, may conflict**

---

## 3. EarthDial Validation

| Field | Verified Value |
|-------|---------------|
| **Repository** | https://github.com/hiyamdebary/EarthDial |
| **HuggingFace** | `akshaydudhane/EarthDial_4B_RGB` |
| **Exists on HuggingFace** | ✅ YES |
| **Config loads** | ✅ YES (with `trust_remote_code=True`) |
| **Architecture** | InternVLChatModel with Phi-3 backbone |
| **Parameters** | ~4.1B (verified from `total_size`: 8.29 GB FP16) |
| **Files** | 2 safetensors shards + config + tokenizer + inference.py |
| **Custom code required** | ✅ YES — needs `earthdial` package from GitHub |
| **Loading method** | NOT standard `AutoModelForCausalLM`. Requires `InternVLChatModel.from_pretrained()` |
| **Inference script** | Included in repo as `inference.py` |
| **License** | Apache 2.0 (from InternVL lineage) |
| **VRAM FP16** | ~8 GB ❌ doesn't fit on this machine |
| **VRAM4-bit (bitsandbytes)** | ~2-2.5 GB ✅ FITS on this machine |
| **VRAM8-bit** | ~4 GB ⚠️ BARELY fits |
| **Dependencies** | transformers, torch, PIL, earthdial (custom) |
| **Supports SAR** | ❌ NO — trained on RGB only |
| **Supports optical RS** | ✅ YES — captioning, VQA, grounding, classification |
| **Model type tag** | `internvl_chat` (not standard transformers pipeline) |

### EarthDial Verdict

**KEEP as PRIMARY VLM** — It's the only model that fits in 4 GB VRAM (with4-bit quantization) and provides real RS VLM capabilities (captioning, VQA, grounding, classification). Requires cloning the GitHub repo and installing the `earthdial` package.

---

## 4. GeoChat Validation

| Field | Verified Value |
|-------|---------------|
| **Repository** | https://github.com/mbzuai-oryx/GeoChat |
| **HuggingFace** | `MBZUAI/geochat-7B` |
| **Exists on HuggingFace** | ✅ YES |
| **Architecture** | GeoChatLlamaForCausalLM (LLaVA-1.5 lineage) |
| **Parameters** | ~7.1B (verified from `total_size`: 14.13 GB FP16) |
| **Files** | 2 pytorch_model.bin shards + config + tokenizer |
| **Custom code required** | ✅ YES — GeoChatLlamaForCausalLM is not in standard transformers |
| **License** | Apache 2.0 (LLaVA/Vicuna lineage) |
| **VRAM FP16** | ~14 GB ❌ |
| **VRAM4-bit** | ~3.5-4 GB model weights ❌ too tight with KV cache |
| **VRAM (realistic4-bit inference)** | ~5-6 GB with activations and KV cache ❌ DOES NOT FIT |
| **Dependencies** | transformers, torch, flash-attn (recommended) |
| **Supports SAR** | ❌ NO — trained on optical RS only |
| **Supports optical RS** | ✅ YES — captioning, VQA, grounding, classification |

### GeoChat Verdict

**DROP as primary model. KEEP as documented FALLBACK only.** The 7B model simply cannot run on a 4 GB VRAM laptop. Even4-bit quantization produces a model that needs ~5-6 GB with KV cache overhead. Would require cloud GPU (A100) to function.

---

## 5. SARDet-100K Validation

| Field | Verified Value |
|-------|---------------|
| **Repository** | https://github.com/zcablii/SARDet_100K |
| **HuggingFace** | NOT on HuggingFace |
| **Paper** | NeurIPS 2024 Spotlight |
| **Weight download** | OneDrive + Baidu Disk (ZIP = 20 GB for all weights) |
| **Smallest useful weight** | `van_t_sar_wavelet_epoch_100.pth` = 66 MB |
| **Recommended weight** | `fg_frcnn_dota_pretrain_sar_r50_wavelet` = 287 MB (Faster-RCNN + ResNet50) |
| **Best weight** | `gfl_r50_denodet_sardet` = 2.2 GB (too large for this machine) |
| **License** | CC BY-NC 4.0 (NonCommercial only) |
| **Framework** | mmdet + mmcv + mmengine (OpenMMLab stack) |
| **PyTorch version** | Requires PyTorch 2.0.1 with CUDA 11.8 (⚠️ may conflict with newer PyTorch) |
| **VRAM (Faster-RCNN R50)** | ~0.5-1 GB ✅ FITS easily |
| **Categories detected** | 6: ship, aircraft, car, bridge, harbour, tank |
| **Input** | SAR images (any resolution, patched to 512×512) |
| **Output** | Bounding boxes + class labels + confidence scores |
| **Accepts text queries** | ❌ NO — it's a standard object detector |
| **Can do general SAR understanding** | ❌ NO — only detects 6 fixed categories |
| **Can do VQA** | ❌ NO |
| **OneDrive link accessible** | ✅ Blog post (tech.marksblogg.com) confirms successful download |

### SARDet-100K Verdict

**KEEP as SAR detection tool.** It's the only publicly available SAR detection model with verified accessible weights. But it is NOT a VLM. It detects objects. You must wrap it with template-based captioning + LLM summarization to produce natural language output.

**Dependency conflict risk**: The mmdet stack (mmengine 0.8.4, mmcv 2.0.1, mmdet 3.1.0) is heavy and may conflict with transformers/EarthDial. Consider using a separate conda environment for SARDet-100K.

---

## 6. DeltaVLM Validation

| Field | Verified Value |
|-------|---------------|
| **Repository** | https://github.com/hanlinwu/DeltaVLM |
| **HuggingFace** | `hlwu/DeltaVLM` (correct ID, not `hanlinwu/DeltaVLM`) |
| **Also** | `dengpeipei/deltavlm_weights` (alternative mirror) |
| **Exists on HuggingFace** | ✅ YES |
| **Checkpoint file** | `checkpoint_best.pth` |
| **Config** | `configs/evaluate.yaml` |
| **Backbone** | **Vicuna-7B** (verified from config: `model_type: vicuna7b`, `arch: instruct_vicuna7b`) |
| **Parameters** | ~7B |
| **License** | BSD-3-Clause |
| **VRAM FP16** | ~14 GB ❌ |
| **VRAM4-bit** | ~4 GB model ❌ (too tight with KV cache) |
| **Input** | Bi-temporal optical image pairs |
| **Output** | Change description text |
| **Supports SAR** | ❌ NO — optical only |
| **Dependencies** | lavis (Salesforce), torch, transformers |

### DeltaVLM Verdict

**DROP for this machine.** The Vicuna-7B backbone requires ~5-6 GB VRAM minimum (4-bit + KV cache), which exceeds the 4 GB budget. Would need cloud GPU.

**Alternative needed for change detection**: Consider a smaller model or rule-based approach for the prototype.

---

## 7. ChangeChat Validation

| Field | Verified Value |
|-------|---------------|
| **Repository** | https://github.com/hanlinwu/ChangeChat |
| **HuggingFace** | NOT found on HuggingFace search |
| **Checkpoint availability** | ⚠️ README says "coming soon" — **NOT YET RELEASED** |
| **Paper** | Sep 2024 (arXiv 2409.08582) |
| **Backbone** | Also Vicuna-7B (same as DeltaVLM) |
| **VRAM** | Same as DeltaVLM — ~14 GB FP16 |

### ChangeChat Verdict

**DROP.** Weights not publicly available yet. Even if released, same 7B VRAM problem as DeltaVLM.

---

## 8. GroundingDINO + SAM Validation

### GroundingDINO

| Field | Verified Value |
|-------|---------------|
| **HuggingFace** | `IDEA-Research/grounding-dino-tiny` (✅) and `IDEA-Research/grounding-dino-base` (✅) |
| **Architecture** | GroundingDinoForObjectDetection (BERT text encoder + SwinT vision encoder) |
| **Tiny variant params** | ~340M (estimated from config) |
| **Tiny VRAM** | ~1.3 GB FP32, ~0.7 GB FP16 ✅ FITS |
| **Base variant VRAM** | ~680M params, ~2.7 GB FP32 ⚠️ tight |
| **License** | Apache 2.0 |
| **Input** | Image + text prompt (e.g., "ship . aircraft . car") |
| **Output** | Bounding boxes + confidence for text-referenced objects |
| **Custom code** | YES — GroundingDinoForObjectDetection not in standard transformers |

### SAM (Segment Anything)

| Field | Verified Value |
|-------|---------------|
| **HuggingFace** | `ybelkada/segment-anything` (✅) |
| **Available checkpoints** | sam_vit_b_01ec64.pth, sam_vit_h_4b8939.pth, sam_vit_l_0b3195.pth, mobile_sam.pth |
| **MobileSAM** | ~40 MB ✅ extremely lightweight |
| **SAM ViT-B** | ~375M params, ~1.5 GB FP32, ~750 MB FP16 ✅ FITS |
| **SAM ViT-H** | ~636M params, ~2.5 GB FP32 ⚠️ tight |
| **License** | Apache 2.0 (Meta) |
| **Input** | Image + optional bounding boxes / points |
| **Output** | Segmentation masks |

### GroundingDINO + SAM Verdict

**KEEP for optional grounding feature.** Both models fit comfortably in VRAM:
- GroundingDINO Tiny: ~1.3 GB
- MobileSAM: ~40 MB
- **Total together**: ~1.35 GB ✅ fits alongside EarthDIAL at4-bit

However: Do NOT load them simultaneously with the VLM. Load on-demand when the user requests grounding/segmentation.

**These are NOT essential for the core prototype.** They're a nice-to-have for the demo. Implement after core VLM pipeline works.

---

## 9. Router Options

### Option A: Local LLM Router

| Model | HuggingFace | Params | VRAM4-bit | Verdict |
|-------|------------|--------|-----------|---------|
| Qwen2-0.5B | `Qwen/Qwen2-0.5B` | 0.5B | ~0.5 GB | ✅ FITS |
| Qwen2-1.5B | `Qwen/Qwen2-1.5B` | 1.5B | ~1 GB | ✅ FITS |
| Phi-3.5-mini | `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~2.5 GB | ⚠️ TIGHT (can't coexist with VLM) |

### Option B: API-based Router

| Service | Cost | Latency | Reliability |
|---------|------|---------|-------------|
| Google Gemini Flash (free tier) | Free (15 RPM) | ~1-2s | ⚠️ needs internet |
| OpenAI GPT-4o-mini | ~$0.15/1M tokens | ~1s | ✅ reliable |
| Local keyword matching | Free | ~0ms | ✅ always works |

### Router Verdict

**RECOMMENDATION**: For the hackathon prototype, use **keyword/intent matching** (Option C) as the primary router. It's deterministic, zero-latency, zero-VRAM, and always works. Optionally add Gemini Flash API as an upgrade path.

Do NOT spend VRAM on a local LLM router when it competes with EarthDial for the 4 GB budget.

---

## 10. Dependency Conflicts

### Critical Conflict: mmdet vs transformers

| Model | Needs | Version |
|-------|-------|---------|
| EarthDial | transformers (recent), torch | transformers ≥4.37 |
| GeoChat | transformers, flash-attn | transformers ≥4.36 |
| SARDet-100K | mmengine, mmcv, mmdet | mmengine=0.8.4, mmcv=2.0.1, mmdet=3.1.0, **torch=2.0.1** |

**SARDet-100K pins PyTorch 2.0.1**. EarthDial and GeoChat need newer PyTorch. This is a real conflict.

### Mitigation Options

1. **Two separate conda environments**: One for SARDet-100K (PyTorch 2.0.1 + mmdet), one for EarthDial (latest PyTorch + transformers). Call SARDet-100K via subprocess/REST API.
2. **One environment, latest PyTorch**: Risk mmdet incompatibility with PyTorch 2.13. Some mmdet versions work with newer PyTorch but it's not guaranteed.
3. **Use SARDet-100K pre-inference**: Run SARDet-100K once, save results as JSON, load in EarthDial environment.

**RECOMMENDATION**: Option 1 (two environments) or Option 3 (pre-compute SAR detections).

---

## 11. VRAM / Resource Budget

### Scenario: Core Pipeline (Optical VLM only)

| Component | VRAM | Loaded? |
|-----------|------|---------|
| EarthDial 4B (4-bit) | ~2.5 GB | ✅ |
| System overhead | ~0.5 GB | — |
| **Total** | **~3.0 GB** | **✅ FITS** |
| Headroom | ~1.0 GB | for KV cache |

### Scenario: Optical VLM + Grounding

| Component | VRAM | Loaded? |
|-----------|------|---------|
| EarthDial 4B (4-bit) | ~2.5 GB | ✅ |
| GroundingDINO Tiny | ~1.3 GB | on-demand |
| **Total** | **~3.8 GB** | **⚠️ TIGHT** |

**Must unload EarthDial before loading GroundingDINO, or vice versa.** Can't run both simultaneously.

### Scenario: SAR Detection

| Component | VRAM | Loaded? |
|-----------|------|---------|
| SARDet-100K R50 | ~0.5 GB | ✅ |
| **Total** | **~1.0 GB** | **✅ FITS easily** |

### Scenario: Change Detection

| Component | VRAM | Loaded? |
|-----------|------|---------|
| DeltaVLM (7B, 4-bit) | ~4+ GB | ❌ DOES NOT FIT |
| **Total** | **N/A** | **❌ BLOCKED** |

### VRAM Strategy: Lazy Loading

```
User uploads image →
  If SAR:
    Load SARDet-100K (0.5 GB) → detect → UNLOAD
    Load EarthDial (2.5 GB) → describe detections → UNLOAD
  If Optical:
    Load EarthDial (2.5 GB) → caption/VQA/ground → UNLOAD
  If Grounding requested:
    Load GroundingDINO (1.3 GB) → detect → UNLOAD
    Load MobileSAM (0.04 GB) → segment → UNLOAD
```

**Never load more than one model at a time.**

---

## 12. Recommended Model Stack

### Core Stack (MUST HAVE)

| Role | Model | Why |
|------|-------|-----|
| **Optical VLM** | EarthDial 4B RGB (4-bit) | Only VLM that fits in 4 GB VRAM |
| **SAR Detection** | SARDet-100K (Faster-RCNN R50) | Only available SAR detector |
| **Intent Router** | Keyword matching + rules | Zero VRAM, zero latency, always works |

### Optional Stack (NICE TO HAVE)

| Role | Model | Why |
|------|-------|-----|
| **Text-grounded detection** | GroundingDINO Tiny | User can ask "find ships" with boxes |
| **Segmentation** | MobileSAM | GroundingDINO boxes → masks |
| **API Router upgrade** | Gemini Flash API | Smarter intent classification if online |

### Dropped Models

| Model | Reason |
|-------|--------|
| GeoChat 7B | Too large (7B) for 4 GB VRAM |
| DeltaVLM | Too large (7B Vicuna) for 4 GB VRAM |
| ChangeChat | Weights not released yet |
| SARCLIP | Not a VLM — only computes similarity scores |
| SAR-KnowLIP | Too new, unverified, research-grade |

---

## 13. KEEP / FALLBACK / DROP Table

| Model | Status | Reason |
|-------|--------|--------|
| **EarthDial 4B RGB** | 🟢 **KEEP** (PRIMARY) | Fits in4-bit, real RS VLM capabilities, verified on HuggingFace |
| **SARDet-100K** | 🟡 **KEEP** (SAR tool) | Verified accessible, but heavy deps, non-commercial license, not a VLM |
| **GroundingDINO Tiny** | 🟡 **KEEP** (OPTIONAL) | Fits in VRAM, useful for demo, implement after core works |
| **MobileSAM** | 🟡 **KEEP** (OPTIONAL) | Tiny (40 MB), pairs with GroundingDINO |
| **GeoChat 7B** | 🟠 **FALLBACK** | Only if EarthDial fails or cloud GPU is available |
| **DeltaVLM** | 🔴 **DROP** | 7B Vicuna backbone doesn't fit. No smaller variant available. |
| **ChangeChat** | 🔴 **DROP** | Weights not released. Same7B problem. |
| **SARCLIP** | 🔴 **DROP** | Not a VLM. Only computes embedding similarity. |
| **Qwen2-0.5B** | 🟡 **KEEP** (optional) | Could be smart router if VRAM allows |

---

## 14. Critical Risks

### Risk 1: EarthDial Custom Code May Not Load Cleanly
- EarthDial requires `from earthdial.model.internvl_chat import InternVLChatModel`
- This means cloning the EarthDial repo, installing it, and using their custom class
- **Mitigation**: Test loading in Loop 1 before building anything around it

### Risk 2: SARDet-100K Dependency Hell
- mmdet 3.1.0 requires PyTorch 2.0.1
- Current PyTorch is 2.13.0
- mmdet + transformers may conflict
- **Mitigation**: Use separate conda environments or run SARDet-100K via subprocess

### Risk 3: 4-bit Quantization Quality
- EarthDial was trained in bfloat16
- Loading in4-bit (NF4) may degrade output quality significantly
- **Mitigation**: Test output quality in Loop 1. If bad, fall back to 8-bit (4 GB exactly) or CPU offloading

### Risk 4: No Change Detection Model Available
- Both DeltaVLM and ChangeChat are7B models that don't fit
- No smaller change detection VLM exists publicly
- **Mitigation**: Implement change detection as: (a) pixel-level differencing for simple cases, or (b) use EarthDial to describe two images separately, then use rules/LLM to diff the descriptions

### Risk 5: Demo Failure on 4 GB VRAM
- If any model OOMs during demo, the presentation fails
- **Mitigation**: Pre-compute results for demo scenarios. Have fallback cached outputs.

### Risk 6: SARDet-100K Non-Commercial License
- CC BY-NC 4.0 — cannot be used commercially
- For SIH prototype this is acceptable, but state it in documentation
- **Mitigation**: Disclose license. For commercial deployment, would need alternative.

---

## 15. GO / NO-GO Decision

### GO Criteria

| Criterion | Status |
|-----------|--------|
| At least one VLM runs on this machine | ✅ EarthDial 4B (4-bit) should fit |
| SAR detection possible | ✅ SARDet-100K R50 available |
| Weights are downloadable | ✅ Verified for EarthDial + SARDet-100K |
| Python/PyTorch compatible | ⚠️ Need CUDA PyTorch install |
| Disk space sufficient | ✅ 253 GB free |
| RAM sufficient | ✅ 16 GB (enough for CPU offloading if needed) |

### Conditions for GO

1. **Loop 1 MUST verify**: EarthDial actually loads and runs inference in4-bit on this machine
2. **Loop 1 MUST verify**: SARDet-100K weights download from OneDrive and run
3. If EarthDial fails in4-bit: Fall back to loading with `device_map="auto"` + CPU offloading (slower but works)
4. If both fail: Pivot to API-only approach (call Gemini/GPT-4o for VLM, no local model)

### **DECISION: CONDITIONAL GO** ✅

The project is feasible on this machine **with the simplified model stack**:
- EarthDial 4B (optical VLM)
- SARDet-100K R50 (SAR detection)
- Keyword router
- Optional: GroundingDINO + MobileSAM

It is NOT feasible with the full original plan (GeoChat, DeltaVLM, evidence-level fusion, etc.).

---

### FINAL RECOMMENDATION

**Use this exact stack for the prototype:**

```
1. EarthDial 4B RGB     — optical image understanding (4-bit quantized)
2. SARDet-100K R50      — SAR object detection (Faster-RCNN + ResNet50)
3. Keyword router       — intent classification (zero VRAM)
4. GroundingDINO Tiny   — optional, text-grounded detection
5. MobileSAM            — optional, segmentation masks
```

**Do NOT attempt:**
- Running GeoChat (too large)
- Running DeltaVLM/ChangeChat (too large, ChangeChat weights unavailable)
- Cross-modal SAR+optical fusion (no model supports this without training)
- "Evidence-level fusion" (just aggregate tool outputs honestly)
- Running multiple models simultaneously (lazy load one at a time)

---

### NEXT LOOP — Loop 1 Plan

Loop 1 should implement **"Single Image Inference Test"** — prove that the core pipeline works end-to-end on ONE image:

1. **Install CUDA PyTorch** on this machine
2. **Clone EarthDial repo**, install the package
3. **Download EarthDial 4B RGB weights** from HuggingFace
4. **Test EarthDial inference** on a sample optical RS image:
   - Caption generation
   - VQA ("what objects are in this image?")
   - Grounding ("where are the buildings?")
5. **Download SARDet-100K R50 weights** from OneDrive
6. **Test SARDet-100K inference** on a sample SAR image:
   - Object detection (ships, aircraft, etc.)
   - Verify bounding box output
7. **Test keyword router**:
   - Classify 10 sample queries into intent categories
   - Verify correct routing

**Do NOT build:**
- Web UI
- Backend API
- Multi-image pipeline
- Change detection
- Grounding (save for Loop 2)

**Success criteria for Loop 1:**
- EarthDial produces sensible caption for an optical RS image ✅
- SARDet-100K detects objects in a SAR image ✅
- Keyword router classifies sample queries correctly ✅
- All three run on this machine without OOM ✅
