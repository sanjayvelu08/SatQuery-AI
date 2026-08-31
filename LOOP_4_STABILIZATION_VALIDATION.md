# LOOP 4 — Stabilization & Demo Quality Report

> **Date**: August 31, 2026
> **Project**: SatQuery AI (SIH26167 — ISRO)
> **Machine**: RTX 3050 Laptop (4 GB VRAM), 16 GB RAM
> **Purpose**: Stabilize prototype, improve demo quality, clean up code

---

## 1. Changes Made

### 1.1 Router Improvements (`router.py`)

| Change | Before | After |
|--------|--------|-------|
| Intent rule ordering | caption → vqa → detect → ... | change → sar → detect → grounding → classification → caption → vqa |
| "What type of scene?" | Misclassified as VQA | ✅ Correctly classified as classification |
| "Is this urban?" | Misclassified as general | ✅ Correctly classified as VQA |
| "Is this rural?" | Misclassified as general | ✅ Correctly classified as VQA |
| "What can you see?" | Misclassified as VQA | ✅ Correctly classified as caption |
| Prompt for VQA | Raw user question | RS expert context prepended |
| Prompt for caption | Generic template | Detailed RS analyst template with structured output |
| Prompt for detect/grounding | Generic template | Expert RS analyst with bounding box format |
| Prompt for classification | Basic template | Detailed RS analyst with land cover categories |
| Prompt for general | Raw question | RS expert context + user question |
| Unsupported SAR message | Brief | Detailed explanation + suggestion |
| Unsupported change message | Brief | Detailed explanation + suggestion |

**Key fix**: More-specific intents (change, sar, detect, grounding, classification) are now checked before less-specific ones (caption, vqa). This prevents "What type of scene?" from matching "what is the" in VQA keywords.

### 1.2 Demo Improvements (`demos.py`)

| Change | Before | After |
|--------|--------|-------|
| Number of demos | 3 | 5 |
| Demo 1 | Basic description | Detailed RS analysis with ISRO relevance |
| Demo 2 | Basic VQA | Structured assessment with land use table |
| Demo 3 | Basic detection | Expert detection with confidence scores |
| NEW Demo 4 | — | Scene classification with land cover breakdown |
| NEW Demo 5 | — | Water body/terrain analysis |
| Answer format | Plain text | Markdown with headers, tables, bullet points |
| Technical depth | Surface level | ISRO-relevant (FASAL, PMFBY mentions) |

### 1.3 Gradio UI Improvements (`app.py`)

| Feature | Before | After |
|---------|--------|-------|
| Image preview | None | Shows uploaded image during/after analysis |
| Intent display | Plain text badge | Rich badge with emoji + name |
| Answer formatting | Raw Markdown | Structured with intent header + bounding box count |
| Query history | None | Last 5 queries with intent badges |
| Status display | Plain text | Emoji-prefixed status |
| Clear button | None | Clears all fields and history |
| Error messages | Generic | Specific and helpful |
| Loading feedback | Gradio spinner | Gradio spinner + status update |

### 1.4 Pipeline Improvements (`pipeline.py`)

| Change | Before | After |
|--------|--------|-------|
| History tracking | None | QueryHistory class (deque, max 5) |
| History in PipelineResult | No | Yes, stored on each run |
| Error handling | Basic try/catch | Structured PipelineResult for errors |

### 1.5 New Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Exact pinned dependencies |
| `README.md` | Project overview, architecture, run instructions |

---

## 2. Test Results

### 2.1 Router Regression: 15/15 PASS ✅

All 15 test queries correctly classified:

| Intent | Tests | All Pass |
|--------|-------|----------|
| caption | 3 | ✅ |
| vqa | 4 | ✅ |
| detect | 3 | ✅ |
| classification | 2 | ✅ |
| change | 1 | ✅ (unsupported) |
| sar | 1 | ✅ (unsupported) |
| grounding | 1 | ✅ |

### 2.2 Demo Scenarios: 5/5 PASS ✅

| Demo | Intent | Answer Length | Status |
|------|--------|---------------|--------|
| Agricultural Landscape | caption | 1098 chars | ✅ |
| Urban Area Assessment | vqa | 900 chars | ✅ |
| Infrastructure Detection | detect | 877 chars | ✅ |
| Scene Classification | classification | 810 chars | ✅ |
| Water Body Analysis | caption | 836 chars | ✅ |

### 2.3 App Functionality: 5/5 PASS ✅

| Test | Input | Expected | Result |
|------|-------|----------|--------|
| Demo mode | Agricultural demo | Instant answer | ✅ |
| Demo history | Classification demo | History accumulates | ✅ (2 entries) |
| No image | None + query | Graceful error | ✅ |
| No query | Image + empty | Graceful error | ✅ |
| SAR unsupported | SAR query | Unsupported message | ✅ |

### 2.4 Live VLM: 1/1 PASS ✅

| Test | Query | Answer | Time |
|------|-------|--------|------|
| VQA | "Is this an urban or rural area?" | "rural" | 86.4s |

**Note**: The improved RS expert prompt produces more contextual answers. The model correctly identifies the agricultural image as "rural".

### 2.5 Server: ✅ RUNNING

| Property | Value |
|----------|-------|
| URL | http://127.0.0.1:7860 |
| PID | 29464 |
| Status | Running |

---

## 3. Dependency Status

| Package | Version | Conflict | Resolution |
|---------|---------|----------|------------|
| transformers | 4.37.2 | Required by EarthDial | ✅ Pinned |
| huggingface-hub | 0.23.5 | Gradio wants ≥1.16 | ⚠️ Acceptable (Hub features unused) |
| tokenizers | 0.15.1 | Required by EarthDial | ✅ Pinned |
| gradio | 6.26.0 | Uses newer Hub | ⚠️ Works for UI |
| peft | 0.7.0 | Required by EarthDial | ✅ Pinned |
| bitsandbytes | 0.50.2 | — | ✅ Compatible |

**Summary**: The huggingface-hub version conflict is cosmetic — Gradio's HuggingFace Hub integration is unused. All core functionality (VLM inference, UI) works correctly.

---

## 4. Files Modified/Created in Loop 4

| File | Action | Lines Changed |
|------|--------|---------------|
| `satquery/router.py` | Modified | Reordered rules, added keywords, improved prompts |
| `satquery/demos.py` | Modified | Added 2 demos, improved all answers |
| `satquery/pipeline.py` | Modified | Added QueryHistory class |
| `satquery/app.py` | Modified | Improved UI, added history, image preview |
| `requirements.txt` | Created | New file |
| `README.md` | Created | New file |

---

## 5. Known Remaining Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| Live VLM slow (50-260s) | Medium | Users wait for results |
| huggingface-hub version conflict | Low | Gradio Hub features broken (unused) |
| No SAR support | Medium | SAR queries return "coming soon" |
| No change detection | Medium | Change queries return "coming soon" |
| EarthDial sys.path workaround | Low | Must use sys.path.insert |

---

## 6. LOOP_4 Verdict

### 🟢 GO for Loop 5

All stabilization tasks completed:
- ✅ Router keywords fixed (15/15 tests pass)
- ✅ Prompt templates improved (RS expert context)
- ✅ UI improved (image preview, intent badges, history)
- ✅ 5 demo scenarios (up from 3)
- ✅ Unsupported intent handling improved
- ✅ Dependencies documented (requirements.txt)
- ✅ README added with run instructions
- ✅ All regression tests pass
- ✅ Server running and functional

### Loop 5 should focus on:

1. **SARDet-100K integration** (isolated conda env or subprocess)
2. **GroundingDINO integration** (optional, for richer bounding boxes)
3. **Pre-computed demo images** (download real Sentinel-2 scenes)
4. **Performance optimization** (faster inference or better loading UX)
5. **Deployment prep** (Docker, cloud deployment, final README polish)
6. **Presentation script** (judge walkthrough)
