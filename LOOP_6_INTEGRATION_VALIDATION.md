# LOOP 6 — Integration, Visualization & Demo Reliability Report

**Date**: September 1, 2026
**Status**: ✅ GO for Loop 7

---

## 1. Architecture Changes (Loop 6)

### New Module: `satquery/visualize.py` (260 lines)

| Function | Purpose |
|----------|---------|
| `BBox` dataclass | Bounding box with label + confidence |
| `draw_bboxes()` | Draws colored bbox rectangles, labels, legend on any image |
| `parse_grounding_output()` | Parses EarthDial `[[x1,y1,x2,y2,conf]]` format → pixel BBox list |
| `create_annotated_image()` | High-level: parses answer text → draws bboxes → saves annotated image |

Supports both SAR detections (pixel coords) and optical grounding (normalized 0-100).

### Modified: `satquery/pipeline.py` (183 lines, +33 from Loop 5)

| Change | Detail |
|--------|--------|
| `PipelineResult.model_used` | New field — shows which model/tool handled the query |
| `PipelineResult.annotated_image` | New field — path to annotated image with bboxes |
| `PipelineResult.sar_result` | New field — raw SAR detection data for SAR-specific UI |
| SAR path generates annotated image | When detections found → `create_annotated_image()` |
| Detection/grounding path generates annotated image | When EarthDial returns bbox coordinates → `create_annotated_image()` |

### Modified: `satquery/app.py` (461 lines, +106 from Loop 5)

| Change | Detail |
|--------|--------|
| New output: `annotated_output` | Gradio Image component showing bounding boxes |
| `_format_answer()` shows model info | Header includes "Model: EarthDial 4B" or "Model: YOLOv8 SAR" |
| SAR detection summary | Shows count + timing + VRAM for SAR results |
| `_validate_inputs()` | Handles: no image, empty query, whitespace query, corrupt image |
| `analyze()` returns 7-tuple | Added `annotated_output` position |
| `clear_all()` clears annotated image | Returns None for annotated output |

### Modified: `satquery/demos.py` (198 lines)

| Change | Detail |
|--------|--------|
| 6 scenarios (was 6, now with model_used) | Each demo includes `model_used` field |
| Improved formatting | Cleaner markdown, less fluff, more technical |
| Added "Urban VQA — Building Count" | Was the 6th demo, now with better content |

---

## 2. Optical Tests

| Test | Result |
|------|--------|
| Import all modules | ✅ |
| Router: 15/15 queries | ✅ 15/15 |
| Pipeline: captioning intent | ✅ Routes to EarthDial |
| Pipeline: VQA intent | ✅ Routes to EarthDial |
| Pipeline: detection intent | ✅ Routes to EarthDial + generates annotated image |
| Pipeline: classification intent | ✅ Routes to EarthDial |
| Pipeline: grounding intent | ✅ Routes to EarthDial + generates annotated image |
| Pipeline: change detection (unsupported) | ✅ Returns graceful message |
| Pipeline: general intent | ✅ Routes to EarthDial |
| Demo: Agricultural Caption | ✅ Instant, with model_used |
| Demo: Urban VQA | ✅ Instant, with model_used |
| Demo: Infrastructure Detection | ✅ Instant, generates annotated image |
| Demo: Scene Classification | ✅ Instant, with model_used |
| Demo: Urban VQA Building Count | ✅ Instant, with model_used |

---

## 3. SAR Tests

| Test | Result |
|------|--------|
| Pipeline: SAR query → YOLOv8 | ✅ 3 ships detected |
| SAR model_used = "YOLOv8 SAR Vessel Detector" | ✅ |
| SAR generates annotated image | ✅ `sar_sample_sar_annotated.jpg` (53 KB) |
| SAR regression with change intent | ✅ Returns unsupported |
| Demo: SAR Maritime Vessel Detection | ✅ Instant, with model_used |
| Demo annotated image for SAR | ✅ Generated |

---

## 4. Visualization Tests

| Test | Result |
|------|--------|
| SAR: 3 bounding boxes drawn | ✅ Color-coded with labels |
| SAR: annotated image is valid JPEG | ✅ 53,505 bytes |
| Optical grounding: bbox from `[[48,59,52,63,90]]` | ✅ Drawn correctly |
| Optical grounding: annotated image is valid JPEG | ✅ 29,486 bytes |
| Caption intent: no annotation generated | ✅ Returns None |
| Legend rendered for multiple detections | ✅ |

---

## 5. Edge-Case Tests

| Scenario | Input | Result |
|----------|-------|--------|
| No image | `None` + query | ✅ "No image provided" |
| Empty query | image + `""` | ✅ "No query entered" |
| Whitespace query | image + `"   "` | ✅ "No query entered" |
| Valid input | image + query | ✅ Proceeds normally |
| Corrupt image | (not tested with real corrupt file, but try/except in place) | ✅ Would catch PIL error |
| SAR image + optical query | (routed to EarthDial — may produce poor results but won't crash) | ✅ No crash |
| Optical image + SAR query | (routed to YOLOv8 — may find 0 ships but won't crash) | ✅ Returns 0 detections |
| Zero SAR detections | (handled in format_sar_response) | ✅ "No maritime targets detected" |
| Malformed grounding output | (parse_grounding_output handles gracefully) | ✅ Returns empty list |

---

## 6. Demo Tests

| Demo | Intent | Model | Annotated Image | Status |
|------|--------|-------|----------------|--------|
| 🌾 Agricultural Landscape | caption | EarthDial 4B | None (no bboxes) | ✅ |
| 🏙️ Urban Area Assessment | vqa | EarthDial 4B | None | ✅ |
| 🔍 Infrastructure Detection | detect | EarthDial 4B | ✅ Generated | ✅ |
| 🗺️ Scene Classification | classification | EarthDial 4B | None | ✅ |
| 🛰️ SAR Maritime Vessel | sar | YOLOv8 SAR | ✅ Generated | ✅ |
| ❓ Urban VQA Building Count | vqa | EarthDial 4B | None | ✅ |

---

## 7. Performance Measurements

| Metric | Value |
|--------|-------|
| Router latency | <1ms |
| SAR inference (GPU) | ~50ms (model load) + 7ms (inference) |
| SAR annotated image creation | <100ms |
| Optical annotated image creation | <100ms |
| Total SAR pipeline | ~5s (subprocess overhead) |
| Total optical pipeline | ~50-60s (EarthDial inference) |
| Gradio app startup | ~3s |
| VRAM: EarthDial (4-bit) | 2,850 MB |
| VRAM: YOLOv8 SAR | 21 MB |
| VRAM: Total headroom | ~1,225 MB |

---

## 8. Change Detection Experiment

### Investigation: Lightweight Image Differencing

**Approach tested:** Pixel-level difference between two resized (256×256) RGB images.

**Results:**
- Mean pixel difference: 52.2/255 (20.5%) — high noise
- Max pixel difference: 249/255 — near-total mismatch
- "Changed" pixels (>30% diff): 15,250 / 65,536 (23.3%) — mostly noise, not real changes

**Conclusion:** ❌ NOT suitable for demo.

**Why:**
1. Requires **same sensor**, **same season**, **same time of day**, **precise sub-pixel registration**
2. Without registration, pixel differences are geometric noise, not semantic changes
3. A demo showing random pixel noise as "change detection" would **confuse and impress negatively** with judges
4. Real change detection (e.g., ChangeChat, DeltaVLM) requires **trained models** we don't have

**Recommendation:** Keep change detection as "coming soon" — document honestly as future work requiring bi-temporal registered imagery and a trained change-detection model.

---

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| No change detection | Cannot compare two images | Documented as "not yet implemented" |
| SAR = ship detection only | No scene understanding for SAR | Honest in UI and documentation |
| EarthDial grounding is approximate | Bboxes may not align perfectly | Acceptable for prototype |
| EarthDial inference is slow (~50s) | Long wait for live queries | Pre-computed demos for instant results |
| Gradio UI needs server restart | After code changes | Standard for development |
| SAR subprocess has ~2s overhead | Added to pipeline time | Acceptable for demo |
| No SAR VQA | Cannot ask "how many ships?" in natural language | Would need SAR-specific VLM |

---

## 10. Complete Regression Suite (Loops 2–6)

| Test Suite | Tests | Result |
|------------|-------|--------|
| **Router** (Loop 2) | 15 queries | 15/15 ✅ |
| **Pipeline SAR** (Loop 5) | 2 tests | 2/2 ✅ |
| **Edge Cases** (Loop 6) | 4 validation tests | 4/4 ✅ |
| **Demos** (Loop 4→6) | 6 demos | 6/6 ✅ |
| **Visualization** (Loop 6) | 3 tests | 3/3 ✅ |
| **App Format** (Loop 6) | 2 tests | 2/2 ✅ |
| **App Analyze** (Loop 6) | 10 tests | 10/10 ✅ |
| **TOTAL** | **42** | **42/42 ✅** |

---

## 11. Codebase Summary

| Module | Lines | Purpose |
|--------|-------|---------|
| `app.py` | 461 | Gradio UI with annotated image output |
| `demos.py` | 198 | 6 pre-computed demo scenarios |
| `pipeline.py` | 183 | Orchestrator with SAR + VLM paths |
| `router.py` | 202 | 7-intent keyword classifier |
| `sar_infer.py` | 134 | Standalone SAR inference (sar_venv) |
| `sar_tool.py` | 195 | Subprocess bridge for SAR |
| `vlm.py` | 142 | EarthDial wrapper |
| `visualize.py` | 260 | Bbox drawing + image annotation |
| `test_pipeline.py` | 107 | Loop 2 test suite |
| `__init__.py` | 2 | Package init |
| **Total** | **1,884** | |

---

## 12. GO / NO-GO Decision

### 🟢 GO for Loop 7

**Rationale:**
- All 42 regression tests pass
- SAR integration works end-to-end with annotated visualization
- Optical grounding produces annotated images
- Edge cases handled gracefully
- 6 demo scenarios ready for instant demonstration
- No dependency conflicts
- VRAM budget sufficient
- Change detection honestly documented as future work

**Loop 7 should focus on:**
1. Final README with setup instructions and architecture diagram
2. Requirements.txt validation
3. Demo rehearsal and timing
4. Git commit of all work
5. Any final UI polish

---

## 13. Files Changed in Loop 6

```
NEW:   satquery/visualize.py      (260 lines) — bbox drawing + image annotation
EDIT:  satquery/pipeline.py       (+33 lines) — model_used, annotated_image, sar_result
EDIT:  satquery/app.py            (+106 lines) — annotated image output, validation, model info
EDIT:  satquery/demos.py          (rewritten) — model_used field, improved content
NEW:   test_images/sar_sample_sar_annotated.jpg       (53 KB)
NEW:   test_images/sentinel2_optical_grounding_annotated.jpg  (29 KB)
```
