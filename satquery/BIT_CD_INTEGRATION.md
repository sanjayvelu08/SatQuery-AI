# BIT-CD Change Detection Integration

## Overview

SatQuery now supports **bi-temporal change detection** using the BIT (Bitemporal Image Transformer) model. Users can upload two satellite images (T1/Before and T2/After) and ask what changed between them.

## Architecture

```
User uploads T1 + T2 + change query
        ↓
   Router (intent = "change")
        ↓
   Pipeline validates two images
        ↓
   BIT-CD inference (~30-80ms per pair)
        ↓
   Binary change mask
        ↓
   Connected components → bounding boxes
        ↓
   Overlay visualization
        ↓
   Structured response with regions + timing
```

## Model

| Property | Value |
|----------|-------|
| Model | BIT (Bitemporal Image Transformer) |
| Architecture | BASE_Transformer (ResNet-18 + Transformer decoder) |
| Parameters | 11,943,754 |
| Checkpoint | `best_ckpt.pt` (57.3 MB) |
| Training data | LEVIR-CD (building change detection) |
| Input | Two RGB satellite images, resized to 256×256 |
| Output | 2-class logits → argmax → binary change mask |
| VRAM | ~81 MB (peak) |
| Inference | ~30-80ms (after model warmup) |

### Prototype Validation Result

Isolated validation on LEVIR-CD (7 samples):

| Metric | Value |
|--------|-------|
| F1 | 0.9263 |
| IoU | 0.8640 |
| Precision | 0.9039 |
| Recall | 0.9515 |

**Note:** This is a prototype validation result on a small subset, NOT general accuracy. The model was trained on Chinese building change data and generalization to other change types is untested.

## Files

| File | Purpose |
|------|---------|
| `satquery/bit_tool.py` | BIT-CD model loading, inference, region extraction, visualization |
| `satquery/router.py` | Change intent now routes as `supported=True` |
| `satquery/pipeline.py` | `run()` accepts `image_t2_path`; change detection workflow |
| `satquery/app.py` | T2 image upload widget in Gradio UI |
| `change_output/` | Generated masks, overlays, bboxes (runtime) |

## Usage

### Via UI
1. Upload a **Before (T1)** satellite image
2. Upload an **After (T2)** satellite image (under "Change Detection")
3. Enter a change query: *"What changed between these two satellite images?"*
4. Click **Analyze**

### Via API
```python
from satquery.pipeline import SatQueryPipeline

pipe = SatQueryPipeline()
result = pipe.run(
    image_path="t1.png",
    query="What changed between these two satellite images?",
    image_t2_path="t2.png",
)
```

## Input Validation

- **Missing T2:** Returns clear error: "Two images required for change detection."
- **Invalid path:** Returns error with specific path.
- **Different dimensions:** Handled safely via resize preprocessing.

## Output

The pipeline returns a `PipelineResult` with:

- `change_result.change_detected` — boolean
- `change_result.change_pct` — percentage of changed pixels
- `change_result.regions` — list of bounding boxes with area
- `change_result.overlay_path` — annotated before/after overlay
- `change_result.bbox_path` — bounding box visualization
- `change_result.inference_time_ms` — timing
- `change_result.vram_peak_mb` — GPU memory usage
- `change_result.format_markdown()` — human-readable summary

## Limitations

1. **Domain:** Trained on LEVIR-CD (Chinese building change detection). May not generalize to all change types.
2. **Semantic:** Detects WHERE changes occurred, not WHAT type of change.
3. **No EarthDial integration:** The change mask is not fed to EarthDial for natural-language explanation (future work).
4. **Resolution:** Operates at 256×256 internal resolution regardless of input size.

## Resource Management

- BIT-CD loads lazily on first change query (~2s)
- Peak VRAM: ~81 MB (negligible on 4GB GPU)
- Model stays loaded in memory for subsequent queries
- Can be freed via `unload_bit_tool()` if needed

## Reference

Chen, H. & Shi, Z. (2021). "Temporal Semantic Contrastive Learning for Remote Sensing Change Detection." ICCV 2021.
GitHub: https://github.com/justchenhao/BIT_CD
