"""
SatQuery AI — FastAPI Backend.

Wraps SatQueryPipeline for the React frontend.
Supports single-image, bi-temporal change detection, and optical+SAR joint analysis.

Run:  python -m satquery.api
URL:  http://localhost:8000
"""

from __future__ import annotations

import os
import sys
import uuid
import time
from pathlib import Path
from typing import Optional

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from satquery.pipeline import SatQueryPipeline, build_result_summary
from satquery.demos import get_demo_list, get_demo_by_name
from satquery.visualize import create_annotated_image
from satquery.vlm import SatQueryVLM

# ── App setup ──────────────────────────────────────────────────

app = FastAPI(
    title="SatQuery AI API",
    description="Multimodal Remote Sensing Assistant — optical, SAR, change detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static directories ─────────────────────────────────────────

ANNOTATED_DIR = os.path.join(ROOT, "annotated")
os.makedirs(ANNOTATED_DIR, exist_ok=True)
app.mount("/annotated", StaticFiles(directory=ANNOTATED_DIR), name="annotated")

TEST_IMAGES_DIR = os.path.join(ROOT, "test_images")
if os.path.isdir(TEST_IMAGES_DIR):
    app.mount("/test-images", StaticFiles(directory=TEST_IMAGES_DIR), name="test-images")

# Change overlay outputs
CHANGES_DIR = os.path.join(ROOT, "change_output")
os.makedirs(CHANGES_DIR, exist_ok=True)
app.mount("/changes", StaticFiles(directory=CHANGES_DIR), name="changes")


# ── Pipeline singleton ─────────────────────────────────────────

_pipeline: SatQueryPipeline | None = None


def get_pipeline() -> SatQueryPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SatQueryPipeline(max_history=10)
    return _pipeline


# ── Helpers ────────────────────────────────────────────────────

def _save_upload(upload: UploadFile, prefix: str = "upload") -> str:
    """Save an uploaded file and return the path."""
    ext = os.path.splitext(upload.filename or "img.jpg")[1] or ".jpg"
    name = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(ANNOTATED_DIR, name)
    content = upload.file.read()
    with open(path, "wb") as f:
        f.write(content)
    return path


def _serve_path(path: str | None, mount: str = "/annotated") -> str | None:
    """Convert a local file path to a serveable URL."""
    if not path:
        return None
    basename = os.path.basename(path)
    return f"{mount}/{basename}"


def _serialize_change_result(cr) -> dict | None:
    """Serialize a ChangeDetectionResult to JSON-safe dict."""
    if cr is None:
        return None
    if hasattr(cr, "to_dict"):
        d = cr.to_dict()
    else:
        d = {"raw": str(cr)}
    # to_dict() omits file paths — read from dataclass and map to serveable URLs
    for field_name, url_prefix in [
        ("overlay_path", "/changes"),
        ("bbox_path", "/changes"),
        ("mask_path", "/changes"),
    ]:
        url_key = field_name.replace("_path", "_url")
        raw_path = getattr(cr, field_name, None)
        if raw_path and os.path.isfile(raw_path):
            d[url_key] = f"{url_prefix}/{os.path.basename(raw_path)}"
        else:
            d[url_key] = None
    return d


def _serialize_joint_result(jr) -> dict | None:
    """Serialize a JointAnalysisResult to JSON-safe dict."""
    if jr is None:
        return None
    if hasattr(jr, "to_dict"):
        return jr.to_dict()
    return {"raw": str(jr)}


# ── Demo scenarios ─────────────────────────────────────────────

@app.get("/api/demos")
async def list_demos():
    """Return demo scenarios with image URLs."""
    demo_names = get_demo_list()
    result = []
    for name in demo_names:
        demo = get_demo_by_name(name)
        if not demo:
            continue
        image_basename = os.path.basename(demo["image"])
        result.append({
            "name": demo["name"],
            "image_url": f"/test-images/{image_basename}",
            "query": demo["query"],
            "intent": demo["intent"],
            "model_used": demo.get("model_used", ""),
            "answer": demo["answer"],
            "all_intents": demo.get("all_intents", [demo["intent"]]),
            "supported": demo.get("supported", True),
        })
    return result


# ── Analysis endpoint ──────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    image: Optional[UploadFile] = File(None),
    image_t2: Optional[UploadFile] = File(None),
    image_sar: Optional[UploadFile] = File(None),
    query: str = Form(...),
    demo: Optional[str] = Form(None),
):
    """
    Run analysis on image(s) + query.

    Modes:
      - Single image:   image only
      - Change detect:  image (T1) + image_t2 (T2)
      - Joint analysis: image (optical) + image_sar (SAR)
    """

    # ── Demo mode ───────────────────────────────────────────────
    if demo:
        demo_data = get_demo_by_name(demo)
        if not demo_data:
            raise HTTPException(status_code=404, detail=f"Demo not found: {demo}")

        annotated_path = None
        try:
            annotated_path = create_annotated_image(
                demo_data["image"], demo_data["answer"], demo_data["intent"]
            )
        except Exception:
            pass

        return {
            "query": demo_data["query"],
            "intent": demo_data["intent"],
            "all_intents": demo_data.get("all_intents", [demo_data["intent"]]),
            "supported": demo_data.get("supported", True),
            "answer": demo_data["answer"],
            "unsupported_reason": "",
            "model_used": demo_data.get("model_used", ""),
            "annotated_image_url": _serve_path(annotated_path),
            "change_result": None,
            "joint_result": None,
            "elapsed_route_ms": 0,
            "elapsed_vlm_s": 0,
            "elapsed_total_s": 0,
            "sar_result": None,
            "trace": [],
            "summary": {
                "query": demo_data["query"],
                "intent": demo_data["intent"],
                "models_used": demo_data.get("model_used", ""),
                "evidence_reliability": None,
                "reliability_reasoning": None,
                "reliability_note": (
                    "pre-recorded demo result"
                    if not demo_data.get("supported", True)
                    else "qualitative model result \u2014 reliability not quantified"
                ),
                "warnings": [] if demo_data.get("supported", True) else [
                    demo_data.get("unsupported_reason", "unsupported")],
                "trace_step_count": 0,
            },
        }

    # ── Live mode ───────────────────────────────────────────────
    if image is None:
        raise HTTPException(status_code=400, detail="No image provided")

    # Save uploaded files
    image_path = _save_upload(image, "upload")

    image_t2_path = None
    if image_t2 is not None:
        image_t2_path = _save_upload(image_t2, "t2")

    image_sar_path = None
    if image_sar is not None:
        image_sar_path = _save_upload(image_sar, "sar")

    try:
        # Run the full pipeline
        result = get_pipeline().run(
            image_path,
            query,
            image_t2_path=image_t2_path,
            image_sar_path=image_sar_path,
        )

        # Build response
        annotated_url = _serve_path(result.annotated_image)

        sar_result = None
        if result.sar_result:
            sr = result.sar_result
            sar_result = {
                "success": sr.success,
                "detections": [
                    {
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox_xyxy": d.bbox_xyxy,
                    }
                    for d in sr.detections
                ],
                "num_detections": sr.num_detections,
                "inference_time_ms": sr.inference_time_ms,
                "gpu_vram_mb": sr.gpu_vram_mb,
                "error": sr.error,
            }

        return {
            "query": result.query,
            "intent": result.intent,
            "all_intents": result.all_intents,
            "supported": result.supported,
            "answer": result.answer,
            "unsupported_reason": result.unsupported_reason,
            "model_used": result.model_used,
            "annotated_image_url": annotated_url,
            "change_result": _serialize_change_result(result.change_result),
            "joint_result": _serialize_joint_result(result.joint_result),
            "elapsed_route_ms": result.elapsed_route_ms,
            "elapsed_vlm_s": result.elapsed_vlm_s,
            "elapsed_total_s": result.elapsed_total_s,
            "sar_result": sar_result,
            "trace": [
                {
                    "step": t.step, "name": t.name, "tool": t.tool,
                    "status": t.status, "duration_ms": t.duration_ms,
                    "input_summary": t.input_summary,
                    "output_summary": t.output_summary,
                    "error": t.error,
                }
                for t in result.trace
            ],
            "summary": build_result_summary(result),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ───────────────────────────────────────────────

@app.get("/api/health")
async def health():
    vram = {}
    try:
        vram = SatQueryVLM.vram_info()
    except Exception:
        vram = {"error": "CUDA not available"}

    return {
        "status": "ok",
        "pipeline_ready": _pipeline is not None and _pipeline.vlm.is_loaded,
        "vram": vram,
        "models": {
            "earthdial": os.path.exists(
                os.path.join(ROOT, "checkpoints", "EarthDial_4B_RGB", "config.json")
            ),
            "sar_yolo": os.path.exists(
                os.path.join(ROOT, "checkpoints", "sar_vessel", "unquantized", "best.pt")
            ),
        },
    }


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
