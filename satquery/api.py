"""
SatQuery AI — FastAPI Backend (Loop 9B).

Wraps the existing SatQueryPipeline for the React frontend.
Serves annotated images and demo scenarios via REST API.

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

# Ensure satquery is importable
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from satquery.pipeline import SatQueryPipeline, PipelineResult
from satquery.demos import get_demo_list, get_demo_by_name
from satquery.visualize import create_annotated_image
from satquery.vlm import SatQueryVLM

# ── App setup ──────────────────────────────────────────────────

app = FastAPI(
    title="SatQuery AI API",
    description="Remote Sensing Vision-Language Assistant backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Annotated images directory ─────────────────────────────────

ANNOTATED_DIR = os.path.join(ROOT, "annotated")
os.makedirs(ANNOTATED_DIR, exist_ok=True)
app.mount("/annotated", StaticFiles(directory=ANNOTATED_DIR), name="annotated")

# Test images directory
TEST_IMAGES_DIR = os.path.join(ROOT, "test_images")
app.mount("/test-images", StaticFiles(directory=TEST_IMAGES_DIR), name="test-images")

# ── Pipeline singleton ─────────────────────────────────────────

pipeline: SatQueryPipeline | None = None


def get_pipeline() -> SatQueryPipeline:
    global pipeline
    if pipeline is None:
        pipeline = SatQueryPipeline(max_history=10)
    return pipeline


# ── Demo scenarios with web URLs ───────────────────────────────

@app.get("/api/demos")
async def list_demos():
    """Return demo scenarios with image URLs the frontend can load."""
    demo_names = get_demo_list()
    result = []
    for name in demo_names:
        demo = get_demo_by_name(name)
        if not demo:
            continue

        # Map image path to web URL
        image_path = demo["image"]
        image_basename = os.path.basename(image_path)
        image_url = f"/test-images/{image_basename}"

        result.append({
            "name": demo["name"],
            "image_url": image_url,
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
    query: str = Form(...),
    demo: Optional[str] = Form(None),
):
    """Run analysis on an image + query, or return pre-computed demo."""

    # ── Demo mode ───────────────────────────────────────────────
    if demo:
        demo_data = get_demo_by_name(demo)
        if not demo_data:
            raise HTTPException(status_code=404, detail=f"Demo not found: {demo}")

        # Try to create annotated image
        annotated_path = None
        try:
            annotated_path = create_annotated_image(
                demo_data["image"], demo_data["answer"], demo_data["intent"]
            )
        except Exception:
            pass

        # Build response
        image_basename = os.path.basename(demo_data["image"])
        annotated_url = None
        if annotated_path:
            annotated_url = f"/annotated/{os.path.basename(annotated_path)}"

        return {
            "query": demo_data["query"],
            "intent": demo_data["intent"],
            "all_intents": demo_data.get("all_intents", [demo_data["intent"]]),
            "supported": demo_data.get("supported", True),
            "answer": demo_data["answer"],
            "unsupported_reason": "",
            "model_used": demo_data.get("model_used", ""),
            "annotated_image_url": annotated_url,
            "elapsed_route_ms": 0,
            "elapsed_vlm_s": 0,
            "elapsed_total_s": 0,
            "sar_result": None,
        }

    # ── Live mode ───────────────────────────────────────────────
    if image is None:
        raise HTTPException(status_code=400, detail="No image provided")

    # Save uploaded image to temp file
    ext = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
    temp_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    temp_path = os.path.join(ANNOTATED_DIR, temp_name)

    content = await image.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        # Run pipeline
        result = get_pipeline().run(temp_path, query)

        # Create annotated image if applicable
        annotated_url = None
        if result.annotated_image:
            annotated_url = f"/annotated/{os.path.basename(result.annotated_image)}"

        # Build SAR result
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
            "elapsed_route_ms": result.elapsed_route_ms,
            "elapsed_vlm_s": result.elapsed_vlm_s,
            "elapsed_total_s": result.elapsed_total_s,
            "sar_result": sar_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ───────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Check system health and VRAM status."""
    vram = {}
    try:
        vram = SatQueryVLM.vram_info()
    except Exception:
        vram = {"error": "CUDA not available"}

    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None and pipeline.vlm.is_loaded,
        "vram": vram,
        "models": {
            "earthdial": os.path.exists(os.path.join(ROOT, "checkpoints", "EarthDial_4B_RGB", "config.json")),
            "sar_yolo": os.path.exists(os.path.join(ROOT, "checkpoints", "sar_vessel", "unquantized", "best.pt")),
        },
    }


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
