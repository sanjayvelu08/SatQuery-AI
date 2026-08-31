"""
SAR Inference Script — runs in isolated sar_venv.
Called via subprocess from the main SatQuery pipeline.

Usage:
    python -m satquery.sar_infer <image_path> [--conf 0.25] [--device auto]
"""

import json
import sys
import os
import time
import argparse


def run_sar_detection(image_path: str, conf: float = 0.25, device: str = "auto") -> dict:
    """Run YOLOv8 SAR vessel detection on an image. Returns structured result."""
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        return {"error": f"Missing dependency: {e}", "success": False}

    # Resolve model path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "checkpoints", "sar_vessel", "unquantized", "best.pt")

    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}", "success": False}

    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}", "success": False}

    try:
        model = YOLO(model_path)

        # Device selection
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        start = time.time()
        results = model(image_path, conf=conf, device=device)
        elapsed = time.time() - start

        detections = []
        class_names = {}
        for r in results:
            class_names = r.names
            for box in r.boxes:
                cls = int(box.cls[0])
                detections.append({
                    "class": class_names.get(cls, f"unknown_{cls}"),
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox_xyxy": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                })

        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        vram_used = 0
        if torch.cuda.is_available():
            vram_used = round(torch.cuda.memory_allocated() / 1024**2, 1)

        return {
            "success": True,
            "model": os.path.basename(model_path),
            "device": str(device),
            "num_detections": len(detections),
            "detections": detections,
            "supported_classes": class_names,
            "inference_time_ms": round(elapsed * 1000, 1),
            "gpu_vram_mb": vram_used,
        }
    except Exception as e:
        return {"error": f"Inference failed: {e}", "success": False}


def format_natural_language(result: dict) -> str:
    """Convert structured SAR detection result to natural language summary."""
    if not result.get("success"):
        return f"SAR analysis failed: {result.get('error', 'Unknown error')}"

    dets = result["detections"]
    n = result["num_detections"]

    if n == 0:
        return (
            "**SAR Analysis Result:**\n\n"
            "No vessels or maritime targets detected in this SAR image with sufficient confidence.\n\n"
            "_Note: This detector identifies ships and maritime targets only. "
            "It does not analyze terrain, vegetation, or other SAR features._"
        )

    lines = [f"**SAR Analysis Result:**\n"]
    lines.append(f"Detected **{n}** maritime target(s):\n")
    lines.append("| # | Object | Confidence | Bounding Box (x1,y1,x2,y2) |")
    lines.append("|---|--------|-----------|---------------------------|")

    for i, d in enumerate(dets, 1):
        bb = d["bbox_xyxy"]
        lines.append(
            f"| {i} | {d['class'].title()} | {d['confidence']:.1%} | "
            f"[{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}] |"
        )

    lines.append("")
    lines.append(
        "_Note: This detector identifies ships and maritime targets in SAR imagery. "
        "It uses a YOLOv8 model trained on SAR vessel detection data. "
        "It does not provide natural-language scene understanding of SAR imagery._"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SAR Vessel Detection Inference")
    parser.add_argument("image_path", help="Path to SAR image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda:0")
    parser.add_argument("--text", action="store_true", help="Output natural language instead of JSON")
    args = parser.parse_args()

    result = run_sar_detection(args.image_path, conf=args.conf, device=args.device)

    if args.text:
        print(format_natural_language(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
