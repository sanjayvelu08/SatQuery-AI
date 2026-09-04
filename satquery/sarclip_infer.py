"""
SAR-CLIP Inference Script — runs in the isolated earthdial_test_venv
(has transformers + torch). Called via subprocess from the SAR-CLIP bridge
(sarclip_tool.py), so it never shares GPU memory with the main process.

Model: BiliSakura/AlignEarth-SAR-ViT-B-16 (MIT) — a CLIP-style model
adapted to SAR via knowledge distillation (SegEarth-OV / AlignEarth).

Usage:
    python -m satquery.sarclip_infer <image_path> [--model <dir>]

Output: JSON with zero-shot softmax scores over two label sets:
  * coarse: water, urban/built-up area, vegetation, agriculture
  * fine:   the OpenEarthMap-SAR class list shipped with the model

These are IMAGE-LEVEL zero-shot scene labels — NOT pixel-level semantic
segmentation.
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

COARSE_LABELS = [
    "water",
    "urban or built-up area",
    "vegetation",
    "agriculture",
]

FINE_LABELS = [
    "background",
    "bareland or barren",
    "grass",
    "pavement",
    "road",
    "tree or forest",
    "water or river",
    "cropland",
    "building roof or house",
]


def default_model_dir() -> str:
    """The downloaded AlignEarth-SAR checkpoint (external artifact)."""
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, "changemodel_test",
                        "sarclip_feas", "model_alignearth_sar")


def run_sarclip_scene(image_path: str, model_dir: str | None = None) -> dict:
    try:
        import torch
        from PIL import Image
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}"}

    if model_dir is None:
        model_dir = os.environ.get("SATQUERY_SARCLIP_MODEL") or default_model_dir()
    if not os.path.isdir(model_dir):
        return {"success": False,
                "error": f"SAR-CLIP model dir not found: {model_dir}"}
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return {"success": False,
                "error": f"SAR-CLIP model dir invalid (no config.json): {model_dir}"}
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    t_total = time.time()
    try:
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor

        t0 = time.time()
        model = CLIPModel.from_pretrained(model_dir)
        # Slow tokenizer: the repo's tokenizer.json needs a newer tokenizers
        # build than transformers 4.37.2 ships; the slow path reads
        # vocab.json/merges.txt and works on all supported versions.
        tok = CLIPTokenizer.from_pretrained(model_dir)
        iproc = CLIPImageProcessor.from_pretrained(model_dir)
        load_s = time.time() - t0

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(dev).eval()

        image = Image.open(image_path).convert("RGB")

        def scores_for(labels):
            pix = iproc(images=image, return_tensors="pt")["pixel_values"]
            txt = tok(labels, return_tensors="pt", padding=True)
            t0 = time.time()
            with torch.no_grad():
                out = model(
                    pixel_values=pix.to(dev),
                    input_ids=txt["input_ids"].to(dev),
                    attention_mask=txt["attention_mask"].to(dev),
                )
            infer_s = time.time() - t0
            probs = out.logits_per_image.softmax(dim=1)[0]
            return {lab: round(float(p), 4) for lab, p in zip(labels, probs)}, infer_s

        coarse, t_c = scores_for(COARSE_LABELS)
        fine, t_f = scores_for(FINE_LABELS)

        vram = 0.0
        if dev != "cpu":
            vram = round(torch.cuda.memory_allocated(0) / 1024**2, 1)

        return {
            "success": True,
            "model": os.path.basename(model_dir),
            "device": dev,
            "scores": {"coarse": coarse, "fine": fine},
            "load_ms": round(load_s * 1000, 0),
            "infer_ms": round((t_c + t_f) * 1000, 0),
            "total_ms": round((time.time() - t_total) * 1000, 0),
            "vram_used_mb": vram,
            "note": "image-level zero-shot scene labels, not pixel segmentation",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "total_ms": round((time.time() - t_total) * 1000, 0),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    result = run_sarclip_scene(args.image_path, model_dir=args.model)
    print(json.dumps(result))


if __name__ == "__main__":
    main()