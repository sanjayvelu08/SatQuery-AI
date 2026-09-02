"""
EarthDial Inference Script — runs in isolated earthdial_test_venv.
Called via subprocess from the main SatQuery pipeline.

Usage:
    python -m satquery.earthdial_infer <image_path> <prompt> [--max_tokens 200] [--num_beams 5]

Output: JSON with answer, timing, success status.
"""

import json
import os
import sys
import time
import warnings
import argparse

warnings.filterwarnings("ignore")


def run_earthdial_inference(
    image_path: str,
    prompt: str,
    max_tokens: int = 200,
    num_beams: int = 5,
) -> dict:
    """Run EarthDial inference on a single image. Returns structured result."""
    try:
        import torch
        from PIL import Image
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}"}

    # Find EarthDial source
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    earthdial_src = os.path.join(project_root, "EarthDial", "src")
    if not os.path.isdir(earthdial_src):
        return {"success": False, "error": f"EarthDial source not found: {earthdial_src}"}

    if earthdial_src not in sys.path:
        sys.path.insert(0, earthdial_src)

    # Find checkpoint
    model_dir = os.path.join(project_root, "checkpoints", "EarthDial_4B_RGB")
    if not os.path.isdir(model_dir):
        return {"success": False, "error": f"EarthDial checkpoint not found: {model_dir}"}

    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    t_total = time.time()

    try:
        from earthdial.model.internvl_chat import InternVLChatModel
        from earthdial.train.dataset import build_transform
        from transformers import AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
        model = InternVLChatModel.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto",
        ).eval()
        image_size = model.config.force_image_size or model.config.vision_config.image_size
        transform = build_transform(is_train=False, input_size=image_size, normalize_type="imagenet")
        load_ms = (time.time() - t0) * 1000

        vram_used = torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0

        # Preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).cuda().to(torch.bfloat16)

        # Inference
        gen_cfg = {
            "num_beams": num_beams,
            "max_new_tokens": max_tokens,
            "min_new_tokens": 1,
            "do_sample": False,
        }

        t0 = time.time()
        answer = model.chat(tok, pixel_values, prompt, gen_cfg, verbose=False)
        inference_ms = (time.time() - t0) * 1000

        # Cleanup
        del model, tok, transform
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_ms = (time.time() - t_total) * 1000

        return {
            "success": True,
            "answer": answer,
            "load_ms": round(load_ms, 0),
            "inference_ms": round(inference_ms, 0),
            "total_ms": round(total_ms, 0),
            "vram_used_mb": round(vram_used, 0),
            "model_dir": model_dir,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "total_ms": round((time.time() - t_total) * 1000, 0),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to satellite image")
    parser.add_argument("prompt", help="Prompt to send to EarthDial")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()

    result = run_earthdial_inference(
        image_path=args.image_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        num_beams=args.num_beams,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
