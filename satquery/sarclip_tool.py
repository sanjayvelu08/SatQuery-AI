"""
SatQuery AI — SAR-CLIP Subprocess Bridge + Lightweight Intensity Indicators.

Two pieces:

1. SAR-CLIP scene labelling: runs satquery.sarclip_infer in the isolated
   earthdial_test_venv (same venv as EarthDial) via subprocess, so the
   ~0.6 GB CLIP model never coexists with EarthDial in GPU memory.

2. Otsu intensity indicators: dark/water-like and bright/built-up-like
   pixel fractions computed with numpy on the raw SAR image. These are
   image-relative STATISTICAL INDICATORS — explicitly NOT semantic
   segmentation (no trained segmentation model is used).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

from .vlm_bridge import _find_earthdial_venv_python


# ── 1. SAR-CLIP scene labels (subprocess) ────────────────────────────────

def run_sarclip_scene(
    image_path: str,
    timeout: int = 120,
    model_dir: str | None = None,
) -> dict:
    """Run zero-shot SAR scene labelling via isolated subprocess.

    Returns a dict (never raises): {success, scores, error, ...}.
    """
    venv_python = _find_earthdial_venv_python()
    if venv_python is None:
        return {"success": False,
                "error": "earthdial_test_venv not found (SAR-CLIP unavailable)"}
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    cmd = [venv_python, "-X", "utf8", "-m", "satquery.sarclip_infer",
           os.path.abspath(image_path)]
    if model_dir:
        cmd += ["--model", os.path.abspath(model_dir)]

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, cwd=project_root)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"SAR-CLIP timed out (>{timeout}s)"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}

    if result.returncode != 0:
        err = result.stderr.strip().split("\n")
        msg = next((l for l in err if "Error" in l or "error" in l), result.stderr[:300])
        return {"success": False, "error": f"SAR-CLIP subprocess failed: {msg}"}

    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse SAR-CLIP output: {e}"}
    return data


def format_scene_scores(scores: dict, top: int = 4) -> str:
    """Render zero-shot scores as 'label 72.6%, ...' (descending)."""
    if not scores:
        return "unavailable"
    items = sorted(scores.items(), key=lambda x: -x[1])[:top]
    return ", ".join(f"{k} {v*100:.1f}%" for k, v in items)


# ── 2. Otsu intensity indicators (in-process, numpy only) ────────────────

def otsu_intensity_indicators(image_path: str) -> dict:
    """Compute image-relative intensity indicators on the SAR image.

    Returns:
        threshold      Otsu intensity threshold (0-255)
        dark_fraction  fraction of pixels BELOW threshold (water-like:
                       smooth/dark radar returns) — indicator only
        bright_fraction fraction of pixels above 200 (built-up-like:
                       strong/bright radar returns) — indicator only
        mid_fraction   remaining pixels
        dark_components_ge50  count of dark regions >= 50 px (scipy) or None
        usable_mask    whether a mask could be derived
    """
    import numpy as np
    from PIL import Image

    im = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
    tot = im.size
    hist, _ = np.histogram(im, bins=256, range=(0, 256))
    pdf = hist / hist.sum()
    cum = np.cumsum(pdf)
    mu = np.cumsum(pdf * np.arange(256))
    mu_t = mu[-1]
    best_t, best_var = 0, -1.0
    for t in range(1, 255):
        w0 = cum[t]
        if 0.0 < w0 < 1.0:
            var_b = (mu_t * w0 - mu[t]) ** 2 / (w0 * (1.0 - w0))
            if var_b > best_var:
                best_var, best_t = var_b, t

    dark_frac = float((im < best_t).mean())
    bright_frac = float((im > 200).mean())
    mask = im < best_t

    components = None
    try:
        from scipy import ndimage
        lab, _ = ndimage.label(mask)
        sizes = np.bincount(lab.ravel())[1:]
        components = int((sizes >= 50).sum())
    except ImportError:
        pass  # scipy optional

    return {
        "threshold": int(best_t),
        "dark_fraction": round(dark_frac, 4),
        "bright_fraction": round(bright_frac, 4),
        "mid_fraction": round(max(0.0, 1.0 - dark_frac - bright_frac), 4),
        "dark_pixels": int(dark_frac * tot),
        "bright_pixels": int(bright_frac * tot),
        "dark_components_ge50": components,
        "usable_mask": True,
        "note": ("image-relative Otsu intensity indicators (dark = water-like, "
                 "bright = built-up-like); NOT semantic segmentation"),
    }


def format_intensity_indicators(ind: dict) -> str:
    """Render the intensity indicators as an honest text summary."""
    if not ind:
        return "unavailable"
    comp = ind.get("dark_components_ge50")
    comp_txt = f"; {comp} dark region(s) >= 50px" if comp is not None else ""
    return (
        f"dark/water-like returns {ind['dark_fraction']*100:.1f}% of pixels, "
        f"bright/built-up-like returns {ind['bright_fraction']*100:.1f}% "
        f"(Otsu threshold {ind['threshold']}){comp_txt}"
    )