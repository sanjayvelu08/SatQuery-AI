"""
SatQuery AI — GeoTIFF / TIFF utilities (CPU-only).

Implements the SIH 26167 geospatial-input milestone with the already-installed
tifffile + numpy + Pillow (no rasterio / GDAL):

  probe_image(path)           -> dict of image + geospatial metadata.
                                 Missing metadata is None / "unknown" — never
                                 guessed.
  render_rgb(path, ...)       -> band-preserving RGB (uint8) PNG for a
                                 (possibly multispectral / float) TIFF. Raw
                                 TIFFs are never sent down the old Pillow
                                 "convert('RGB')" path, which silently dropped
                                 bands (e.g. a 13-band S2 scene read as
                                 all-white single band).
  check_pair_compat(a, b)     -> verdict on dims / CRS / resolution / bounds
                                 for an optical+SAR or T1+T2 pair. Equal
                                 dimensions alone NEVER imply co-registration.

Rendering notes
  - 3-band RGB (Photometric=RGB) -> channels preserved verbatim.
  - multispectral 12/13 band     -> default RGB = bands (3, 2, 1) i.e. S2
                                    B4/B3/B2 true color (documented heuristic,
                                    configurable via band_indices).
  - other >=3 band               -> default first three bands (band_indices
                                    configurable).
  - 2-band (e.g. S1 VV/VH)       -> rendered as R=VV, G=VH, B=VV so both
                                    channels inform the view (false colour —
                                    NOT true colour); metadata retains both.
  - 1-band                       -> grayscale replicated to RGB.
  - NaN / nodata pixels          -> masked out of the percentile stretch and
                                    rendered black; never propagated.

No reprojection / resampling / GDAL functionality is implemented here by design.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
from PIL import Image

# Sentinel-2 true colour band order inside the native 13-band stack:
# 0=B1, 1=B2(blue), 2=B3(green), 3=B4(red), ...
S2_TRUE_COLOR_BANDS = (3, 2, 1)
# 2-band SAR (e.g. Sentinel-1 GRD VV/VH) false-colour visualisation.
SAR_VV_VH_BANDS = (0, 1, 0)

# TIFF tag codes used for geospatial metadata
_TAG_MODEL_PIXEL_SCALE = 33550      # (sx, sy, sz)
_TAG_MODEL_TIEPOINT = 33922         # (i, j, k, x, y, z) first point
_TAG_GEO_KEY_DIRECTORY = 34735
_TAG_GEO_ASCII_PARAMS = 34737
_TAG_GDAL_NODATA = 42113
_TAG_DATE_TIME = 306                # "YYYY:MM:DD hh:mm:ss"
_TAG_IMAGE_DESCRIPTION = 270

# GeoKey IDs inside GeoKeyDirectory
_KID_GT_MODEL_TYPE = 1024           # 1=projected 2=geographic
_KID_PROJECTED_CS = 3072
_KID_GEOGRAPHIC_CS = 2048

_BITS_TO_DTYPE = {
    (8, 1): "uint8", (8, 2): "int8",
    (16, 1): "uint16", (16, 2): "int16",
    (32, 1): "uint32", (32, 2): "int32", (32, 3): "float32",
    (64, 3): "float64",
}


# ─────────────────────────────────────────────────────────────────────────
# Probe
# ─────────────────────────────────────────────────────────────────────────
def probe_image(path: str) -> dict:
    """Return image + (where present) geospatial metadata for any input.

    Works for TIFF/GeoTIFF (via tifffile tags — no full raster read) and for
    ordinary JPEG/PNG/BMP/WebP (Pillow dims only). Raises ValueError if the
    file is unreadable / not an image.
    """
    if not os.path.exists(path):
        raise ValueError(f"image not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return _probe_tiff(path)
    return _probe_raster(path)


def _probe_tiff(path: str) -> dict:
    import tifffile
    try:
        tf = tifffile.TiffFile(path)
    except Exception as exc:  # corrupt / not a real TIFF
        raise ValueError(f"unreadable TIFF '{os.path.basename(path)}': {exc}") from exc

    try:
        page = tf.pages[0]
        tags = {}
        for code in (_TAG_MODEL_PIXEL_SCALE, _TAG_MODEL_TIEPOINT,
                     _TAG_GEO_KEY_DIRECTORY, _TAG_GEO_ASCII_PARAMS,
                     _TAG_GDAL_NODATA, _TAG_DATE_TIME, _TAG_IMAGE_DESCRIPTION):
            try:
                tags[code] = page.tags[code].value
            except KeyError:
                pass

        width = int(page.tags[256].value)
        height = int(page.tags[257].value)
        spp = int(page.tags[277].value) if 277 in page.tags else 1
        photometric = int(page.tags[262].value) if 262 in page.tags else None
        bits = 8
        if 258 in page.tags:
            bv = page.tags[258].value
            bits = int(bv[0]) if isinstance(bv, (list, tuple, np.ndarray)) else int(bv)
        sample_format = 1
        if 339 in page.tags:
            sf = page.tags[339].value
            sample_format = int(sf[0]) if isinstance(sf, (list, tuple, np.ndarray)) else int(sf)

        geo_keys = _parse_geo_keys(tags.get(_TAG_GEO_KEY_DIRECTORY))
        epsg, crs_type = geo_keys
        pixel_scale = _as_float_list(tags.get(_TAG_MODEL_PIXEL_SCALE))
        tiepoint = _as_float_list(tags.get(_TAG_MODEL_TIEPOINT))
        bounds = _bounds_from_geotransform(width, height, pixel_scale, tiepoint)

        nodata = None
        if _TAG_GDAL_NODATA in tags:
            raw_nd = tags[_TAG_GDAL_NODATA]
            if isinstance(raw_nd, (bytes, bytearray)):
                raw_nd = bytes(raw_nd).decode("ascii", "replace")
            try:
                nodata = float(str(raw_nd).strip())
            except (TypeError, ValueError):
                nodata = None
        date_time = _normalize_datetime(tags.get(_TAG_DATE_TIME))

        has_geo = any(k in tags for k in (
            _TAG_MODEL_PIXEL_SCALE, _TAG_MODEL_TIEPOINT, _TAG_GEO_KEY_DIRECTORY))
        fmt = "geotiff" if has_geo else "tiff"
        layout = ("rgb-last" if (photometric == 2 and spp >= 3)
                  else "first" if spp > 1 else "single")

        return {
            "path": path, "format": fmt, "is_tiff": True,
            "width": width, "height": height, "bands": spp,
            "dtype": _BITS_TO_DTYPE.get((bits, sample_format), f"unknown({bits}bit)"),
            "photometric": photometric, "page_count": len(tf.pages),
            "layout_hint": layout,
            "epsg": epsg, "crs_type": crs_type,
            "crs_wkt": _ascii_param(tags.get(_TAG_GEO_ASCII_PARAMS)),
            "pixel_scale": pixel_scale,          # native units (deg for EPSG:4326)
            "tiepoint": tiepoint,
            "bounds": bounds,                    # {left, top, right, bottom}
            "nodata": nodata, "date_time": date_time,
            "description": str(tags[_TAG_IMAGE_DESCRIPTION])[:200]
            if _TAG_IMAGE_DESCRIPTION in tags else None,
        }
    finally:
        tf.close()


def _probe_raster(path: str) -> dict:
    try:
        im = Image.open(path)
        im.load()
        fmt = (im.format or os.path.splitext(path)[1].lstrip(".")).lower()
        mode = im.mode or ""
        bands = {"RGB": 3, "RGBA": 4, "L": 1, "LA": 2, "P": 3}.get(mode, 3)
        return {
            "path": path, "format": fmt, "is_tiff": False,
            "width": int(im.width), "height": int(im.height), "bands": bands,
            "dtype": "uint8", "photometric": None, "page_count": 1,
            "layout_hint": "single" if bands == 1 else "unknown",
            "epsg": None, "crs_type": None, "crs_wkt": None,
            "pixel_scale": None, "tiepoint": None, "bounds": None,
            "nodata": None, "date_time": None, "description": None,
        }
    except Exception as exc:
        raise ValueError(f"unreadable image '{os.path.basename(path)}': {exc}") from exc


def _parse_geo_keys(raw) -> tuple[Optional[int], Optional[str]]:
    """Extract (epsg, crs_type) from a GeoKeyDirectory tag value."""
    if raw is None:
        return None, None
    vals = list(raw)
    if len(vals) < 4:
        return None, None
    n = int(vals[3])
    epsg = None
    crs_type = None
    for i in range(n):
        start = 4 + 4 * i
        if start + 4 > len(vals):
            break
        kid, _loc, _cnt, val = (int(vals[start]), int(vals[start + 1]),
                                int(vals[start + 2]), int(vals[start + 3]))
        if kid == _KID_GT_MODEL_TYPE:
            crs_type = "projected" if val == 1 else "geographic" if val == 2 else None
        elif kid in (_KID_PROJECTED_CS, _KID_GEOGRAPHIC_CS) and val not in (0, 32767):
            epsg = val
    return epsg, crs_type


def _ascii_param(raw) -> Optional[str]:
    if raw is None:
        return None
    text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8", "replace")
    return text.split("|")[0].strip() or None


def _as_float_list(raw) -> Optional[list[float]]:
    if raw is None:
        return None
    return [float(x) for x in raw]


def _bounds_from_geotransform(width, height, pixel_scale, tiepoint) -> Optional[dict]:
    """Bounds from GDAL ModelPixelScale + first ModelTiepoint, if both present."""
    if not pixel_scale or not tiepoint or len(pixel_scale) < 2 or len(tiepoint) < 6:
        return None
    sx, sy = float(pixel_scale[0]), float(pixel_scale[1])
    ox, oy = float(tiepoint[3]), float(tiepoint[4])
    return {
        "left": ox, "top": oy,
        "right": ox + width * sx,
        "bottom": oy - height * sy,
    }


def _normalize_datetime(raw) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    # TIFF DateTime: "YYYY:MM:DD hh:mm:ss"
    if len(text) >= 19 and text[4] == ":":
        return f"{text[0:4]}-{text[5:7]}-{text[8:10]} {text[11:19]}"
    return text[:25] or None


# ─────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────
def suggest_rgb_bands(num_bands: int, photometric: Optional[int] = None):
    """Default RGB band indices for render_rgb (documented heuristics)."""
    if num_bands == 1:            # single band -> grayscale replicated to RGB
        return (0, 0, 0)
    if photometric == 2 and num_bands >= 3:
        return (0, 1, 2)          # native RGB(A) -> preserve channels
    if num_bands in (12, 13):     # S2-like stacks: true colour B4/B3/B2
        return S2_TRUE_COLOR_BANDS
    if num_bands == 2:            # SAR VV/VH false colour
        return SAR_VV_VH_BANDS
    return (0, 1, 2)


def render_rgb(path: str, out_path: Optional[str] = None,
               band_indices: Optional[Sequence[int]] = None,
               stretch_percentiles=(2.0, 98.0)) -> str:
    """Render a TIFF to an 8-bit RGB PNG for downstream specialists.

    Returns the path written (defaults to '<stem>_geo_rgb.png' beside the
    source). Raises ValueError for unreadable files, unexpected band layouts,
    or out-of-range band indices. Never returns the previous all-white /
    band-dropped Pillow output.
    """
    import tifffile
    if out_path is None:
        out_path = f"{os.path.splitext(path)[0]}_geo_rgb.png"
    out_path = os.path.abspath(out_path)

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        spp = int(page.tags[277].value) if 277 in page.tags else 1
        photometric = int(page.tags[262].value) if 262 in page.tags else None
        bits = 8
        if 258 in page.tags:
            bv = page.tags[258].value
            bits = int(bv[0]) if isinstance(bv, (list, tuple, np.ndarray)) else int(bv)

    bands = _split_bands(arr, spp, photometric)
    if band_indices is None:
        idx = suggest_rgb_bands(len(bands), photometric)
    else:
        idx = list(band_indices)
    if len(idx) != 3:
        raise ValueError(f"render_rgb needs exactly 3 band indices, got {idx}")
    if any(i < 0 or i >= len(bands) for i in idx):
        raise ValueError(f"band indices {idx} out of range for {len(bands)} band(s)")

    nodata = probe_image(path).get("nodata")
    channels = [
        _band_to_uint8(bands[i], nodata, stretch_percentiles)
        for i in idx
    ]
    rgb = np.stack(channels, axis=-1)

    img = Image.fromarray(rgb, "RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def _split_bands(arr: np.ndarray, spp: int,
                 photometric: Optional[int]) -> list[np.ndarray]:
    """Normalize a tifffile array into a list of 2-D per-band arrays."""
    if arr.ndim == 2:
        return [arr]
    if arr.ndim != 3:
        raise ValueError(f"unsupported TIFF layout: shape {arr.shape}")
    if photometric == 2 and arr.shape[-1] == spp:
        return [np.ascontiguousarray(arr[..., i]) for i in range(arr.shape[-1])]
    if arr.shape[0] == spp:
        return [np.ascontiguousarray(arr[i]) for i in range(spp)]
    if arr.shape[-1] == spp:
        return [np.ascontiguousarray(arr[..., i]) for i in range(spp)]
    raise ValueError(f"cannot determine band layout for shape {arr.shape} "
                     f"spp={spp} photometric={photometric}")


def _band_to_uint8(band: np.ndarray, nodata: Optional[float],
                   stretch_percentiles) -> np.ndarray:
    """Convert one band to 0..255 uint8 with NaN/nodata handled safely."""
    if band.dtype == np.uint8:
        out = np.asarray(band, dtype=np.uint8)
    else:
        f = band.astype(np.float64)
        invalid = ~np.isfinite(f)
        if nodata is not None and np.isfinite(nodata):
            invalid |= np.isclose(f, nodata)
        valid = f[~invalid]
        if valid.size >= 10:
            lo = float(np.percentile(valid, stretch_percentiles[0]))
            hi = float(np.percentile(valid, stretch_percentiles[1]))
        else:
            lo, hi = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)
        span = hi - lo
        scaled = np.zeros_like(f)
        if np.isfinite(span) and span > 0:
            scaled[~invalid] = (
                np.clip((f[~invalid] - lo) / span, 0.0, 1.0) * 255.0)
        # invalid (NaN/nodata) stays 0 -> black, never propagated
        out = scaled.astype(np.uint8)
    return np.ascontiguousarray(out)


# ─────────────────────────────────────────────────────────────────────────
# Pair compatibility
# ─────────────────────────────────────────────────────────────────────────
def check_pair_compat(meta_a: dict, meta_b: dict) -> dict:
    """Compare two probe_image() dicts for a pair workflow.

    Distinguishes compatible / incompatible and co_registration
    verified / unverified. Never infers co-registration from equal dimensions
    alone; without sufficient geospatial metadata it returns the warning
    "co-registration unverified — insufficient geospatial metadata".
    """
    verdict = {
        "status": "compatible",
        "co_registration": "unverified",
        "reasons": [],
        "warnings": [],
        "summary": "",
    }
    a_missing = not meta_a or meta_a.get("width") is None or meta_a.get("height") is None
    b_missing = not meta_b or meta_b.get("width") is None or meta_b.get("height") is None
    if a_missing or b_missing:
        verdict["warnings"].append(
            "co-registration unverified — insufficient geospatial metadata")
        verdict["co_registration"] = "unverified"
        _finalize_verdict(verdict)
        return verdict

    dims_a = (int(meta_a["width"]), int(meta_a["height"]))
    dims_b = (int(meta_b["width"]), int(meta_b["height"]))
    epsg_a, epsg_b = meta_a.get("epsg"), meta_b.get("epsg")
    scale_a, scale_b = meta_a.get("pixel_scale"), meta_b.get("pixel_scale")
    bounds_a, bounds_b = meta_a.get("bounds"), meta_b.get("bounds")

    # Explicit contradiction 1: known CRS differ
    if epsg_a is not None and epsg_b is not None and epsg_a != epsg_b:
        verdict["status"] = "incompatible"
        verdict["reasons"].append(
            f"CRS mismatch: EPSG:{epsg_a} vs EPSG:{epsg_b} (no reprojection support)")
        _finalize_verdict(verdict)
        return verdict

    # Explicit contradiction 2: bounding boxes exist and do not overlap
    if bounds_a and bounds_b:
        ix = (min(bounds_a["right"], bounds_b["right"])
              - max(bounds_a["left"], bounds_b["left"]))
        iy = (min(bounds_a["top"], bounds_b["top"])
              - max(bounds_a["bottom"], bounds_b["bottom"]))
        if ix <= 0 or iy <= 0:
            verdict["status"] = "incompatible"
            verdict["reasons"].append("spatial extents do not overlap")
            _finalize_verdict(verdict)
            return verdict

    geo_complete = bool(scale_a and scale_b and bounds_a and bounds_b
                        and epsg_a is not None and epsg_b is not None)
    same_grid = (dims_a == dims_b
                 and _vec_equal(scale_a, scale_b)
                 and _bounds_equal(bounds_a, bounds_b))
    if geo_complete and same_grid:
        verdict["co_registration"] = "verified"
        verdict["reasons"].append("identical georeferenced pixel grid "
                                  "(equal dims, CRS, resolution, bounds)")
    else:
        if dims_a != dims_b:
            verdict["warnings"].append(
                f"different pixel dimensions {dims_a[0]}×{dims_a[1]} vs "
                f"{dims_b[0]}×{dims_b[1]}; no resampling performed — "
                "spatial alignment unverified")
        elif not _vec_equal(scale_a, scale_b):
            verdict["warnings"].append(
                f"different pixel resolutions "
                f"({_fmt(scale_a[0])}/{_fmt(scale_b[0])} "
                f"{_unit_hint(meta_a)}); no resampling performed")
        if not geo_complete:
            verdict["warnings"].append(
                "co-registration unverified — insufficient geospatial metadata")

    _finalize_verdict(verdict)
    return verdict


def _finalize_verdict(verdict: dict) -> None:
    # De-duplicate while preserving order
    seen = set()
    for key in ("reasons", "warnings"):
        uniq = []
        for item in verdict[key]:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        verdict[key] = uniq
    bits = [f"status={verdict['status']}",
            f"co-registration={verdict['co_registration']}"]
    if verdict["reasons"]:
        bits.append("; ".join(verdict["reasons"]))
    if verdict["warnings"]:
        bits.append("warnings: " + " | ".join(verdict["warnings"]))
    verdict["summary"] = " · ".join(bits)


def _vec_equal(a, b) -> Optional[bool]:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    return abs(float(a[0]) - float(b[0])) <= 1e-9 * max(1.0, abs(float(a[0]))) \
        and abs(float(a[1]) - float(b[1])) <= 1e-9 * max(1.0, abs(float(b[1])))


def _bounds_equal(a, b, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return all(abs(float(a[k]) - float(b[k])) <= tol
               for k in ("left", "top", "right", "bottom"))


def _unit_hint(meta: dict) -> str:
    if meta.get("crs_type") == "geographic":
        return "deg"
    if meta.get("crs_type") == "projected":
        return "m"
    return "px-unit"


def _fmt(x) -> str:
    x = float(x)
    return f"{x:.6g}" if abs(x) >= 1e-4 else f"{x:.3e}"


def format_meta_line(label: str, meta: dict) -> str:
    """One compact human-readable line for an input image's metadata."""
    if not meta:
        return f"{label}: (no metadata)"
    parts = [f"{meta.get('width', '?')}×{meta.get('height', '?')}"]
    if meta.get("is_tiff"):
        parts.append(f"{meta.get('bands', '?')} band(s) {meta.get('dtype', '?')}")
        parts.append(meta.get("format", "tiff"))
    if meta.get("epsg"):
        parts.append(f"EPSG:{meta['epsg']}")
    if meta.get("pixel_scale"):
        parts.append(f"res {_fmt(meta['pixel_scale'][0])} {_unit_hint(meta)}/px")
    if not meta.get("is_tiff"):
        parts.append("no geospatial metadata")
    return f"{label}: " + " · ".join(parts)


def format_pair_compat(verdict: dict) -> str:
    """Compact block for the final answer / report."""
    if not verdict:
        return "pair compatibility not evaluated"
    lines = [f"- **{verdict.get('summary', '')}**"]
    for w in verdict.get("warnings", []):
        lines.append(f"- warning: {w}")
    return "\n".join(lines)
