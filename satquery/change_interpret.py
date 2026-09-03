"""
SatQuery AI — Change Interpretation (bi-temporal semantic description).

Division of responsibility:
  BIT-CD    → WHERE change occurred + HOW MUCH image area changed (authoritative)
  EarthDial → WHAT the detected changed areas visually appear to contain

This module is a thin orchestrator: it builds one evidence-grounded prompt,
composes T1|T2 into a single image for the existing single-image EarthDial
interface, runs exactly ONE VLM call, and returns a cautious text answer.

It never claims:
  - semantic change accuracy / object counts / class percentages
  - physical area (m²) — the pipeline has no georeferencing / GSD
  - geospatial coordinates
  - that the change percentage refers to a semantic class

No EarthDial or BIT-CD source is modified here.
"""

from __future__ import annotations

import os
import re
import tempfile
import time

# ── Interpretation request heuristic ─────────────────────────────

# Queries that only ask WHERE / HOW MUCH are answered by the existing
# BIT-CD mask + statistics; no VLM call is needed for them.
_ACTION_RE = re.compile(
    r"\b(describe|explain|summar|look|appear|tell me about)\b",
    re.IGNORECASE,
)
_LOCATION_RE = re.compile(
    r"\b(where|locate|location|which regions?|which areas?|show me)\b",
    re.IGNORECASE,
)
_STATS_RE = re.compile(
    r"\b(how much|how many|what percentage|what percent|percent)\b",
    re.IGNORECASE,
)


def should_interpret(query: str) -> tuple[bool, str]:
    """Decide whether a change query needs EarthDial interpretation.

    Returns (run: bool, reason: str). Ordered rules:
      1. Explicit description requests ("describe", "what does it look
         like", "explain") always trigger interpretation.
      2. Location-only queries ("where did changes occur?") are answered
         by the BIT-CD mask/regions alone.
      3. Statistics-only queries ("how much of the image changed?",
         "what percentage...?") are answered by BIT-CD alone.
      4. Anything else ("what changed?", "has construction increased?",
         "did anything change?") triggers interpretation.
    """
    q = query.strip().lower()
    if not q:
        return False, "empty query"
    if _ACTION_RE.search(q):
        return True, "query requests interpretation/description"
    if _LOCATION_RE.search(q):
        return False, "query asks only for change location; mask output sufficient"
    if _STATS_RE.search(q):
        return False, "query asks only for change statistics; mask output sufficient"
    return True, "query requests interpretation/description"


# ── Prompt construction ──────────────────────────────────────────

def build_region_table(regions) -> str:
    """Render the BIT-CD region list as prompt text (grid-space bboxes)."""
    rows = []
    for r in regions:
        bb = ",".join(str(int(v)) for v in r.bbox)
        rows.append(
            f"- Region R{r.region_id}: image-grid bbox [{bb}], "
            f"~{getattr(r, 'area_pct', 0.0):.1f}% of image area"
        )
    if not rows:
        return "No individual regions above the detection threshold."
    return "\n".join(rows)


def build_change_prompt(
    query: str,
    change_pct: float,
    num_regions: int,
    regions,
) -> str:
    """Build the evidence-grounded EarthDial prompt for change interpretation.

    Args:
        query:        the user's original question
        change_pct:   BIT-CD change percentage (image-area based)
        num_regions:  number of detected change regions
        regions:      BIT-CD ChangeRegion list (id, bbox in grid space, area_pct)
    """
    region_table = build_region_table(regions)
    return (
        "You are an expert remote sensing analyst looking at a side-by-side "
        "bi-temporal comparison.\n\n"
        "The image you are given shows the SAME area at TWO dates:\n"
        "  - LEFT half  = T1 (before / earlier image)\n"
        "  - RIGHT half = T2 (after / later image)\n\n"
        "A specialized change detector (BIT-CD) already located where pixel-level "
        "change was detected between T1 and T2. Its statistics are authoritative "
        "for WHERE the change is and HOW MUCH image area it covers.\n\n"
        "=== CHANGE STATISTICS (BIT-CD change mask — authoritative) ===\n"
        f"- Changed image area: {change_pct:.1f}% of the image area "
        "(percentage of image area, NOT of any semantic class)\n"
        f"- Number of detected change regions: {num_regions}\n"
        "- Detected change regions (bounding boxes are in the detector's "
        "image grid, not geospatial coordinates):\n"
        f"{region_table}\n\n"
        "=== USER QUESTION ===\n"
        f"{query}\n\n"
        "=== RULES ===\n"
        "1. Treat the BIT-CD change statistics as authoritative for change "
        "location and extent.\n"
        "2. Use the imagery ONLY to describe the visible appearance of the "
        "changed areas (what T2 now shows there compared with T1).\n"
        "3. Do NOT invent object counts (e.g. do not count buildings, roads, "
        "or vehicles).\n"
        "4. Do NOT estimate physical area in m² or hectares — you do not know "
        "the ground sampling distance or georeferencing.\n"
        "5. Do NOT invent semantic classes when they are not visually "
        "supported; only name a class if it is clearly visible in the imagery.\n"
        "6. Do NOT claim that buildings, roads, vegetation, or water changed "
        "unless the imagery provides visible evidence of it.\n"
        "7. Do NOT treat the change percentage as the percentage of a "
        "semantic class (e.g. do not say 'X% of built-up area changed').\n"
        "8. Do NOT provide geospatial coordinates — this comparison is not "
        "georeferenced.\n"
        "9. Phrase semantic conclusions as 'appears consistent with ...' when "
        "the evidence is qualitative.\n"
        "10. If you are uncertain or the changed areas are too small to "
        "describe, say so explicitly.\n\n"
        "Answer the user's question directly, referencing detected regions "
        "(R1, R2, ...) where useful, and clearly separating what the change "
        "statistics show from what the imagery appears to show."
    )


# ── T1 | T2 composite (single-image EarthDial interface) ─────────

_CELL = 256          # each temporal half is 256×256 (matches BIT-CD grid)
_CELLS = 2
_CANVAS_W = _CELL * _CELLS
_CANVAS_H = _CELL


def build_t1_t2_composite(
    t1_path: str,
    t2_path: str,
    out_path: str | None = None,
) -> str:
    """Create a deterministic T1|T2 side-by-side composite image.

    Layout (fixed): a single canvas, left half = T1 resized to 256×256,
    right half = T2 resized to 256×256, with 'T1 (before)' / 'T2 (after)'
    labels drawn in the top-left of each half. Both halves are square, so the
    model resize preserves their relative geometry.

    Returns the path of the written file. If out_path is None a temporary
    file is created (caller is responsible for cleanup).
    """
    from PIL import Image, ImageDraw, ImageFont

    im1 = Image.open(t1_path).convert("RGB").resize((_CELL, _CELL), Image.BICUBIC)
    im2 = Image.open(t2_path).convert("RGB").resize((_CELL, _CELL), Image.BICUBIC)

    canvas = Image.new("RGB", (_CANVAS_W, _CANVAS_H), (0, 0, 0))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (_CELL, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for label, x in (("T1 (before)", 6), ("T2 (after)", _CELL + 6)):
        draw.text((x + 1, 5), label, fill=(0, 0, 0), font=font)
        draw.text((x, 4), label, fill=(255, 255, 255), font=font)

    if out_path is None:
        fd, out_path = tempfile.mkstemp(prefix="satquery_change_", suffix=".png")
        os.close(fd)
    canvas.save(out_path)
    return out_path


# ── Runner ───────────────────────────────────────────────────────

def run_change_interpretation(
    vlm,
    t1_path: str,
    t2_path: str,
    change_result,
    query: str,
    max_tokens: int = 180,
) -> tuple[str | None, float]:
    """Run ONE EarthDial interpretation call over a T1|T2 composite.

    Returns (answer_text, elapsed_seconds). answer_text is None when the
    VLM call fails or returns nothing usable — the caller should then fall
    back to the BIT-CD statistics-only result (this never raises for model
    failures; only genuine local errors propagate and the caller catches them).

    The composite lives in a temporary directory that is removed when the
    call finishes, so no artifacts are left in the repository.
    """
    if not change_result.success:
        return None, 0.0
    if not getattr(change_result, "change_detected", False):
        return None, 0.0

    prompt = build_change_prompt(
        query=query,
        change_pct=float(getattr(change_result, "change_pct", 0.0) or 0.0),
        num_regions=int(getattr(change_result, "num_regions", 0) or 0),
        regions=list(getattr(change_result, "regions", []) or []),
    )

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="satquery_change_interp_") as tmp_dir:
        composite_path = build_t1_t2_composite(
            t1_path, t2_path, out_path=os.path.join(tmp_dir, "t1_t2_composite.png")
        )
        result = vlm.query(composite_path, prompt, max_tokens=max_tokens)
    elapsed = round(time.time() - t0, 1)

    if not getattr(result, "model_loaded", False):
        return None, elapsed
    answer = (result.answer or "").strip()
    if not answer:
        return None, elapsed
    return answer, elapsed
