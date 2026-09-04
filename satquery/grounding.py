"""Query-conditioned text-guided grounding for EarthDial.

This module makes grounding *genuinely text-conditioned*: the user's target
phrase (e.g. "water body", "airport") is extracted from the query and passed
explicitly into the EarthDial grounding prompt, instead of asking the model to
box generic "key objects and features".

Coordinate frame contract
-------------------------
EarthDial's eval preprocessing (``build_transform(is_train=False,
input_size=448, normalize_type="imagenet")`` in EarthDial/src/
earthdial/train/dataset.py) applies a *non-uniform squash* resize
``T.Resize((448, 448))`` — NOT letterboxing or padding.  Consequently,
normalized coordinates (0-100) in the model frame map *linearly* back to the
original image pixels:

    x_px = x_norm / 100 * image_width
    y_px = y_norm / 100 * image_height

There is no padding offset to subtract.  ``normalized_to_pixel`` implements
that exact inverse mapping; the pipeline feeds it the dimensions of the actual
image the worker consumed (after GeoTIFF RGB rendering when applicable).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Target phrase extraction ────────────────────────────────────

# Leading wrapper verbs/phrases; the target is whatever remains.
_WRAPPER_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"highlight|mark|show|locate|find|identify|detect|indicate|"
    r"point\s+out|outline|circle|"
    r"draw\s+(?:a\s+|the\s+)?(?:bounding\s+)?box(?:es)?\s*(?:around|for|on)?|"
    r"give\s+(?:me\s+)?(?:the\s+)?(?:bounding\s+)?box(?:es)?\s*(?:for|of)?|"
    r"bounding\s+box(?:es)?\s*(?:for|of)?|"
    r"where\s+(?:is|are)\s+|where's\s+|"
    r"can\s+you\s+(?:show|locate|find|mark|highlight|point\s+out)\s+"
    r")\b[\s,:]*",
    re.IGNORECASE,
)

# Trailing qualifier phrases (e.g. "referred to in the query", "in the image").
_TRAILER_RE = re.compile(
    r"\s*(?:"
    r"referred\s+to\s+(?:in|by)\s+(?:the\s+)?(?:query|prompt|text|sentence|image|scene|photo)|"
    r"in\s+(?:this\s+|the\s+)?(?:satellite\s+|aerial\s+|remote\s+)?(?:image|scene|photo|picture)|"
    r"on\s+(?:this\s+|the\s+)?(?:satellite\s+|aerial\s+)?(?:image|scene|photo)|"
    r"for\s+me|in\s+the\s+query"
    r")[.!?\s]*$",
    re.IGNORECASE,
)

_ARTICLE_RE = re.compile(r"^(?:the|a|an|some|any|all)\s+", re.IGNORECASE)

# Generic words that carry no target information.
_STOPWORDS = {
    "it", "this", "them", "those", "these", "that", "one",
    "object", "objects", "feature", "features", "thing", "things",
    "item", "items", "everything", "anything", "something", "target",
    "image", "scene", "photo", "picture", "satellite image",
    "aerial image", "area", "region", "content", "contents",
}
_GENERIC_PATTERN = re.compile(
    r"^(?:key\s+|main\s+|primary\s+|major\s+|all\s+|the\s+)?"
    r"(?:objects?|features?|items?|things?)(?:\s+and\s+.*)?$",
    re.IGNORECASE,
)
# Description/QA-style commands that are not grounding requests at all.
_NON_GROUNDING_PREFIXES = (
    "describe", "description", "summarize", "overview",
    "tell me about", "what is in", "what's in", "explain",
    "classify", "what type of",
)


def extract_target(query: str) -> Optional[str]:
    """Extract the target entity from a grounding-style query.

    Examples:
        "Highlight the water body referred to in the query." -> "water body"
        "Locate the airport."                                -> "airport"
        "Show me the buildings"                              -> "buildings"
        "Where is the river?"                                -> "river"

    Returns None when no specific target can be identified (the caller then
    falls back to the generic detection prompt rather than inventing one).
    """
    if not query:
        return None
    text = query.strip().strip('"\'“”‘’')
    text = _WRAPPER_RE.sub("", text)
    text = _TRAILER_RE.sub("", text)
    text = text.strip(" .!?,;:-\"'“”‘’")
    # Drop leading "me/us" left over after a wrapper like "Show me ...".
    text = re.sub(r"^(?:me|us)\s+", "", text, flags=re.IGNORECASE).strip()
    # Strip articles repeatedly ("all the water bodies" -> "water bodies").
    while _ARTICLE_RE.search(text):
        text = _ARTICLE_RE.sub("", text).strip()
    text = text.strip(" .!?,;:-")
    if not text:
        return None
    low = text.lower()
    if low in _STOPWORDS or _GENERIC_PATTERN.match(low):
        return None
    if low.startswith(_NON_GROUNDING_PREFIXES):
        return None
    return text


# ── Prompt building ─────────────────────────────────────────────

_BOX_CONTRACT = "[[x1, y1, x2, y2, confidence]]"


def build_grounding_prompt(target: str) -> str:
    """Prompt EarthDial to box *the requested target* (not generic features).

    Keeps the existing normalized [0-100] coordinate contract so the answer
    remains machine-parseable by ``parse_grounding_bboxes``.
    """
    return (
        "You are an expert remote sensing analyst. "
        "Locate every instance of the following target in this satellite image: "
        f"'{target}'. "
        f"For each instance provide its bounding box in the exact format "
        f"{_BOX_CONTRACT}, where coordinates are normalized 0-100. "
        "Respond with the bounding box(es) and a one-sentence description of "
        f"what each box contains. If no {target} is visible, say so plainly and "
        "do not output any bounding box."
    )


# ── Output parsing ──────────────────────────────────────────────

# Strict contract: [[x1, y1, x2, y2]] or [[x1, y1, x2, y2, confidence]].
_BOX_PATTERN = re.compile(
    r"\[\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*"
    r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*"
    r"(?:,\s*(\d+(?:\.\d+)?))?\s*\]\]"
)


def parse_grounding_bboxes(answer: str) -> list[dict]:
    """Parse strict [[x1,y1,x2,y2,(conf)]] boxes (normalized 0-100).

    Only tokens matching the exact output contract are accepted — prose is
    ignored.  Coordinates outside [0, 100] and non-finite values are dropped.
    ``confidence`` is None when the model did not supply one (never invented).
    """
    if not answer:
        return []
    boxes: list[dict] = []
    for m in _BOX_PATTERN.finditer(answer):
        try:
            vals = [float(m.group(i)) for i in range(1, 5)]
            if any(v < 0 or v > 100 for v in vals):
                continue
            conf_raw = m.group(5)
            conf = float(conf_raw) / 100.0 if conf_raw is not None else None
            if conf is not None and (conf < 0 or conf > 1):
                continue
            x1, y1, x2, y2 = vals
            if x2 < x1 or y2 < y1:
                continue
            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": conf,
            })
        except (ValueError, TypeError):
            continue
    return boxes


def normalized_to_pixel(
    box: dict, image_width: int, image_height: int
) -> dict:
    """Inverse of EarthDial's squash-resize: normalized 0-100 -> pixels.

    Exact inverse of the verified ``T.Resize((448, 448))`` non-uniform squash
    (no letterbox padding), so the mapping is linear per axis.
    """
    return {
        "x1": box["x1"] / 100.0 * image_width,
        "y1": box["y1"] / 100.0 * image_height,
        "x2": box["x2"] / 100.0 * image_width,
        "y2": box["y2"] / 100.0 * image_height,
    }


@dataclass
class GroundingDetection:
    """One structured grounding box for a target, in both frames."""

    target: str
    x1: float  # normalized 0-100
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None  # fraction 0-1, None if not provided

    def to_dict(self, image_width: int, image_height: int) -> dict:
        px = normalized_to_pixel(
            {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2},
            image_width, image_height,
        )
        return {
            "target": self.target,
            "x1": round(px["x1"], 1), "y1": round(px["y1"], 1),
            "x2": round(px["x2"], 1), "y2": round(px["y2"], 1),
            "x1_norm": round(self.x1, 2), "y1_norm": round(self.y1, 2),
            "x2_norm": round(self.x2, 2), "y2_norm": round(self.y2, 2),
            "confidence": self.confidence,
        }