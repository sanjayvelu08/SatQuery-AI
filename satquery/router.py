"""
SatQuery AI — Keyword-based intent router (v2, Loop 4).

Classifies natural-language queries into actionable intents.
Zero VRAM, zero latency, deterministic.

Changes from v1:
  - Added missing RS-specific keywords ("urban", "rural", "agricultural", etc.)
  - Improved prompt templates for each intent
  - Better general/caption disambiguation
  - Fixed "what type" → classification routing
"""

from dataclasses import dataclass

# ── Intent definitions ───────────────────────────────────────────────
# Each tuple: (intent_name, keywords, prompt_template, description)

_INTENT_RULES: list[tuple[str, list[str], str, str]] = [
    # Order matters: more specific intents first to avoid false matches
    (
        "change",
        ["change", "difference", "before", "after", "temporal",
         "compare", "diff", "changed", "changed between"],
        None,
        "Change detection (requires two images)",
    ),
    (
        "sar",
        ["sar", "radar", "synthetic aperture", "backscatter",
         "sentinel-1", "sar image", "sar imagery"],
        None,
        "SAR-specific analysis",
    ),
    (
        "detect",
        ["detect", "find", "locate", "identify", "show me", "point out",
         "where are the", "where are"],
        None,
        "Object detection / grounding",
    ),
    (
        "grounding",
        ["grounding", "bounding box", "box", "region", "area of",
         "draw a box", "mark the", "highlight"],
        None,
        "Visual grounding with coordinates",
    ),
    (
        "classification",
        ["classify", "class", "category", "type of scene",
         "type of area", "type of terrain", "scene type",
         "land cover", "land use", "landcover", "what type of"],
        None,
        "Scene classification",
    ),
    (
        "caption",
        ["describe", "caption", "summarize", "what is in", "what's in",
         "overview", "tell me about", "explain this", "what does this show",
         "what can you see", "walk me through"],
        None,
        "Image captioning / description",
    ),
    (
        "vqa",
        ["is there", "are there", "how many", "what color", "where is",
         "can you see", "does the", "do you see", "what kind", "which",
         "is this", "is it", "what is the", "what's the",
         "urban", "rural", "agricultural", "residential", "commercial",
         "water body", "forest", "vegetation"],
        None,
        "Visual question answering",
    ),
]

# ── Enhanced prompt templates ─────────────────────────────────────────

_PROMPTS = {
    "caption": (
        "You are an expert remote sensing analyst. "
        "Please describe this satellite image in detail. "
        "Include: the type of landscape (urban, agricultural, forest, water, etc.), "
        "key features visible (buildings, roads, fields, water bodies, vegetation), "
        "the general condition (clear, cloudy, seasonal), "
        "and any notable patterns or anomalies."
    ),
    "classification": (
        "You are an expert remote sensing analyst. "
        "Classify this satellite image into one or more of these categories: "
        "urban, suburban, rural, agricultural, forest, water body, desert, industrial, "
        "coastal, mountainous, wetland. "
        "Provide your classification with a brief justification."
    ),
    "detect": (
        "You are an expert remote sensing analyst. "
        "Detect and locate the main features in this satellite image. "
        "For each feature found, provide its approximate bounding box coordinates "
        "in the format [[x1, y1, x2, y2, confidence]] where coordinates are "
        "normalized 0-100. Describe each detected feature."
    ),
    "grounding": (
        "You are an expert remote sensing analyst. "
        "Identify and locate the key objects and features in this satellite image. "
        "Provide bounding box coordinates in [[x1, y1, x2, y2, confidence]] format "
        "for each detected item. Describe what each box contains."
    ),
    "general": (
        "You are an expert remote sensing analyst. "
        "Analyze this satellite image and respond to the user's question: "
    ),
}


@dataclass
class RouteResult:
    """Result of intent classification."""
    query: str
    primary_intent: str
    all_intents: list[str]
    prompt: str | None
    supported: bool
    reason: str = ""


def classify(query: str) -> RouteResult:
    """Classify a user query and produce the EarthDial prompt to send."""
    q = query.lower()
    matches: list[tuple[str, str]] = []  # (intent, matched_keyword)

    for intent, keywords, _tmpl, _desc in _INTENT_RULES:
        for kw in keywords:
            if kw in q:
                matches.append((intent, kw))
                break

    if not matches:
        matches.append(("general", ""))

    all_intents = [m[0] for m in matches]
    primary = all_intents[0]

    # ── Determine prompt & support status ─────────────────────────
    if primary == "change":
        return RouteResult(
            query=query, primary_intent=primary, all_intents=all_intents,
            prompt=None, supported=False,
            reason=(
                "Change detection requires two images (bi-temporal comparison). "
                "This feature is not yet implemented. "
                "Please provide a single-image query instead."
            ),
        )

    if primary == "sar":
        return RouteResult(
            query=query, primary_intent=primary, all_intents=all_intents,
            prompt=None, supported=False,
            reason=(
                "SAR (Synthetic Aperture Radar) analysis requires specialized "
                "models that are not yet integrated. "
                "Currently only optical satellite imagery is supported. "
                "Try uploading an optical/Sentinel-2 image instead."
            ),
        )

    if primary in ("detect", "grounding"):
        prompt = _PROMPTS.get(primary, _PROMPTS["detect"])
        return RouteResult(
            query=query, primary_intent=primary, all_intents=all_intents,
            prompt=prompt, supported=True,
        )

    if primary == "vqa":
        # Pass the user's question directly but with RS context
        prompt = (
            "You are an expert remote sensing analyst. "
            f"Based on this satellite image, answer: {query}"
        )
        return RouteResult(
            query=query, primary_intent=primary, all_intents=all_intents,
            prompt=prompt, supported=True,
        )

    if primary == "classification":
        return RouteResult(
            query=query, primary_intent=primary, all_intents=all_intents,
            prompt=_PROMPTS["classification"], supported=True,
        )

    if primary == "caption":
        return RouteResult(
            query=query, primary_intent=primary, all_intents=all_intents,
            prompt=_PROMPTS["caption"], supported=True,
        )

    # general / unknown — prepend RS context
    prompt = _PROMPTS["general"] + query
    return RouteResult(
        query=query, primary_intent=primary, all_intents=all_intents,
        prompt=prompt, supported=True,
    )
