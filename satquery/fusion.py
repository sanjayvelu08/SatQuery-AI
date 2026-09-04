"""
SatQuery AI — Evidence Fusion for Joint Optical + SAR Analysis.

Combines structured evidence from the specialists (EarthDial optical,
YOLOv8 SAR vessels, SAR-CLIP zero-shot scene labels, Otsu intensity
indicators) into a joint interpretation with traceable confidence.

The fusion is NOT prompt concatenation. It produces structured intermediate
results (FusedEvidence) and uses EarthDial for a separate joint-interpretation
call that receives a co-registered OPTICAL | SAR side-by-side composite plus
both evidences with explicit capability boundaries and honest uncertainty.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import TYPE_CHECKING

from .evidence import (
    FusedEvidence,
    JointAnalysisResult,
    OpticalEvidence,
    SAREvidence,
)

if TYPE_CHECKING:
    from .vlm import SatQueryVLM

_CELL = 256


def build_optical_sar_composite(
    optical_path: str,
    sar_path: str,
    out_path: str | None = None,
) -> str:
    """Create a deterministic OPTICAL | SAR side-by-side composite image.

    Layout (fixed, mirrors the proven T1|T2 change composite): a single
    canvas, left half = optical resized to 256×256, right half = SAR
    resized to 256×256, with 'OPTICAL' / 'SAR' labels. Both halves are
    square so the model resize preserves their relative geometry.

    Returns the path of the written file (temp file if out_path is None;
    caller is responsible for cleanup).
    """
    from PIL import Image, ImageDraw, ImageFont

    im1 = Image.open(optical_path).convert("RGB").resize(
        (_CELL, _CELL), Image.BICUBIC)
    im2 = Image.open(sar_path).convert("RGB").resize(
        (_CELL, _CELL), Image.BICUBIC)

    canvas = Image.new("RGB", (_CELL * 2, _CELL), (0, 0, 0))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (_CELL, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for label, x in (("OPTICAL", 6), ("SAR", _CELL + 6)):
        draw.text((x + 1, 5), label, fill=(0, 0, 0), font=font)
        draw.text((x, 4), label, fill=(255, 255, 255), font=font)

    if out_path is None:
        fd, out_path = tempfile.mkstemp(prefix="satquery_joint_", suffix=".png")
        os.close(fd)
    canvas.save(out_path)
    return out_path


def _render_scene_evidence(sar: SAREvidence) -> str:
    """Text block for SAR-CLIP scene labels (image-level, honest framing)."""
    if not (sar.success and sar.scene_scores):
        return "SAR-CLIP scene labelling unavailable."
    coarse = sar.scene_scores.get("coarse") or {}
    fine = sar.scene_scores.get("fine") or {}
    from .sarclip_tool import format_scene_scores
    lines = []
    if fine:
        lines.append(f"Scene labels, fine OpenEarthMap classes "
                     f"(model's native set): {format_scene_scores(fine, top=4)}")
    if coarse:
        lines.append(f"Scene labels, coarse generic classes (weak prior): "
                     f"{format_scene_scores(coarse, top=4)}")
    lines.append("(both zero-shot, image-level estimates — NOT pixel-level segmentation; "
                  "fine labels are more reliable than coarse)")
    return "\n".join(lines)


def _render_indicator_evidence(sar: SAREvidence) -> str:
    """Text block for Otsu intensity indicators (statistics, not semantics)."""
    if not (sar.success and sar.intensity_indicators):
        return "Intensity indicators unavailable."
    from .sarclip_tool import format_intensity_indicators
    return format_intensity_indicators(sar.intensity_indicators)


def fuse_evidence(optical: OpticalEvidence, sar: SAREvidence) -> FusedEvidence:
    """
    Structurally combine optical and SAR evidence.

    This is NOT text concatenation. It identifies what both sources
    can jointly assess and what remains unresolved.
    """
    joint_capabilities = []
    unresolved_gaps = []

    # What both sources can jointly address
    if optical.success and sar.success:
        if sar.num_detections > 0:
            joint_capabilities.append(
                "SAR-confirmed vessel presence with optical spatial context"
            )
        joint_capabilities.append(
            "Combined maritime + land-use scene understanding"
        )
        if sar.scene_scores and sar.scene_scores.get("coarse"):
            top = max(sar.scene_scores["coarse"].items(), key=lambda x: x[1])
            joint_capabilities.append(
                f"SAR scene evidence: dominant label '{top[0]}' ({top[1]*100:.1f}%) "
                "— image-level zero-shot estimate"
            )
        if sar.intensity_indicators:
            ind = sar.intensity_indicators
            if ind.get("dark_fraction", 0.0) > 0.5:
                joint_capabilities.append(
                    f"Intensity indicator: predominantly dark/water-like SAR "
                    f"returns ({ind['dark_fraction']*100:.1f}% of pixels)"
                )
            if ind.get("bright_fraction", 0.0) > 0.05:
                joint_capabilities.append(
                    f"Intensity indicator: bright/built-up-like SAR returns "
                    f"({ind['bright_fraction']*100:.1f}% of pixels)"
                )
    elif optical.success:
        joint_capabilities.append("Optical scene understanding (SAR unavailable)")
    elif sar.success:
        joint_capabilities.append("Maritime vessel detection (optical unavailable)")

    # What NEITHER specialist can address
    unresolved_gaps.extend([
        "Building age or construction timeline",
        "Vegetation health or environmental assessment",
        "Terrain elevation or topographic analysis",
        "Historical land-use change beyond vessel presence",
    ])

    # SAR-specific gaps (honest about limitations)
    if sar.success:
        unresolved_gaps.extend(sar.limitations)

    return FusedEvidence(
        optical=optical,
        sar=sar,
        joint_capabilities=joint_capabilities,
        unresolved_gaps=unresolved_gaps,
    )


def compute_confidence(
    fused: FusedEvidence,
    query: str,
) -> tuple[float, str]:
    """
    Compute an evidence-reliability score from observable factors.

    This is NOT prediction accuracy. It measures how much reliable
    evidence was available and whether sources are consistent.

    Returns (score 0.0–1.0, human-readable reasoning string).
    """
    factors = []
    reasons = []

    # Factor 1: Both specialists succeeded
    if fused.optical.success and fused.sar.success:
        factors.append(1.0)
        reasons.append("Both optical and SAR specialists succeeded")
    elif fused.optical.success:
        factors.append(0.5)
        reasons.append("Optical analysis succeeded; SAR specialist failed")
    elif fused.sar.success:
        factors.append(0.3)
        reasons.append("SAR detection succeeded; optical analysis failed")
    else:
        factors.append(0.0)
        reasons.append("Both specialists failed")

    # Factor 2: SAR detection quality (if available)
    if fused.sar.success:
        if fused.sar.num_detections > 0:
            # Higher confidence when detections exist with reasonable confidence
            factors.append(1.0)
            reasons.append(
                f"SAR detected {fused.sar.num_detections} vessel(s)"
            )
        else:
            # SAR ran but found nothing — could be expected or a miss
            factors.append(0.6)
            reasons.append("SAR ran but found no vessels (may be expected for non-maritime scenes)")

    # Factor 2b: extra evidence sources raise reliability
    extra = 0
    if fused.sar.success and fused.sar.scene_scores:
        extra += 1
    if fused.sar.success and fused.sar.intensity_indicators:
        extra += 1
    if extra:
        factors.append(0.9)
        reasons.append(f"{extra} additional SAR evidence source(s) "
                       "(scene labels / intensity indicators)")

    # Factor 3: Evidence consistency
    # Check if optical analysis mentions features that SAR findings support
    optical_lower = fused.optical.answer.lower() if fused.optical.success else ""
    if fused.sar.success and fused.sar.num_detections > 0:
        water_keywords = ["water", "ocean", "sea", "coast", "harbor", "port", "maritime", "bay"]
        has_water = any(kw in optical_lower for kw in water_keywords)
        if has_water:
            factors.append(1.0)
            reasons.append("Optical analysis mentions water/maritime features consistent with SAR vessel detections")
        else:
            factors.append(0.7)
            reasons.append("SAR found vessels but optical does not mention water features — possible spatial mismatch")

    # Factor 4: Unresolved gaps penalty
    n_gaps = len(fused.unresolved_gaps)
    if n_gaps > 0:
        # Small penalty for each unresolved gap
        gap_penalty = min(0.3, n_gaps * 0.03)
        factors.append(max(0.0, 1.0 - gap_penalty))
        reasons.append(f"{n_gaps} capability gap(s) acknowledged in output")

    # Compute final score
    if not factors:
        return 0.0, "No evidence available"

    confidence = sum(factors) / len(factors)
    confidence = round(min(1.0, max(0.0, confidence)), 2)
    reasoning = "; ".join(reasons)
    return confidence, reasoning


def build_optical_prompt_with_sar_context(
    original_query: str,
    sar_detection_summary: str,
    sar_capabilities: list[str],
    sar_limitations: list[str],
    sar_scene_scores: dict | None = None,
    sar_intensity_indicators: dict | None = None,
) -> str:
    """
    Build an EarthDial prompt for optical analysis that is INFORMED
    (not overwhelmed) by SAR context.

    The optical analysis runs first to produce clean OpticalEvidence.
    SAR context is used to guide attention, not replace optical understanding.
    """
    caps = "; ".join(sar_capabilities[:4])
    limits = "; ".join(sar_limitations[:3])

    extra = ""
    if sar_scene_scores:
        from .sarclip_tool import format_scene_scores
        fine = sar_scene_scores.get("fine") or {}
        coarse = sar_scene_scores.get("coarse") or {}
        if fine:
            extra += (f"\nSAR scene labels, fine OpenEarthMap classes (native, more "
                      f"reliable): {format_scene_scores(fine, top=4)}\n")
        if coarse:
            extra += (f"\nSAR scene labels, coarse generic (weak prior): "
                      f"{format_scene_scores(coarse, top=4)}\n")
    if sar_intensity_indicators:
        from .sarclip_tool import format_intensity_indicators
        extra += (f"\nSAR intensity indicators (image-relative statistics, "
                  f"NOT segmentation): {format_intensity_indicators(sar_intensity_indicators)}\n")

    return (
        "You are an expert remote sensing analyst performing optical image analysis.\n\n"
        "ADDITIONAL CONTEXT: A SAR (radar) analysis of the same geographic area was performed "
        "by specialized SAR tools (maritime vessel detector + zero-shot scene labelling).\n"
        f"SAR findings: {sar_detection_summary}\n"
        f"SAR capabilities: {caps}\n"
        f"SAR limitations: {limits}\n"
        f"{extra}"
        "Use this SAR context to guide your optical analysis, but base your observations "
        "solely on what you can see in the optical image.\n\n"
        f"User question: {original_query}\n\n"
        "Provide your analysis of the optical image, noting where SAR context is relevant."
    )


def build_joint_interpretation_prompt(
    original_query: str,
    optical_answer: str,
    sar_detection_summary: str,
    sar_limitations: list[str],
    sar_scene_scores: dict | None = None,
    sar_intensity_indicators: dict | None = None,
) -> str:
    """
    Build the final joint-interpretation prompt for EarthDial.

    The composite image (OPTICAL | SAR) is attached to this call, and the
    prompt separates, for the model, four evidence categories:
      * OPTICAL EVIDENCE
      * SAR EVIDENCE
      * DERIVED INDICATORS
      * UNCERTAIN OR INFERRED
    """
    fine_txt = ""
    coarse_txt = ""
    if sar_scene_scores:
        from .sarclip_tool import format_scene_scores
        if sar_scene_scores.get("fine"):
            fine_txt = format_scene_scores(sar_scene_scores["fine"], top=4)
        if sar_scene_scores.get("coarse"):
            coarse_txt = format_scene_scores(sar_scene_scores["coarse"], top=4)

    ind_txt = ""
    if sar_intensity_indicators:
        from .sarclip_tool import format_intensity_indicators
        ind_txt = format_intensity_indicators(sar_intensity_indicators)

    limits_text = "\n".join(f"- {l}" for l in sar_limitations[:6])

    return (
        "You are an expert remote sensor analyst performing joint multi-source analysis.\n\n"
        "You are shown a COMPOSITE image: left half = OPTICAL (multispectral) view, "
        "right half = SAR (radar) view of the same area. Treat the two halves as "
        "co-registered evidence of the same scene.\n\n"
        "=== OPTICAL EVIDENCE (EarthDial 4B Vision-Language Model) ===\n"
        f"{optical_answer}\n\n"
        "=== SAR EVIDENCE (YOLOv8 vessel detector + SAR-CLIP zero-shot labels) ===\n"
        f"{sar_detection_summary}\n"
        f"SAR scene labels, fine OpenEarthMap classes (native, more reliable): "
        f"{fine_txt if fine_txt else 'unavailable'}\n"
        f"SAR scene labels, coarse generic (weak prior): "
        f"{coarse_txt if coarse_txt else 'unavailable'}\n\n"
        "=== DERIVED INDICATORS (Otsu intensity statistics — NOT segmentation) ===\n"
        f"{ind_txt if ind_txt else 'unavailable'}\n\n"
        "=== UNCERTAIN OR INFERRED (do not present as fact) ===\n"
        "- Anything not directly supported by the evidence above is an inference; label it as such.\n"
        "- SAR scene labels are zero-shot image-level estimates, not pixel-level semantic segmentation.\n"
        "- The fine OpenEarthMap labels are the model's native classes and are more "
        "reliable than the coarse generic labels; if they disagree, treat the coarse "
        "generic label as a weak prior only and say so.\n"
        "- Intensity indicators are image-relative statistics, not semantic segmentation.\n"
        "- The SAR vessel detector recognizes ships only; do not claim it detected buildings or land cover.\n\n"
        "RULES:\n"
        "1. Analyse what you SEE in BOTH halves of the composite image (left optical, "
        "right SAR) before using the text evidence. Dark smooth areas in the SAR half "
        "indicate calm water; bright textured areas indicate strong radar returns "
        "(built-up-like).\n"
        "2. Weigh the evidence: optical appearance and the SAR dark-water indicator are "
        "strong for water; built-up needs optical appearance and/or bright SAR returns — "
        "the coarse generic scene label ALONE is never sufficient to claim built-up.\n"
        "3. When signals disagree (e.g., coarse scene label says built-up but the SAR half "
        "shows large dark water areas and the fine labels say water), say the signals "
        "disagree and explain which evidence supports which interpretation.\n"
        "4. When evidence is insufficient, say so explicitly.\n"
        "5. Never claim pixel-level semantic segmentation — you only have image-level "
        "scene labels and intensity statistics for the SAR side.\n\n"
        f"USER QUESTION: {original_query}\n\n"
        "Write a concise interpretation (4-8 sentences) of where built-up and water-covered "
        "regions are, combining BOTH halves:\n"
        "- what the optical half shows;\n"
        "- what the SAR half shows (dark smooth areas = calm water; bright returns = built-up-like);\n"
        "- what the numbers suggest: the Otsu dark/water-like fraction and bright/built-up-like "
        "fraction, and the fine vs coarse scene labels;\n"
        "- where the two modalities agree and where they disagree (e.g., coarse generic label vs "
        "fine native label, or scene label vs dark-water indicator);\n"
        "- what remains uncertain or inferred.\n"
        "Do NOT repeat the evidence lists verbatim and do NOT output bracketed section headers - "
        "just write the interpretation in plain sentences."
    )


def run_joint_interpretation(
    vlm: SatQueryVLM,
    original_query: str,
    optical_evidence: OpticalEvidence,
    sar_evidence: SAREvidence,
    composite_path: str | None = None,
) -> tuple[str, float]:
    """
    Run the second EarthDial call for joint interpretation.

    The call receives the OPTICAL | SAR composite image (built here if
    composite_path is not given) so the VLM can see both modalities.

    Returns (answer_text, inference_time_s).
    """
    sar_summary = (
        sar_evidence.detection_summary
        if sar_evidence.success and sar_evidence.detection_summary
        else "SAR analysis did not detect any maritime vessels."
    )

    image_for_query = optical_evidence.image_path
    if composite_path is not None and os.path.isfile(composite_path):
        image_for_query = composite_path

    prompt = build_joint_interpretation_prompt(
        original_query=original_query,
        optical_answer=optical_evidence.answer if optical_evidence.success else "Optical analysis failed.",
        sar_detection_summary=sar_summary,
        sar_limitations=sar_evidence.limitations if sar_evidence.success else [],
        sar_scene_scores=sar_evidence.scene_scores if sar_evidence.success else None,
        sar_intensity_indicators=sar_evidence.intensity_indicators if sar_evidence.success else None,
    )

    t0 = time.time()
    result = vlm.query(image_for_query, prompt, max_tokens=400)
    elapsed = time.time() - t0

    return result.answer, round(elapsed, 1)