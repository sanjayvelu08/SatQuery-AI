"""
SatQuery AI — Evidence Fusion for Joint Optical + SAR Analysis.

Combines structured evidence from two specialists (EarthDial optical + YOLOv8 SAR)
into a joint interpretation with traceable confidence scoring.

The fusion is NOT prompt concatenation. It produces structured intermediate results
(FusedEvidence) and uses EarthDial for a separate joint-interpretation call that
receives both evidences with explicit capability boundaries.
"""

from __future__ import annotations

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
) -> str:
    """
    Build an EarthDial prompt for optical analysis that is INFORMED
    (not overwhelmed) by SAR context.

    The optical analysis runs first to produce clean OpticalEvidence.
    SAR context is used to guide attention, not replace optical understanding.
    """
    caps = "; ".join(sar_capabilities[:3])
    limits = "; ".join(sar_limitations[:3])

    return (
        "You are an expert remote sensing analyst performing optical image analysis.\n\n"
        "ADDITIONAL CONTEXT: A SAR (radar) analysis of the same geographic area was performed "
        "by a specialized maritime vessel detector.\n"
        f"SAR findings: {sar_detection_summary}\n"
        f"SAR capabilities: {caps}\n"
        f"SAR limitations: {limits}\n\n"
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
) -> str:
    """
    Build the final joint-interpretation prompt for EarthDial.

    This is the second EarthDial call that receives BOTH evidences
    and produces the user-facing answer.
    """
    limits_text = "\n".join(f"- {l}" for l in sar_limitations[:4])

    return (
        "You are an expert remote sensor analyst performing joint multi-source analysis.\n\n"
        "You have evidence from two independent specialists:\n\n"
        "=== OPTICAL IMAGE ANALYSIS (EarthDial 4B Vision-Language Model) ===\n"
        f"{optical_answer}\n\n"
        "=== SAR IMAGE ANALYSIS (YOLOv8 Maritime Vessel Detector) ===\n"
        f"{sar_detection_summary}\n\n"
        "CRITICAL CONSTRAINTS:\n"
        f"The SAR specialist is a vessel-only detector. It CANNOT:\n"
        f"{limits_text}\n\n"
        "RULES:\n"
        "1. Clearly label which evidence source supports each claim.\n"
        "2. Never claim SAR detected anything outside its vessel-detection capability.\n"
        "3. Do not invent facts not present in either evidence source.\n"
        "4. When evidence is insufficient to answer a part of the question, say so explicitly.\n"
        "5. If SAR found vessels and optical shows water/maritime features, note the consistency.\n"
        "6. If SAR found vessels but optical shows no water, note the discrepancy.\n\n"
        f"USER QUESTION: {original_query}\n\n"
        "Provide your joint analysis:"
    )


def run_joint_interpretation(
    vlm: SatQueryVLM,
    original_query: str,
    optical_evidence: OpticalEvidence,
    sar_evidence: SAREvidence,
) -> tuple[str, float]:
    """
    Run the second EarthDial call for joint interpretation.

    Returns (answer_text, inference_time_s).
    """
    sar_summary = (
        sar_evidence.detection_summary
        if sar_evidence.success and sar_evidence.detection_summary
        else "SAR analysis did not detect any maritime vessels."
    )

    prompt = build_joint_interpretation_prompt(
        original_query=original_query,
        optical_answer=optical_evidence.answer if optical_evidence.success else "Optical analysis failed.",
        sar_detection_summary=sar_summary,
        sar_limitations=sar_evidence.limitations if sar_evidence.success else [],
    )

    t0 = time.time()
    result = vlm.query(optical_evidence.image_path, prompt)
    elapsed = time.time() - t0

    return result.answer, round(elapsed, 1)
