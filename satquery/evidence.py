"""
SatQuery AI — Structured Evidence Types for Joint Analysis.

Defines the intermediate and final data structures for the
optical + SAR joint-analysis pipeline. Each specialist produces
typed evidence; the fusion layer combines them into a traceable result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OpticalEvidence:
    """Output from EarthDial VLM on a single optical image."""
    source: str
    answer: str
    intent: str
    prompt_sent: str
    image_path: str
    elapsed_s: float
    success: bool
    error: Optional[str] = None


@dataclass
class SAREvidence:
    """Output from the SAR specialists (YOLOv8 vessels + SAR-CLIP scene)."""
    source: str
    image_path: str
    num_detections: int
    inference_time_ms: float
    gpu_vram_mb: float
    success: bool
    error: Optional[str] = None
    detection_summary: str = ""
    # Zero-shot SAR-CLIP scene labels: {"coarse": {label: prob}, "fine": {...}}.
    # IMAGE-LEVEL estimates — not pixel-level segmentation.
    scene_scores: Optional[dict] = None
    # Otsu intensity indicators (image-relative statistics).
    intensity_indicators: Optional[dict] = None
    capabilities: List[str] = field(default_factory=lambda: [
        "maritime vessel detection",
        "ship presence/absence",
        "vessel bounding boxes with confidence",
        "zero-shot SAR scene labels (water, built-up, vegetation, agriculture)",
        "intensity indicators for water-like and built-up-like returns",
    ])
    limitations: List[str] = field(default_factory=lambda: [
        "cannot detect buildings or infrastructure with the vessel detector",
        "cannot classify land cover or terrain with the vessel detector",
        "trained only on maritime vessel data (YOLOv8 SAR)",
        "SAR-CLIP scene labels are zero-shot image-level estimates, "
        "not verified pixel-level semantic segmentation",
        "intensity indicators are image-relative statistics, not segmentation",
        "models not validated on RISAT imagery",
    ])


@dataclass
class FusedEvidence:
    """Combined evidence from optical + SAR analysis."""
    optical: OpticalEvidence
    sar: SAREvidence
    joint_capabilities: List[str] = field(default_factory=list)
    unresolved_gaps: List[str] = field(default_factory=list)


@dataclass
class ExecutionTraceStep:
    """One step in the auditable execution trace."""
    step: int
    name: str
    tool: str
    status: str  # "ok", "error", "skipped"
    duration_ms: float
    input_summary: str
    output_summary: str
    error: Optional[str] = None


@dataclass
class JointAnalysisResult:
    """Final output from optical + SAR joint analysis."""
    query: str
    optical_evidence: Optional[OpticalEvidence]
    sar_evidence: Optional[SAREvidence]
    fused_evidence: Optional[FusedEvidence]
    joint_answer: str
    confidence: float
    confidence_reasoning: str
    trace: List[ExecutionTraceStep]
    optical_annotated: Optional[str] = None
    sar_annotated: Optional[str] = None
    total_ms: float = 0.0
    sar_ms: float = 0.0
    optical_ms: float = 0.0
    fusion_ms: float = 0.0
    joint_interpretation_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def format_markdown(self) -> str:
        """Human-readable markdown output.

        The answer is assembled deterministically so it ALWAYS separates the
        four evidence categories required by the SIH 26167 workflow, regardless
        of the model's prose:
          1. OPTICAL EVIDENCE      — EarthDial on the optical image
          2. SAR EVIDENCE          — vessel detections + SAR-CLIP scene labels
          3. DERIVED INTENSITY INDICATORS — Otsu statistics (NOT segmentation)
          4. JOINT / UNCERTAIN CONCLUSION  — model interpretation + caveats
        """
        lines = [
            "**🔗 Joint Optical + SAR Analysis**",
            "",
        ]

        if self.error:
            lines.append(f"**Error:** {self.error}")
            return "\n".join(lines)

        # ── 1. Optical evidence (deterministic) ───────────────────────────
        lines.append("### 🔍 OPTICAL EVIDENCE")
        if self.optical_evidence and self.optical_evidence.success:
            lines.append(self.optical_evidence.answer.strip() or "No optical observations.")
        else:
            lines.append("Optical analysis failed or is unavailable.")
        lines.append("")

        # ── 2. SAR evidence (deterministic) ───────────────────────────────
        lines.append("### 🛰️ SAR EVIDENCE")
        if self.sar_evidence and self.sar_evidence.success:
            det = self.sar_evidence.detection_summary or (
                f"{self.sar_evidence.num_detections} vessel(s) detected")
            lines.append(f"- Vessel detector: {det}")
            scores = self.sar_evidence.scene_scores or {}
            fine = scores.get("fine") or {}
            coarse = scores.get("coarse") or {}
            if fine or coarse:
                from .sarclip_tool import format_scene_scores
                if fine:
                    lines.append(
                        f"- SAR-CLIP scene labels, fine OpenEarthMap classes "
                        f"(image-level, native, more reliable): "
                        f"{format_scene_scores(fine, top=4)}")
                if coarse:
                    lines.append(
                        f"- SAR-CLIP scene labels, coarse generic "
                        f"(image-level, weak prior): "
                        f"{format_scene_scores(coarse, top=4)}")
            lines.append("- SAR-CLIP scene labels are image-level estimates, "
                         "not pixel-level semantic segmentation.")
        else:
            lines.append("SAR analysis failed or is unavailable.")
        lines.append("")

        # ── 3. Derived intensity indicators (deterministic) ───────────────
        lines.append("### 📊 DERIVED INTENSITY INDICATORS")
        ind = self.sar_evidence.intensity_indicators if self.sar_evidence else None
        if ind:
            from .sarclip_tool import format_intensity_indicators
            lines.append(f"- {format_intensity_indicators(ind)}")
        else:
            lines.append("- Intensity indicators unavailable.")
        lines.append("- Image-relative Otsu statistics; NOT semantic segmentation.")
        lines.append("")

        # ── 4. Joint / uncertain conclusion (model prose) ──────────────────
        lines.append("### 🤝 JOINT CONCLUSION (AND WHAT REMAINS UNCERTAIN)")
        lines.append(self.joint_answer.strip() or "No joint interpretation produced.")
        lines.append("")
        lines.append("---")

        # SAR summary
        if self.sar_evidence and self.sar_evidence.success:
            lines.append(
                f"**SAR:** {self.sar_evidence.num_detections} vessel(s) detected "
                f"({self.sar_evidence.inference_time_ms:.0f}ms)"
            )
        elif self.sar_evidence:
            lines.append(f"**SAR:** {self.sar_evidence.error or 'detection failed'}")

        # Optical summary
        if self.optical_evidence and self.optical_evidence.success:
            lines.append(
                f"**Optical:** EarthDial analysis completed "
                f"({self.optical_evidence.elapsed_s:.1f}s)"
            )
        elif self.optical_evidence:
            lines.append(f"**Optical:** {self.optical_evidence.error or 'analysis failed'}")

        # Confidence
        lines.append(
            f"**Evidence Reliability:** {self.confidence:.2f} — {self.confidence_reasoning}"
        )

        # Timing
        lines.append(
            f"_Total: {self.total_ms:.0f}ms "
            f"(SAR {self.sar_ms:.0f}ms + Optical {self.optical_ms:.0f}ms + "
            f"Fusion {self.fusion_ms:.0f}ms + Interpretation {self.joint_interpretation_ms:.0f}ms)_"
        )

        # Models
        lines.append(f"_Models: {', '.join(self.models_used)}_")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "query": self.query,
            "joint_answer": self.joint_answer,
            "confidence": self.confidence,
            "confidence_reasoning": self.confidence_reasoning,
            "total_ms": self.total_ms,
            "models_used": self.models_used,
            "trace": [
                {
                    "step": t.step, "name": t.name, "tool": t.tool,
                    "status": t.status, "duration_ms": t.duration_ms,
                    "input_summary": t.input_summary,
                    "output_summary": t.output_summary,
                }
                for t in self.trace
            ],
        }
        if self.sar_evidence:
            d["sar"] = {
                "success": self.sar_evidence.success,
                "num_detections": self.sar_evidence.num_detections,
                "detection_summary": self.sar_evidence.detection_summary,
                "inference_time_ms": self.sar_evidence.inference_time_ms,
                "error": self.sar_evidence.error,
            }
        if self.optical_evidence:
            d["optical"] = {
                "success": self.optical_evidence.success,
                "elapsed_s": self.optical_evidence.elapsed_s,
                "error": self.optical_evidence.error,
            }
        return d
