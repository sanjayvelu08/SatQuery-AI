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
    """Output from YOLOv8 SAR specialist."""
    source: str
    image_path: str
    num_detections: int
    inference_time_ms: float
    gpu_vram_mb: float
    success: bool
    error: Optional[str] = None
    detection_summary: str = ""
    capabilities: List[str] = field(default_factory=lambda: [
        "maritime vessel detection",
        "ship presence/absence",
        "vessel bounding boxes with confidence",
    ])
    limitations: List[str] = field(default_factory=lambda: [
        "cannot detect buildings or infrastructure",
        "cannot classify land cover or terrain",
        "cannot detect vegetation or environmental changes",
        "trained only on maritime vessel data (YOLOv8 SAR)",
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
        """Human-readable markdown output."""
        lines = [
            "**🔗 Joint Optical + SAR Analysis**",
            "",
        ]

        if self.error:
            lines.append(f"**Error:** {self.error}")
            return "\n".join(lines)

        lines.append(self.joint_answer)
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
