"""
SatQuery AI — Core pipeline (v2, Loop 4).

Changes from v1:
  - Added QueryHistory for last N queries
  - Improved error handling
  - PipelineResult now includes image_path for display
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .router import classify, RouteResult
from .vlm import SatQueryVLM, InferenceResult
from .sar_tool import run_sar_detection, format_sar_response, SARResult
from .visualize import create_annotated_image


@dataclass
class PipelineResult:
    """Full structured output from the SatQuery pipeline."""
    query: str
    image_path: str
    intent: str
    all_intents: list[str]
    supported: bool
    answer: str | None = None
    unsupported_reason: str = ""
    model_used: str = ""  # e.g. "EarthDial 4B", "YOLOv8 SAR"
    annotated_image: str | None = None  # path to annotated image with bboxes
    sar_result: SARResult | None = None  # raw SAR detection data
    change_result: object | None = None  # ChangeDetectionResult from bit_tool
    joint_result: object | None = None  # JointAnalysisResult from fusion
    image_t2_path: str | None = None  # second image for change detection
    image_sar_path: str | None = None  # SAR image for joint analysis
    elapsed_route_ms: float = 0.0
    elapsed_vlm_s: float = 0.0
    elapsed_total_s: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "query": self.query,
            "image_path": self.image_path,
            "intent": self.intent,
            "all_intents": self.all_intents,
            "supported": self.supported,
            "answer": self.answer,
            "unsupported_reason": self.unsupported_reason,
            "model_used": self.model_used,
            "annotated_image": self.annotated_image,
            "elapsed_route_ms": self.elapsed_route_ms,
            "elapsed_vlm_s": self.elapsed_vlm_s,
            "elapsed_total_s": self.elapsed_total_s,
        }
        if self.image_t2_path:
            d["image_t2_path"] = self.image_t2_path
        if self.image_sar_path:
            d["image_sar_path"] = self.image_sar_path
        if self.change_result is not None:
            cr = self.change_result
            d["change_result"] = cr.to_dict() if hasattr(cr, "to_dict") else str(cr)
        if self.joint_result is not None:
            jr = self.joint_result
            d["joint_result"] = jr.to_dict() if hasattr(jr, "to_dict") else str(jr)
        return d

    def format(self) -> str:
        """Human-readable formatted output."""
        lines = [
            f"Query:   {self.query}",
            f"Image:   {self.image_path}",
            f"Intent:  {self.intent} ({', '.join(self.all_intents)})",
        ]
        if not self.supported:
            lines.append(f"Status:  UNSUPPORTED — {self.unsupported_reason}")
        else:
            lines.append(f"Answer:  {self.answer}")
        lines.append(
            f"Timing:  route={self.elapsed_route_ms:.0f}ms  "
            f"vlm={self.elapsed_vlm_s:.1f}s  total={self.elapsed_total_s:.1f}s"
        )
        return "\n".join(lines)


class QueryHistory:
    """Stores the last N query results."""

    def __init__(self, max_size: int = 5):
        self._history: deque[PipelineResult] = deque(maxlen=max_size)

    def add(self, result: PipelineResult) -> None:
        self._history.append(result)

    def get_all(self) -> list[PipelineResult]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


class SatQueryPipeline:
    """Orchestrates router → VLM → output for single-image queries."""

    def __init__(self, vlm: SatQueryVLM | None = None, max_history: int = 5):
        self.vlm = vlm or SatQueryVLM()
        self.history = QueryHistory(max_size=max_history)

    def run(
        self, image_path: str, query: str,
        image_t2_path: str | None = None,
        image_sar_path: str | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline for one (image, query) pair.

        For change detection, image_t2_path must be provided.
        For joint optical+SAR analysis, image_sar_path must be provided.
        """
        t_total = time.time()

        # ── Step 1: Route ─────────────────────────────────────────
        t0 = time.time()
        route: RouteResult = classify(query)
        elapsed_route = (time.time() - t0) * 1000  # ms

        # ── Step 1.5: Joint analysis ────────────────────────────
        if route.primary_intent == "joint_analysis":
            from .evidence import (
                OpticalEvidence, SAREvidence, ExecutionTraceStep,
                JointAnalysisResult,
            )
            from .fusion import (
                fuse_evidence, compute_confidence,
                build_optical_prompt_with_sar_context,
                run_joint_interpretation,
            )
            import os

            trace_steps: list[ExecutionTraceStep] = []
            t_joint = time.time()

            # ── Validate inputs ─────────────────────────────────
            t_val = time.time()
            val_errors = []
            if not image_path:
                val_errors.append("Optical image is required for joint analysis.")
            if not image_sar_path:
                val_errors.append("SAR image is required for joint analysis.")
            for label, path in [("Optical", image_path), ("SAR", image_sar_path)]:
                if path and not os.path.exists(path):
                    val_errors.append(f"{label} image not found: {path}")
                elif path:
                    try:
                        from PIL import Image
                        Image.open(path).verify()
                    except Exception:
                        val_errors.append(f"{label} image is corrupt or unreadable.")
            if not query or not query.strip():
                val_errors.append("No query provided.")
            val_ms = (time.time() - t_val) * 1000

            trace_steps.append(ExecutionTraceStep(
                step=2, name="validate", tool="input_validator",
                status="ok" if not val_errors else "error",
                duration_ms=round(val_ms, 1),
                input_summary=f"optical={'✓' if image_path else '✗'} sar={'✓' if image_sar_path else '✗'} query={'✓' if query else '✗'}",
                output_summary="; ".join(val_errors) if val_errors else "All inputs valid",
                error="; ".join(val_errors) if val_errors else None,
            ))

            if val_errors:
                joint_ms = (time.time() - t_joint) * 1000
                result = PipelineResult(
                    query=query, image_path=image_path, image_sar_path=image_sar_path,
                    intent=route.primary_intent, all_intents=route.all_intents,
                    supported=False,
                    unsupported_reason="\n".join(val_errors),
                    elapsed_route_ms=elapsed_route,
                    elapsed_total_s=round(time.time() - t_total, 1),
                )
                self.history.add(result)
                return result

            # ── Step 3: SAR detection ───────────────────────────
            t_sar = time.time()
            from .sar_tool import run_sar_detection, format_sar_response
            sar_raw = run_sar_detection(image_sar_path)
            sar_ms = (time.time() - t_sar) * 1000

            sar_evidence = SAREvidence(
                source="yolov8_sar_vessel",
                image_path=image_sar_path,
                num_detections=sar_raw.num_detections,
                inference_time_ms=sar_raw.inference_time_ms,
                gpu_vram_mb=sar_raw.gpu_vram_mb,
                success=sar_raw.success,
                error=sar_raw.error,
                detection_summary=format_sar_response(sar_raw),
            )

            trace_steps.append(ExecutionTraceStep(
                step=3, name="sar_detect", tool="yolov8_sar",
                status="ok" if sar_raw.success else "error",
                duration_ms=round(sar_ms, 1),
                input_summary=f"SAR image: {os.path.basename(image_sar_path)}",
                output_summary=f"{sar_raw.num_detections} vessel(s) detected" if sar_raw.success else (sar_raw.error or "failed"),
                error=sar_raw.error,
            ))

            # ── Step 4: Optical analysis ────────────────────────
            t_opt = time.time()
            sar_summary_for_prompt = (
                sar_evidence.detection_summary
                if sar_evidence.success and sar_evidence.detection_summary
                else "No SAR detections."
            )
            optical_prompt = build_optical_prompt_with_sar_context(
                original_query=query,
                sar_detection_summary=sar_summary_for_prompt,
                sar_capabilities=sar_evidence.capabilities,
                sar_limitations=sar_evidence.limitations,
            )
            optical_result = self.vlm.query(image_path, optical_prompt)
            optical_ms = (time.time() - t_opt) * 1000

            optical_evidence = OpticalEvidence(
                source="earthdial_4b",
                answer=optical_result.answer,
                intent="joint_context",
                prompt_sent=optical_prompt,
                image_path=image_path,
                elapsed_s=optical_result.elapsed_s,
                success=optical_result.model_loaded,
            )

            trace_steps.append(ExecutionTraceStep(
                step=4, name="optical_analyze", tool="earthdial_4b",
                status="ok" if optical_evidence.success else "error",
                duration_ms=round(optical_ms, 1),
                input_summary=f"Optical image: {os.path.basename(image_path)}",
                output_summary=f"{len(optical_evidence.answer)} chars response",
            ))

            # ── Step 5: Evidence fusion ──────────────────────────
            t_fuse = time.time()
            fused = fuse_evidence(optical_evidence, sar_evidence)
            fuse_ms = (time.time() - t_fuse) * 1000

            trace_steps.append(ExecutionTraceStep(
                step=5, name="fuse", tool="evidence_fusion",
                status="ok", duration_ms=round(fuse_ms, 1),
                input_summary=f"optical_ok={optical_evidence.success} sar_ok={sar_evidence.success}",
                output_summary=f"{len(fused.joint_capabilities)} capabilities, {len(fused.unresolved_gaps)} gaps",
            ))

            # ── Step 6: Joint interpretation ─────────────────────
            t_interp = time.time()
            joint_answer, interp_ms = run_joint_interpretation(
                vlm=self.vlm,
                original_query=query,
                optical_evidence=optical_evidence,
                sar_evidence=sar_evidence,
            )
            interp_ms_real = (time.time() - t_interp) * 1000

            trace_steps.append(ExecutionTraceStep(
                step=6, name="interpret", tool="earthdial_4b",
                status="ok", duration_ms=round(interp_ms_real, 1),
                input_summary="optical + sar evidence",
                output_summary=f"{len(joint_answer)} chars joint answer",
            ))

            # ── Step 7: Confidence ───────────────────────────────
            t_conf = time.time()
            confidence, conf_reasoning = compute_confidence(fused, query)
            conf_ms = (time.time() - t_conf) * 1000

            trace_steps.append(ExecutionTraceStep(
                step=7, name="confidence", tool="confidence_calc",
                status="ok", duration_ms=round(conf_ms, 1),
                input_summary="evidence quality factors",
                output_summary=f"reliability={confidence:.2f}",
            ))

            # ── Assemble result ──────────────────────────────────
            joint_ms = (time.time() - t_joint) * 1000

            # SAR annotated image (if detections found)
            sar_annotated = None
            if sar_raw.success and sar_raw.num_detections > 0:
                try:
                    sar_annotated = create_annotated_image(
                        image_sar_path, sar_evidence.detection_summary, "sar"
                    )
                except Exception:
                    pass

            joint_result = JointAnalysisResult(
                query=query,
                optical_evidence=optical_evidence,
                sar_evidence=sar_evidence,
                fused_evidence=fused,
                joint_answer=joint_answer,
                confidence=confidence,
                confidence_reasoning=conf_reasoning,
                trace=trace_steps,
                sar_annotated=sar_annotated,
                total_ms=round(joint_ms, 0),
                sar_ms=round(sar_ms, 0),
                optical_ms=round(optical_ms, 0),
                fusion_ms=round(fuse_ms, 0),
                joint_interpretation_ms=round(interp_ms_real, 0),
                models_used=["YOLOv8 SAR Vessel Detector", "EarthDial 4B RGB"],
            )

            result = PipelineResult(
                query=query,
                image_path=image_path,
                image_sar_path=image_sar_path,
                intent=route.primary_intent,
                all_intents=route.all_intents,
                supported=True,
                answer=joint_result.format_markdown(),
                model_used="EarthDial 4B + YOLOv8 SAR (Joint)",
                annotated_image=sar_annotated or optical_evidence.image_path,
                sar_result=sar_raw,
                joint_result=joint_result,
                elapsed_route_ms=elapsed_route,
                elapsed_vlm_s=optical_result.elapsed_s + interp_ms,
                elapsed_total_s=round(time.time() - t_total, 1),
            )
            self.history.add(result)
            return result

        # ── Step 1.6: Change intent validation ────────────────────
        if route.primary_intent == "change":
            if not image_t2_path:
                result = PipelineResult(
                    query=query, image_path=image_path, intent=route.primary_intent,
                    all_intents=route.all_intents, supported=False,
                    unsupported_reason=(
                        "**Two images required for change detection.**\n\n"
                        "Please upload both a **Before (T1)** and **After (T2)** image, "
                        "then enter your change query."
                    ),
                    elapsed_route_ms=elapsed_route,
                    elapsed_total_s=round(time.time() - t_total, 1),
                )
                self.history.add(result)
                return result

            # Run BIT-CD change detection
            from .bit_tool import get_bit_tool
            import os

            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "change_output"
            )
            bit = get_bit_tool()
            change_result = bit.detect(
                image_t1_path=image_path,
                image_t2_path=image_t2_path,
                output_dir=output_dir,
            )

            # Use overlay as annotated image if available
            annotated = change_result.overlay_path

            result = PipelineResult(
                query=query,
                image_path=image_path,
                image_t2_path=image_t2_path,
                intent=route.primary_intent,
                all_intents=route.all_intents,
                supported=change_result.success,
                answer=change_result.format_markdown() if change_result.success else f"Error: {change_result.error}",
                unsupported_reason="" if change_result.success else (change_result.error or "Change detection failed"),
                model_used="BIT-CD (LEVIR-CD pretrained)",
                annotated_image=annotated,
                change_result=change_result,
                elapsed_route_ms=elapsed_route,
                elapsed_total_s=round(time.time() - t_total, 1),
            )
            self.history.add(result)
            return result

        # ── Step 2: SAR intent → SAR tool ────────────────────────
        if route.primary_intent == "sar":
            sar_result = run_sar_detection(image_path)
            answer_text = format_sar_response(sar_result)

            # Generate annotated image if detections found
            annotated = None
            if sar_result.success and sar_result.num_detections > 0:
                try:
                    annotated = create_annotated_image(
                        image_path, answer_text, "sar"
                    )
                except Exception:
                    pass  # Non-critical

            result = PipelineResult(
                query=query,
                image_path=image_path,
                intent=route.primary_intent,
                all_intents=route.all_intents,
                supported=sar_result.success,
                answer=answer_text,
                unsupported_reason="" if sar_result.success else (sar_result.error or "SAR tool failed"),
                model_used="YOLOv8 SAR Vessel Detector",
                annotated_image=annotated,
                sar_result=sar_result,
                elapsed_route_ms=elapsed_route,
                elapsed_total_s=round(time.time() - t_total, 1),
            )
            self.history.add(result)
            return result

        # ── Step 3: Unsupported? ──────────────────────────────────
        if not route.supported:
            result = PipelineResult(
                query=query,
                image_path=image_path,
                intent=route.primary_intent,
                all_intents=route.all_intents,
                supported=False,
                unsupported_reason=route.reason,
                elapsed_route_ms=elapsed_route,
                elapsed_total_s=round(time.time() - t_total, 1),
            )
            self.history.add(result)
            return result

        # ── Step 4: VLM inference ─────────────────────────────────
        vlm_result: InferenceResult = self.vlm.query(image_path, route.prompt)

        # Generate annotated image for grounding/detect intents
        annotated = None
        if route.primary_intent in ("detect", "grounding") and vlm_result.answer:
            try:
                annotated = create_annotated_image(
                    image_path, vlm_result.answer, route.primary_intent
                )
            except Exception:
                pass  # Non-critical

        # ── Step 5: Assemble output ───────────────────────────────
        result = PipelineResult(
            query=query,
            image_path=image_path,
            intent=route.primary_intent,
            all_intents=route.all_intents,
            supported=True,
            answer=vlm_result.answer,
            model_used="EarthDial 4B RGB",
            annotated_image=annotated,
            elapsed_route_ms=elapsed_route,
            elapsed_vlm_s=vlm_result.elapsed_s,
            elapsed_total_s=round(time.time() - t_total, 1),
        )
        self.history.add(result)
        return result
