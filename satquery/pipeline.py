"""
SatQuery AI — Core pipeline with agentic orchestration.

Every analysis path now produces:
  - input validation
  - specialist selection trace
  - execution trace
  - structured result
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .router import classify, RouteResult
from .vlm import SatQueryVLM, InferenceResult
from .sar_tool import run_sar_detection, format_sar_response, SARResult
from .sarclip_tool import (
    run_sarclip_scene,
    otsu_intensity_indicators,
    format_scene_scores,
    format_intensity_indicators,
)
from .visualize import create_annotated_image
from .evidence import ExecutionTraceStep

# Supported image extensions for validation
_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    model_used: str = ""
    annotated_image: str | None = None
    sar_result: SARResult | None = None
    change_result: object | None = None
    joint_result: object | None = None
    image_t2_path: str | None = None
    image_sar_path: str | None = None
    trace: list[ExecutionTraceStep] = field(default_factory=list)
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
        # Serialize trace
        d["trace"] = [
            {
                "step": t.step, "name": t.name, "tool": t.tool,
                "status": t.status, "duration_ms": t.duration_ms,
                "input_summary": t.input_summary,
                "output_summary": t.output_summary,
            }
            for t in self.trace
        ]
        return d

    def format(self) -> str:
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
        if self.trace:
            lines.append(f"Trace:   {len(self.trace)} steps")
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


# ── Specialist selection map ────────────────────────────────────

_SPECIALIST_MAP: dict[str, str] = {
    "caption": "EarthDial 4B RGB",
    "vqa": "EarthDial 4B RGB",
    "detect": "EarthDial 4B RGB",
    "grounding": "EarthDial 4B RGB",
    "classification": "EarthDial 4B RGB",
    "general": "EarthDial 4B RGB",
    "sar": "YOLOv8 SAR Vessel Detector",
    "change": "BIT-CD Change Detector",
    "joint_analysis": "YOLOv8 SAR + SAR-CLIP + EarthDial 4B + Evidence Fusion",
}


class SatQueryPipeline:
    """Orchestrates routing → specialist selection → execution → trace."""

    def __init__(self, vlm: SatQueryVLM | None = None, max_history: int = 5):
        self.vlm = vlm or SatQueryVLM()
        self.history = QueryHistory(max_size=max_history)

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _validate_image(path: str | None, label: str) -> list[str]:
        """Validate a single image path. Returns list of error strings."""
        errors = []
        if not path:
            errors.append(f"{label} image is required.")
            return errors
        if not os.path.exists(path):
            errors.append(f"{label} image not found.")
            return errors
        ext = os.path.splitext(path)[1].lower()
        if ext and ext not in _SUPPORTED_EXTS:
            errors.append(f"{label} has unsupported format '{ext}'.")
        try:
            from PIL import Image
            Image.open(path).verify()
        except Exception:
            errors.append(f"{label} image is corrupt or unreadable.")
        return errors

    @staticmethod
    def _make_step(
        step_num: int, name: str, tool: str, status: str,
        duration_ms: float, input_summary: str, output_summary: str,
        error: str | None = None,
    ) -> ExecutionTraceStep:
        return ExecutionTraceStep(
            step=step_num, name=name, tool=tool, status=status,
            duration_ms=round(duration_ms, 1),
            input_summary=input_summary, output_summary=output_summary,
            error=error,
        )

    # ── Main pipeline ───────────────────────────────────────────

    def run(
        self, image_path: str, query: str,
        image_t2_path: str | None = None,
        image_sar_path: str | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline for one (image, query) pair."""
        t_total = time.time()
        trace: list[ExecutionTraceStep] = []
        step_num = 0

        # ── Step 1: Route ─────────────────────────────────────
        t0 = time.time()
        route: RouteResult = classify(query)
        elapsed_route = (time.time() - t0) * 1000

        # Two temporal images → bi-temporal change request. The frontend
        # change mode always supplies image + image_t2, and semantic change
        # questions ("Has construction increased?") may not contain change
        # keywords. So when image_t2 is supplied, treat the request as change
        # UNLESS it is an explicit optical+SAR joint request (image + image_sar).
        # Joint and single-image behavior are never overridden.
        reroute_note = ""
        if image_t2_path and route.primary_intent not in ("change", "joint_analysis"):
            original_intent = route.primary_intent
            route = RouteResult(
                query=query, primary_intent="change",
                all_intents=route.all_intents + ["change"],
                prompt=None, supported=True,
                reason=f"Rerouted from '{original_intent}': image_t2 supplied.",
            )
            reroute_note = (
                f" rerouted_to=change from={original_intent} (image_t2 supplied)"
            )

        # Pair-aware routing (SIH 26167): when a SAR image is supplied, route
        # to the joint Optical+SAR workflow even if the query does not
        # literally contain the words "optical" and "SAR" — mirrors the
        # image_t2 -> change reroute above. Change requests (explicit change
        # intent or image_t2) keep priority.
        if (image_sar_path
                and route.primary_intent not in ("change", "joint_analysis")):
            original_intent = route.primary_intent
            route = RouteResult(
                query=query, primary_intent="joint_analysis",
                all_intents=route.all_intents + ["joint_analysis"],
                prompt=None, supported=True,
                reason=f"Rerouted from '{original_intent}': image_sar supplied.",
                detected_modalities=["optical", "sar"],
            )
            reroute_note += (
                f" rerouted_to=joint_analysis from={original_intent} "
                "(image_sar supplied)"
            )

        step_num += 1
        trace.append(self._make_step(
            step_num, "route", "keyword_router", "ok", elapsed_route,
            input_summary=f"query='{query[:60]}'",
            output_summary=(
                f"intent={route.primary_intent} supported={route.supported}{reroute_note}"
            ),
        ))

        # ── Step 2: Validate ──────────────────────────────────
        t_val = time.time()
        val_errors: list[str] = []

        if route.primary_intent == "joint_analysis":
            val_errors.extend(self._validate_image(image_path, "Optical"))
            val_errors.extend(self._validate_image(image_sar_path, "SAR"))
        elif route.primary_intent == "change":
            val_errors.extend(self._validate_image(image_path, "T1 (before)"))
            val_errors.extend(self._validate_image(image_t2_path, "T2 (after)"))
        else:
            val_errors.extend(self._validate_image(image_path, "Optical"))

        if not query or not query.strip():
            val_errors.append("No query provided.")

        val_ms = (time.time() - t_val) * 1000
        step_num += 1

        img_info = f"optical={'ok' if image_path else 'missing'}"
        if image_t2_path:
            img_info += f" t2={'ok' if image_t2_path else 'missing'}"
        if image_sar_path:
            img_info += f" sar={'ok' if image_sar_path else 'missing'}"

        trace.append(self._make_step(
            step_num, "validate", "input_validator",
            "ok" if not val_errors else "error",
            val_ms,
            input_summary=img_info,
            output_summary="; ".join(val_errors) if val_errors else "All inputs valid",
            error="; ".join(val_errors) if val_errors else None,
        ))

        if val_errors:
            result = PipelineResult(
                query=query, image_path=image_path, intent=route.primary_intent,
                all_intents=route.all_intents, supported=False,
                unsupported_reason="\n".join(val_errors),
                trace=trace, elapsed_route_ms=elapsed_route,
                elapsed_total_s=round(time.time() - t_total, 1),
            )
            self.history.add(result)
            return result

        # ── Step 3: Specialist selection ───────────────────────
        specialist = _SPECIALIST_MAP.get(route.primary_intent, "EarthDial 4B RGB")
        step_num += 1
        trace.append(self._make_step(
            step_num, "specialist_selection", "orchestrator", "ok", 0,
            input_summary=f"intent={route.primary_intent}",
            output_summary=f"selected: {specialist}",
        ))

        # ── Dispatch to specialist ────────────────────────────

        if route.primary_intent == "joint_analysis":
            return self._run_joint(
                image_path, image_sar_path, query, route, trace,
                step_num, t_total, elapsed_route,
            )

        if route.primary_intent == "change":
            return self._run_change(
                image_path, image_t2_path, query, route, trace,
                step_num, t_total, elapsed_route,
            )

        if route.primary_intent == "sar":
            return self._run_sar(
                image_path, query, route, trace,
                step_num, t_total, elapsed_route,
            )

        if not route.supported:
            result = PipelineResult(
                query=query, image_path=image_path, intent=route.primary_intent,
                all_intents=route.all_intents, supported=False,
                unsupported_reason=route.reason,
                trace=trace, elapsed_route_ms=elapsed_route,
                elapsed_total_s=round(time.time() - t_total, 1),
            )
            self.history.add(result)
            return result

        # ── Default: VLM optical analysis ─────────────────────
        return self._run_vlm(
            image_path, query, route, trace,
            step_num, t_total, elapsed_route,
        )

    # ── Specialist runners ──────────────────────────────────────

    def _run_vlm(
        self, image_path: str, query: str, route: RouteResult,
        trace: list[ExecutionTraceStep], step_num: int,
        t_total: float, elapsed_route: float,
    ) -> PipelineResult:
        """Run EarthDial VLM for caption/vqa/detect/grounding/classification."""
        step_num += 1
        t0 = time.time()
        vlm_result: InferenceResult = self.vlm.query(image_path, route.prompt)
        vlm_ms = (time.time() - t0) * 1000

        trace.append(self._make_step(
            step_num, "vlm_infer", "earthdial_4b",
            "ok" if vlm_result.model_loaded else "error",
            vlm_ms,
            input_summary=f"image={os.path.basename(image_path)} prompt_len={len(route.prompt or '')}",
            output_summary=f"{len(vlm_result.answer)} chars response",
        ))

        # Visual evidence
        annotated = None
        if route.primary_intent in ("detect", "grounding") and vlm_result.answer:
            step_num += 1
            t_vis = time.time()
            try:
                annotated = create_annotated_image(
                    image_path, vlm_result.answer, route.primary_intent
                )
            except Exception:
                pass
            vis_ms = (time.time() - t_vis) * 1000
            trace.append(self._make_step(
                step_num, "visual_evidence", "visualize",
                "ok" if annotated else "skipped",
                vis_ms,
                input_summary=f"intent={route.primary_intent}",
                output_summary="annotated image created" if annotated else "no detectable features",
            ))

        # Final answer step
        step_num += 1
        trace.append(self._make_step(
            step_num, "final_answer", "pipeline", "ok", 0,
            input_summary="vlm output",
            output_summary=f"{len(vlm_result.answer)} chars",
        ))

        result = PipelineResult(
            query=query, image_path=image_path, intent=route.primary_intent,
            all_intents=route.all_intents, supported=True,
            answer=vlm_result.answer, model_used="EarthDial 4B RGB",
            annotated_image=annotated, trace=trace,
            elapsed_route_ms=elapsed_route,
            elapsed_vlm_s=vlm_result.elapsed_s,
            elapsed_total_s=round(time.time() - t_total, 1),
        )
        self.history.add(result)
        return result

    def _run_change(
        self, image_path: str, image_t2_path: str | None,
        query: str, route: RouteResult,
        trace: list[ExecutionTraceStep], step_num: int,
        t_total: float, elapsed_route: float,
    ) -> PipelineResult:
        """Run BIT-CD change detection."""
        from .bit_tool import get_bit_tool

        # BIT-CD detect
        step_num += 1
        t0 = time.time()
        bit = get_bit_tool()
        output_dir = os.path.join(os.path.dirname(__file__), "..", "change_output")
        change_result = bit.detect(
            image_t1_path=image_path, image_t2_path=image_t2_path,
            output_dir=output_dir,
        )
        detect_ms = (time.time() - t0) * 1000

        trace.append(self._make_step(
            step_num, "bit_cd_detect", "bit_cd",
            "ok" if change_result.success else "error",
            detect_ms,
            input_summary=f"t1={os.path.basename(image_path)} t2={os.path.basename(image_t2_path)}",
            output_summary=(
                f"{change_result.change_pct:.1f}% change, {change_result.num_regions} regions"
                if change_result.success
                else (change_result.error or "failed")
            ),
        ))

        # Region extraction (part of BIT-CD detect, but record separately)
        if change_result.success:
            step_num += 1
            trace.append(self._make_step(
                step_num, "region_extraction", "bit_cd", "ok",
                change_result.postprocessing_ms,
                input_summary=f"{change_result.num_regions} raw components",
                output_summary=f"{change_result.num_regions} regions after filtering",
            ))

        # Visual evidence
        if change_result.success and change_result.overlay_path:
            step_num += 1
            trace.append(self._make_step(
                step_num, "visual_evidence", "bit_cd", "ok", 0,
                input_summary="change mask",
                output_summary="overlay + bbox images saved",
            ))

        # ── Change interpretation (one optional EarthDial call) ──
        from .change_interpret import run_change_interpretation, should_interpret

        interpret_text: str | None = None
        if change_result.success:
            step_num += 1
            if not change_result.change_detected:
                trace.append(self._make_step(
                    step_num, "change_interpret", "earthdial_change_interpreter",
                    "skipped", 0.0,
                    input_summary=f"query='{query[:60]}'",
                    output_summary="no change detected; nothing to interpret",
                ))
            else:
                wants_interp, skip_reason = should_interpret(query)
                if not wants_interp:
                    trace.append(self._make_step(
                        step_num, "change_interpret", "earthdial_change_interpreter",
                        "skipped", 0.0,
                        input_summary=f"query='{query[:60]}'",
                        output_summary=skip_reason or "interpretation not requested",
                    ))
                else:
                    t0 = time.time()
                    interp_error = ""
                    try:
                        interp_answer, _interp_s = run_change_interpretation(
                            vlm=self.vlm, t1_path=image_path, t2_path=image_t2_path,
                            change_result=change_result, query=query,
                        )
                    except Exception as exc:  # never fail the whole request
                        interp_answer, interp_error = None, f"{type(exc).__name__}: {exc}"
                    interp_ms = (time.time() - t0) * 1000
                    ok = bool(interp_answer and interp_answer.strip())
                    if ok:
                        interpret_text = interp_answer
                    trace.append(self._make_step(
                        step_num, "change_interpret", "earthdial_change_interpreter",
                        "ok" if ok else "failed", interp_ms,
                        input_summary=(
                            f"composite(T1|T2) + {change_result.num_regions} region(s), "
                            f"{change_result.change_pct:.1f}% change"
                        ),
                        output_summary=(
                            f"{len(interpret_text)} chars interpretation"
                            if ok else "interpretation failed; BIT-CD statistics preserved"
                        ),
                        error=None if ok else (interp_error or "EarthDial model unavailable"),
                    ))

        # Final answer: BIT-CD evidence + interpretation (when available)
        answer = (
            change_result.format_markdown()
            if change_result.success else f"Error: {change_result.error}"
        )
        if interpret_text:
            answer = (
                answer
                + "\n\n---\n\n"
                + "**🧠 What the changes appear to be (EarthDial)**\n\n"
                + interpret_text.strip()
                + "\n\n_Interpretation is qualitative visual description grounded in the "
                "BIT-CD change regions. It is not a semantic change classification, "
                "contains no object counts or physical-area estimates, and carries "
                "no geospatial coordinates._"
            )
        step_num += 1
        trace.append(self._make_step(
            step_num, "final_answer", "pipeline", "ok", 0,
            input_summary=(
                "bit_cd output + change interpretation" if interpret_text else "bit_cd output"
            ),
            output_summary=change_result.summary if change_result.success else "error",
        ))

        annotated = change_result.overlay_path if change_result.success else None

        result = PipelineResult(
            query=query, image_path=image_path, image_t2_path=image_t2_path,
            intent=route.primary_intent, all_intents=route.all_intents,
            supported=change_result.success,
            answer=answer,
            unsupported_reason="" if change_result.success else (change_result.error or "Change detection failed"),
            model_used="BIT-CD (LEVIR-CD pretrained)",
            annotated_image=annotated, change_result=change_result,
            trace=trace, elapsed_route_ms=elapsed_route,
            elapsed_total_s=round(time.time() - t_total, 1),
        )
        self.history.add(result)
        return result

    def _run_sar(
        self, image_path: str, query: str, route: RouteResult,
        trace: list[ExecutionTraceStep], step_num: int,
        t_total: float, elapsed_route: float,
    ) -> PipelineResult:
        """Run YOLOv8 SAR vessel detection."""
        step_num += 1
        t0 = time.time()
        sar_result = run_sar_detection(image_path)
        sar_ms = (time.time() - t0) * 1000

        trace.append(self._make_step(
            step_num, "sar_detect", "yolov8_sar",
            "ok" if sar_result.success else "error",
            sar_ms,
            input_summary=f"image={os.path.basename(image_path)}",
            output_summary=(
                f"{sar_result.num_detections} vessel(s) detected"
                if sar_result.success
                else (sar_result.error or "failed")
            ),
        ))

        answer_text = format_sar_response(sar_result)

        # Visual evidence
        annotated = None
        if sar_result.success and sar_result.num_detections > 0:
            step_num += 1
            t_vis = time.time()
            try:
                annotated = create_annotated_image(image_path, answer_text, "sar")
            except Exception:
                pass
            vis_ms = (time.time() - t_vis) * 1000
            trace.append(self._make_step(
                step_num, "visual_evidence", "visualize",
                "ok" if annotated else "skipped",
                vis_ms,
                input_summary=f"{sar_result.num_detections} detections",
                output_summary="annotated image created" if annotated else "no detections to visualize",
            ))

        # Final answer
        step_num += 1
        trace.append(self._make_step(
            step_num, "final_answer", "pipeline", "ok", 0,
            input_summary="sar output",
            output_summary=answer_text[:80],
        ))

        result = PipelineResult(
            query=query, image_path=image_path, intent=route.primary_intent,
            all_intents=route.all_intents, supported=sar_result.success,
            answer=answer_text,
            unsupported_reason="" if sar_result.success else (sar_result.error or "SAR tool failed"),
            model_used="YOLOv8 SAR Vessel Detector",
            annotated_image=annotated, sar_result=sar_result,
            trace=trace, elapsed_route_ms=elapsed_route,
            elapsed_total_s=round(time.time() - t_total, 1),
        )
        self.history.add(result)
        return result

    def _run_joint(
        self, image_path: str, image_sar_path: str | None,
        query: str, route: RouteResult,
        trace: list[ExecutionTraceStep], step_num: int,
        t_total: float, elapsed_route: float,
    ) -> PipelineResult:
        """Run optical + SAR joint analysis with evidence fusion."""
        from .evidence import (
            OpticalEvidence, SAREvidence, JointAnalysisResult,
        )
        from .fusion import (
            fuse_evidence, compute_confidence,
            build_optical_prompt_with_sar_context,
            build_optical_sar_composite,
            run_joint_interpretation,
        )

        t_joint = time.time()

        # ── SAR detection ─────────────────────────────────────
        step_num += 1
        t0 = time.time()
        sar_raw = run_sar_detection(image_sar_path)
        sar_ms = (time.time() - t0) * 1000

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

        trace.append(self._make_step(
            step_num, "sar_detect", "yolov8_sar",
            "ok" if sar_raw.success else "error",
            sar_ms,
            input_summary=f"sar={os.path.basename(image_sar_path)}",
            output_summary=f"{sar_raw.num_detections} vessel(s) detected" if sar_raw.success else (sar_raw.error or "failed"),
        ))

        # ── SAR-CLIP zero-shot scene labels (isolated subprocess) ─────────
        scene = run_sarclip_scene(image_sar_path)
        step_num += 1
        trace.append(self._make_step(
            step_num, "sarclip_scene", "alignearth_sar_clip",
            "ok" if scene.get("success") else "error",
            scene.get("total_ms", 0.0),
            input_summary=f"sar={os.path.basename(image_sar_path)}",
            output_summary=(
                format_scene_scores(scene["scores"]["coarse"])
                if scene.get("success") else (scene.get("error") or "failed")
            ),
        ))
        if scene.get("success"):
            sar_evidence.scene_scores = scene.get("scores")

        # ── Otsu intensity indicators (in-process, numpy-only) ────────────
        indicators = otsu_intensity_indicators(image_sar_path)
        step_num += 1
        trace.append(self._make_step(
            step_num, "otsu_indicators", "intensity_proxy", "ok", 0.0,
            input_summary=f"sar={os.path.basename(image_sar_path)}",
            output_summary=format_intensity_indicators(indicators)[:110],
        ))
        sar_evidence.intensity_indicators = indicators

        # ── Optical analysis ──────────────────────────────────
        step_num += 1
        t0 = time.time()
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
            sar_scene_scores=sar_evidence.scene_scores,
            sar_intensity_indicators=sar_evidence.intensity_indicators,
        )
        optical_result = self.vlm.query(image_path, optical_prompt)
        optical_ms = (time.time() - t0) * 1000

        optical_evidence = OpticalEvidence(
            source="earthdial_4b",
            answer=optical_result.answer,
            intent="joint_context",
            prompt_sent=optical_prompt,
            image_path=image_path,
            elapsed_s=optical_result.elapsed_s,
            success=optical_result.model_loaded,
        )

        trace.append(self._make_step(
            step_num, "optical_analyze", "earthdial_4b",
            "ok" if optical_evidence.success else "error",
            optical_ms,
            input_summary=f"optical={os.path.basename(image_path)}",
            output_summary=f"{len(optical_evidence.answer)} chars response",
        ))

        # ── Evidence fusion ────────────────────────────────────
        step_num += 1
        t0 = time.time()
        fused = fuse_evidence(optical_evidence, sar_evidence)
        fuse_ms = (time.time() - t0) * 1000

        trace.append(self._make_step(
            step_num, "fuse", "evidence_fusion", "ok", fuse_ms,
            input_summary=f"optical_ok={optical_evidence.success} sar_ok={sar_evidence.success}",
            output_summary=f"{len(fused.joint_capabilities)} capabilities, {len(fused.unresolved_gaps)} gaps",
        ))

        # ── OPTICAL | SAR composite for the joint interpretation call ─────
        step_num += 1
        t0 = time.time()
        composite_path = build_optical_sar_composite(
            optical_path=image_path, sar_path=image_sar_path)
        composite_ms = (time.time() - t0) * 1000
        trace.append(self._make_step(
            step_num, "pair_composite", "evidence_fusion", "ok", composite_ms,
            input_summary=(
                f"optical={os.path.basename(image_path)} "
                f"sar={os.path.basename(image_sar_path)}"
            ),
            output_summary="OPTICAL|SAR 256x256 side-by-side composite",
        ))

        # ── Joint interpretation (on the composite image) ──────
        step_num += 1
        t0 = time.time()
        try:
            joint_answer, interp_s = run_joint_interpretation(
                vlm=self.vlm, original_query=query,
                optical_evidence=optical_evidence, sar_evidence=sar_evidence,
                composite_path=composite_path,
            )
        finally:
            if os.path.isfile(composite_path):
                try:
                    os.remove(composite_path)
                except OSError:
                    pass
        interp_ms = (time.time() - t0) * 1000

        trace.append(self._make_step(
            step_num, "interpret", "earthdial_4b", "ok", interp_ms,
            input_summary="optical + sar evidence",
            output_summary=f"{len(joint_answer)} chars joint answer",
        ))

        # ── Confidence ─────────────────────────────────────────
        step_num += 1
        t0 = time.time()
        confidence, conf_reasoning = compute_confidence(fused, query)
        conf_ms = (time.time() - t0) * 1000

        trace.append(self._make_step(
            step_num, "confidence", "confidence_calc", "ok", conf_ms,
            input_summary="evidence quality factors",
            output_summary=f"reliability={confidence:.2f}",
        ))

        # ── Final answer ───────────────────────────────────────
        step_num += 1
        trace.append(self._make_step(
            step_num, "final_answer", "pipeline", "ok", 0,
            input_summary="joint output",
            output_summary=f"{len(joint_answer)} chars",
        ))

        joint_ms = (time.time() - t_joint) * 1000

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
            trace=trace,
            sar_annotated=sar_annotated,
            total_ms=round(joint_ms, 0),
            sar_ms=round(sar_ms, 0),
            optical_ms=round(optical_ms, 0),
            fusion_ms=round(fuse_ms, 0),
            joint_interpretation_ms=round(interp_ms, 0),
            models_used=(
                ["YOLOv8 SAR Vessel Detector",
                 "AlignEarth-SAR-ViT-B-16 (zero-shot scene)",
                 "EarthDial 4B RGB"]
                if scene.get("success") else
                ["YOLOv8 SAR Vessel Detector", "EarthDial 4B RGB"]
            ),
        )

        result = PipelineResult(
            query=query, image_path=image_path, image_sar_path=image_sar_path,
            intent=route.primary_intent, all_intents=route.all_intents,
            supported=True, answer=joint_result.format_markdown(),
            model_used="EarthDial 4B + YOLOv8 SAR (Joint)",
            annotated_image=sar_annotated or optical_evidence.image_path,
            sar_result=sar_raw, joint_result=joint_result,
            trace=trace, elapsed_route_ms=elapsed_route,
            elapsed_vlm_s=optical_result.elapsed_s + interp_s,
            elapsed_total_s=round(time.time() - t_total, 1),
        )
        self.history.add(result)
        return result
