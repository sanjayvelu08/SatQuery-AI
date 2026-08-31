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
    elapsed_route_ms: float = 0.0
    elapsed_vlm_s: float = 0.0
    elapsed_total_s: float = 0.0

    def to_dict(self) -> dict:
        return {
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

    def run(self, image_path: str, query: str) -> PipelineResult:
        """Execute the full pipeline for one (image, query) pair."""
        t_total = time.time()

        # ── Step 1: Route ─────────────────────────────────────────
        t0 = time.time()
        route: RouteResult = classify(query)
        elapsed_route = (time.time() - t0) * 1000  # ms

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
