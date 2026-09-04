"""
SatQuery AI — EarthDial VLM wrapper.

Uses a subprocess bridge to invoke EarthDial in an isolated venv
(transformers==4.37.2) rather than loading directly in the main environment
(transformers==4.42.4). This avoids the Phi3 position_ids compatibility issue.

Model flavour
-------------
By default the worker loads the VRSBench QLoRA-adapted EarthDial
(domain-adapted RS-VQA, validated 0 → 49% exact-match on 500 held-out
VRSBench questions) and falls back to the clean pretrained base model if the
adapter artifact is unavailable. To force the base model, construct with
SatQueryVLM(adapter_path="") or set SATQUERY_EARTHDIAL_ADAPTER="" for the
worker process.

Public interface is preserved: SatQueryVLM.query() → InferenceResult.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .vlm_bridge import run_earthdial, EarthDialResult


@dataclass
class InferenceResult:
    """Structured output from a single VLM query."""
    answer: str
    query: str
    prompt_sent: str
    elapsed_s: float
    image_path: str
    model_loaded: bool


class SatQueryVLM:
    """EarthDial 4B wrapper via subprocess bridge.

    Args:
        model_dir: kept for API compatibility; not used in bridge mode.
        adapter_path: LoRA adapter dir passed to the EarthDial worker.
            None (default) → worker auto-resolves: env
            SATQUERY_EARTHDIAL_ADAPTER, else the standard VRSBench-adapted
            checkpoint if present, else the pretrained base model.
            "" (empty) → force the pretrained base model.
    """

    def __init__(self, model_dir: str | None = None,
                 adapter_path: str | None = None):
        # model_dir kept for API compatibility; not used in bridge mode
        self.model_dir = model_dir
        self.adapter_path = adapter_path

    # ── Loading ───────────────────────────────────────────────────

    def load(self) -> None:
        """No-op in bridge mode — EarthDial loads in subprocess."""
        pass

    def unload(self) -> None:
        """No-op in bridge mode — subprocess cleans up after itself."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Returns True if the EarthDial venv is available."""
        from .vlm_bridge import _find_earthdial_venv_python
        return _find_earthdial_venv_python() is not None

    # ── Inference ─────────────────────────────────────────────────

    def query(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 50,
        num_beams: int = 2,
    ) -> InferenceResult:
        """Run a single VLM query via isolated subprocess.

        max_tokens/num_beams default to 50/2 for backwards compatibility;
        richer interpretation tasks (e.g. change interpretation) may request
        a higher token budget explicitly.
        """
        t0 = time.time()

        result: EarthDialResult = run_earthdial(
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            num_beams=num_beams,
            timeout=300,
            adapter=self.adapter_path,
        )

        elapsed = round(time.time() - t0, 1)

        if result.success:
            return InferenceResult(
                answer=result.answer,
                query=prompt,
                prompt_sent=prompt,
                elapsed_s=elapsed,
                image_path=image_path,
                model_loaded=True,
            )
        else:
            return InferenceResult(
                answer=f"**EarthDial Error:** {result.error}",
                query=prompt,
                prompt_sent=prompt,
                elapsed_s=elapsed,
                image_path=image_path,
                model_loaded=False,
            )

    # ── VRAM info ─────────────────────────────────────────────────

    @staticmethod
    def vram_info() -> dict[str, float]:
        """VRAM info for main process (EarthDial runs in subprocess)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}
            used = torch.cuda.memory_allocated(0) / 1e9
            free, total = torch.cuda.mem_get_info(0)
            return {
                "used_GB": round(used, 2),
                "free_GB": round(free / 1e9, 2),
                "total_GB": round(total / 1e9, 2),
            }
        except ImportError:
            return {"error": "torch not available"}
