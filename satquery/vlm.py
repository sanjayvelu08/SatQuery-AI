"""
SatQuery AI — EarthDial VLM wrapper.

Handles model loading, image preprocessing, and inference.
Lazy-loads the model on first query; provides unload() for VRAM management.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import torch
from PIL import Image

# EarthDial lives outside the satquery package; add to path once.
_EARTHDIAL_SRC = os.path.join(os.path.dirname(__file__), "..", "EarthDial", "src")
_EARTHDIAL_SRC = os.path.normpath(_EARTHDIAL_SRC)
if _EARTHDIAL_SRC not in sys.path:
    sys.path.insert(0, _EARTHDIAL_SRC)

_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "EarthDial_4B_RGB"
)

# Suppress flash-attn warnings at import time
import warnings
warnings.filterwarnings("ignore", message=".*flash.*", category=UserWarning)


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
    """EarthDial 4B wrapper with lazy loading."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = os.path.normpath(model_dir or _DEFAULT_MODEL_DIR)
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._image_size = None

    # ── Loading ───────────────────────────────────────────────────

    def load(self) -> None:
        """Load EarthDial model into GPU+CPU (takes ~3-5 s)."""
        if self._model is not None:
            return  # already loaded

        from earthdial.model.internvl_chat import InternVLChatModel
        from earthdial.train.dataset import build_transform
        from transformers import AutoTokenizer

        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True, use_fast=False
        )
        self._model = InternVLChatModel.from_pretrained(
            self.model_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        self._image_size = (
            self._model.config.force_image_size
            or self._model.config.vision_config.image_size
        )
        self._transform = build_transform(
            is_train=False, input_size=self._image_size, normalize_type="imagenet"
        )

        vram_gb = round(torch.cuda.memory_allocated(0) / 1e9, 2)
        print(f"[VLM] Loaded EarthDial in {time.time()-t0:.1f}s — VRAM: {vram_gb} GB")

    def unload(self) -> None:
        """Free all GPU memory."""
        del self._model
        self._model = None
        torch.cuda.empty_cache()
        print(f"[VLM] Unloaded — VRAM free: {round(torch.cuda.mem_get_info(0)[0]/1e9,2)} GB")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Inference ─────────────────────────────────────────────────

    def query(self, image_path: str, prompt: str) -> InferenceResult:
        """Run a single VLM query. Loads model lazily if needed."""
        self.load()

        image = Image.open(image_path).convert("RGB")
        pixel_values = self._transform(image).unsqueeze(0).cuda().to(torch.bfloat16)

        gen_cfg = {
            "num_beams": 5,
            "max_new_tokens": 200,
            "min_new_tokens": 1,
            "do_sample": False,
        }

        t0 = time.time()
        answer = self._model.chat(
            self._tokenizer, pixel_values, prompt, gen_cfg, verbose=False
        )
        elapsed = round(time.time() - t0, 1)

        return InferenceResult(
            answer=answer,
            query=prompt,
            prompt_sent=prompt,
            elapsed_s=elapsed,
            image_path=image_path,
            model_loaded=True,
        )

    # ── VRAM info ─────────────────────────────────────────────────

    @staticmethod
    def vram_info() -> dict[str, float]:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        used = torch.cuda.memory_allocated(0) / 1e9
        free, total = torch.cuda.mem_get_info(0)
        return {
            "used_GB": round(used, 2),
            "free_GB": round(free / 1e9, 2),
            "total_GB": round(total / 1e9, 2),
        }
