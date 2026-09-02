"""
BIT-CD Change Detection Tool — SatQuery AI.

Provides bi-temporal change detection using the BIT (Bitemporal Image Transformer)
model pretrained on LEVIR-CD. Detects pixel-level changes between two satellite images.

Architecture: BASE_Transformer (ResNet-18 encoder + Transformer decoder)
Checkpoint:   best_ckpt.pt (LEVIR-CD pretrained, 57.3 MB)
Input:        Two RGB satellite images (T1/before, T2/after)
Output:       Binary change mask + changed regions + bounding boxes

Isolated validation result (LEVIR-CD, 7 samples):
  F1=0.9263, IoU=0.8640, Precision=0.9039, Recall=0.9515
  (This is a prototype validation result, NOT general accuracy.)

Reference: Chen & Shi, "Temporal Semantic Contrastive Learning for Remote Sensing
Change Detection", ICCV 2021.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*flash.*")
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ChangeRegion:
    """A single detected changed region."""
    region_id: int
    bbox: List[int]  # [x1, y1, x2, y2] in mask coordinates
    area_pixels: int
    area_pct: float
    width: int
    height: int


@dataclass
class ChangeDetectionResult:
    """Full output from BIT-CD change detection."""
    success: bool
    change_detected: bool = False
    change_pct: float = 0.0
    regions: List[ChangeRegion] = field(default_factory=list)
    num_regions: int = 0
    summary: str = ""
    mask_path: Optional[str] = None
    overlay_path: Optional[str] = None
    bbox_path: Optional[str] = None
    model_used: str = "BIT-CD (LEVIR-CD pretrained)"
    inference_time_ms: float = 0.0
    preprocessing_ms: float = 0.0
    postprocessing_ms: float = 0.0
    total_ms: float = 0.0
    vram_peak_mb: float = 0.0
    vram_used_mb: float = 0.0
    error: Optional[str] = None
    img_size: int = 256

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "change_detected": self.change_detected,
            "change_pct": self.change_pct,
            "num_regions": self.num_regions,
            "regions": [
                {"id": r.region_id, "bbox": r.bbox, "area_px": r.area_pixels,
                 "area_pct": r.area_pct, "w": r.width, "h": r.height}
                for r in self.regions
            ],
            "summary": self.summary,
            "model_used": self.model_used,
            "inference_time_ms": self.inference_time_ms,
            "preprocessing_ms": self.preprocessing_ms,
            "postprocessing_ms": self.postprocessing_ms,
            "total_ms": self.total_ms,
            "vram_peak_mb": self.vram_peak_mb,
            "vram_used_mb": self.vram_used_mb,
            "img_size": self.img_size,
        }

    def format_markdown(self) -> str:
        """Human-readable markdown output."""
        if not self.success:
            return f"**Change Detection Error:** {self.error}"

        lines = [
            "**🔄 Change Detection Analysis (BIT-CD)**",
            "",
        ]

        if not self.change_detected:
            lines.append("No significant changes detected between the two images.")
        else:
            lines.append(f"**Change detected:** {self.change_pct:.1f}% of image area")
            lines.append(f"**Changed regions:** {self.num_regions}")
            lines.append("")

            if self.regions:
                lines.append("| Region | Bounding Box | Area |")
                lines.append("|--------|-------------|------|")
                for r in self.regions:
                    bb = r.bbox
                    lines.append(
                        f"| R{r.region_id} | [{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}] "
                        f"| {r.area_pct:.1f}% |"
                    )
                lines.append("")

        lines.extend([
            f"---",
            f"_Model: {self.model_used} · "
            f"Inference: {self.inference_time_ms:.0f}ms · "
            f"VRAM: {self.vram_peak_mb:.0f} MB_",
            "",
            "_Note: BIT-CD detects spatial changes (construction, demolition, "
            "land-use change) between two images. It identifies WHERE changes "
        "occurred, not WHAT type of change. Validated on LEVIR-CD building "
        "change data (F1=0.9263 on 7-sample prototype test)._",
        ])

        return "\n".join(lines)


class BitCDTool:
    """BIT-CD change detection with lazy model loading."""

    def __init__(self, bit_cd_root: str | None = None):
        """
        Args:
            bit_cd_root: Path to the BIT_CD repository root.
                         Defaults to changemodel_test/BIT_CD relative to project root.
        """
        if bit_cd_root is None:
            project_root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            bit_cd_root = os.path.join(
                project_root, "changemodel_test", "BIT_CD"
            )
        self.bit_cd_root = os.path.normpath(bit_cd_root)
        self._model = None
        self._device = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    def load(self) -> None:
        """Load BIT-CD model. Lazy — only loads on first call."""
        if self._loaded:
            return

        # Add BIT_CD to Python path so model imports work
        if self.bit_cd_root not in sys.path:
            sys.path.insert(0, self.bit_cd_root)

        from argparse import Namespace
        import models.networks as nets

        t0 = time.time()

        args = Namespace(
            net_G="base_transformer_pos_s4_dd8_dedim8",
            gpu_ids="",
            n_class=2,
        )

        # Build model without GPU placement
        net_G = nets.define_G(args=args, init_type="normal", init_gain=0.02, gpu_ids=[])

        # Load checkpoint
        ckpt_path = os.path.join(self.bit_cd_root, "checkpoints", "BIT_LEVIR", "best_ckpt.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"BIT-CD checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        net_G.load_state_dict(checkpoint["model_G_state_dict"])

        # Place on CUDA if available, else CPU
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net_G = net_G.to(self._device).eval()
        self._model = net_G
        self._loaded = True

        elapsed = (time.time() - t0) * 1000
        params = sum(p.numel() for p in net_G.parameters())
        vram = torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        print(f"[BIT-CD] Loaded in {elapsed:.0f}ms — {params:,} params — VRAM: {vram:.1f}MB")

    def unload(self) -> None:
        """Free model and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[BIT-CD] Unloaded")

    def detect(
        self,
        image_t1_path: str,
        image_t2_path: str,
        output_dir: str | None = None,
        min_region_area: int = 50,
        img_size: int = 256,
    ) -> ChangeDetectionResult:
        """
        Run change detection on a pair of images.

        Args:
            image_t1_path: Path to earlier/before image (T1).
            image_t2_path: Path to later/after image (T2).
            output_dir: Directory for saving masks/overlays. If None, no files saved.
            min_region_area: Minimum connected-component area (pixels) to keep.
            img_size: Spatial resolution for BIT inference (256 default).

        Returns:
            ChangeDetectionResult with mask, regions, metrics, and timing.
        """
        t_total = time.time()

        # Validate inputs
        if not os.path.exists(image_t1_path):
            return ChangeDetectionResult(
                success=False, error=f"T1 image not found: {image_t1_path}"
            )
        if not os.path.exists(image_t2_path):
            return ChangeDetectionResult(
                success=False, error=f"T2 image not found: {image_t2_path}"
            )

        # Load model if needed
        try:
            self.load()
        except Exception as e:
            return ChangeDetectionResult(
                success=False, error=f"Failed to load BIT-CD model: {e}"
            )

        # ── Preprocessing ─────────────────────────────────────────
        t0 = time.time()
        try:
            pil_t1 = Image.open(image_t1_path).convert("RGB")
            pil_t2 = Image.open(image_t2_path).convert("RGB")

            # Preprocess: resize + normalize to [-1, 1]
            tensor_t1 = self._preprocess(pil_t1, img_size)
            tensor_t2 = self._preprocess(pil_t2, img_size)
            tensor_t1 = tensor_t1.to(self._device)
            tensor_t2 = tensor_t2.to(self._device)
        except Exception as e:
            return ChangeDetectionResult(
                success=False, error=f"Preprocessing failed: {e}"
            )
        preprocess_ms = (time.time() - t0) * 1000

        # ── Inference ─────────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        try:
            with torch.no_grad():
                logits = self._model(tensor_t1, tensor_t2)  # (1, 2, H, W)
        except Exception as e:
            return ChangeDetectionResult(
                success=False, error=f"BIT-CD inference failed: {e}"
            )
        inference_ms = (time.time() - t0) * 1000

        vram_peak = 0.0
        vram_used = 0.0
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated(0) / 1024**2
            vram_used = torch.cuda.memory_allocated(0) / 1024**2

        # ── Postprocessing ────────────────────────────────────────
        t0 = time.time()
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        probs = F.softmax(logits, dim=1)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        change_prob = probs[0, 1].cpu().numpy()
        postprocess_ms = (time.time() - t0) * 1000

        # ── Region extraction ─────────────────────────────────────
        regions, cleaned_mask = self._extract_regions(
            pred_mask, min_area=min_region_area, img_size=img_size
        )

        # ── Change statistics ─────────────────────────────────────
        change_pct = pred_mask.sum() / pred_mask.size * 100
        change_detected = len(regions) > 0 and change_pct > 0.5

        # ── Summary ───────────────────────────────────────────────
        if change_detected:
            summary = (
                f"Detected {len(regions)} spatially distinct changed region(s) "
                f"covering {change_pct:.1f}% of the image."
            )
        else:
            summary = "No significant changes detected between the two images."

        # ── Save outputs ──────────────────────────────────────────
        mask_path = None
        overlay_path = None
        bbox_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            basename = "change"

            # Save raw mask
            mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
            mask_path = os.path.join(output_dir, f"{basename}_mask.png")
            mask_img.save(mask_path)

            # Save overlay
            overlay_path = os.path.join(output_dir, f"{basename}_overlay.png")
            self._save_overlay(pil_t1, pred_mask, regions, overlay_path, img_size)

            # Save bounding box visualization
            bbox_path = os.path.join(output_dir, f"{basename}_bboxes.png")
            self._save_bboxes(pil_t1, regions, bbox_path, img_size)

        total_ms = (time.time() - t_total) * 1000

        return ChangeDetectionResult(
            success=True,
            change_detected=change_detected,
            change_pct=round(change_pct, 2),
            regions=regions,
            num_regions=len(regions),
            summary=summary,
            mask_path=mask_path,
            overlay_path=overlay_path,
            bbox_path=bbox_path,
            inference_time_ms=round(inference_ms, 1),
            preprocessing_ms=round(preprocess_ms, 1),
            postprocessing_ms=round(postprocess_ms, 1),
            total_ms=round(total_ms, 1),
            vram_peak_mb=round(vram_peak, 1),
            vram_used_mb=round(vram_used, 1),
            img_size=img_size,
        )

    # ── Private helpers ───────────────────────────────────────────

    def _preprocess(self, pil_img: Image.Image, img_size: int) -> torch.Tensor:
        """Resize and normalize image to [-1, 1] range for BIT-CD."""
        img_resized = pil_img.resize((img_size, img_size), Image.BICUBIC)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)
        tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        return tensor.unsqueeze(0)  # (1, 3, H, W)

    def _extract_regions(
        self, pred_mask: np.ndarray, min_area: int = 50, img_size: int = 256
    ) -> tuple[list[ChangeRegion], np.ndarray]:
        """Extract connected change regions with bounding boxes."""
        # Morphological cleanup
        struct = ndimage.generate_binary_structure(2, 1)
        cleaned = ndimage.binary_closing(pred_mask, structure=struct, iterations=2)
        cleaned = ndimage.binary_opening(cleaned, structure=struct, iterations=1)

        # Connected components
        labeled, num_features = ndimage.label(cleaned)

        regions: list[ChangeRegion] = []
        for i in range(1, num_features + 1):
            component = labeled == i
            area = int(component.sum())
            if area < min_area:
                continue

            ys, xs = np.where(component)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            pct = area / (img_size * img_size) * 100

            regions.append(ChangeRegion(
                region_id=0,  # renumbered below
                bbox=[x1, y1, x2, y2],
                area_pixels=area,
                area_pct=round(pct, 2),
                width=x2 - x1 + 1,
                height=y2 - y1 + 1,
            ))

        # Sort by area descending and renumber
        regions.sort(key=lambda r: r.area_pixels, reverse=True)
        for i, r in enumerate(regions):
            r.region_id = i + 1

        return regions, cleaned.astype(np.uint8)

    def _save_overlay(
        self, pil_img: Image.Image, pred_mask: np.ndarray,
        regions: list[ChangeRegion], path: str, img_size: int,
    ) -> None:
        """Save before image with red change overlay + bounding boxes."""
        display_size = min(pil_img.width, pil_img.height, 512)
        img = pil_img.resize((display_size, display_size), Image.BILINEAR)
        arr = np.array(img).copy()

        # Red tint where change detected
        mask_resized = np.array(
            Image.fromarray(pred_mask * 255, mode="L").resize(
                (display_size, display_size), Image.NEAREST
            )
        ) > 127

        arr[mask_resized, 0] = np.clip(
            arr[mask_resized, 0].astype(int) + 100, 0, 255
        ).astype(np.uint8)
        arr[mask_resized, 1] = (arr[mask_resized, 1] * 0.3).astype(np.uint8)
        arr[mask_resized, 2] = (arr[mask_resized, 2] * 0.3).astype(np.uint8)

        result = Image.fromarray(arr)
        draw = ImageDraw.Draw(result)

        # Draw bounding boxes
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        scale = display_size / img_size
        for r in regions:
            x1, y1, x2, y2 = [int(v * scale) for v in r.bbox]
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
            label = f"R{r.region_id}"
            label_y = max(0, y1 - 14) if y1 > 16 else y2 + 2
            draw.text((x1 + 2, label_y), label, fill="yellow", font=font)

        result.save(path, quality=95)

    def _save_bboxes(
        self, pil_img: Image.Image, regions: list[ChangeRegion],
        path: str, img_size: int,
    ) -> None:
        """Save before image with bounding boxes only (no overlay)."""
        display_size = min(pil_img.width, pil_img.height, 512)
        img = pil_img.resize((display_size, display_size), Image.BILINEAR).copy()
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        scale = display_size / img_size
        for r in regions:
            x1, y1, x2, y2 = [int(v * scale) for v in r.bbox]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label = f"R{r.region_id} ({r.area_pct:.1f}%)"
            label_y = max(0, y1 - 14) if y1 > 16 else y2 + 2
            draw.text((x1 + 2, label_y), label, fill="red", font=font)

        img.save(path, quality=95)


# ── Module-level singleton ────────────────────────────────────────
_tool: BitCDTool | None = None


def get_bit_tool() -> BitCDTool:
    """Get the singleton BIT-CD tool instance."""
    global _tool
    if _tool is None:
        _tool = BitCDTool()
    return _tool


def unload_bit_tool() -> None:
    """Unload the singleton BIT-CD tool to free memory."""
    global _tool
    if _tool is not None:
        _tool.unload()
        _tool = None
