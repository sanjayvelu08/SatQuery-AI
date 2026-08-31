"""
SAR Tool — bridges the main SatQuery pipeline to the isolated SAR YOLO model.

This module runs in the MAIN environment (with EarthDial).
It invokes the SAR inference script in the isolated sar_venv via subprocess.

This design:
- Keeps mmdet/ultralytics dependencies isolated from EarthDial
- Prevents VRAM conflicts (EarthDial needs 2.85 GB, SAR needs ~21 MB)
- Only loads SAR model when actually needed
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SARDetection:
    class_name: str
    confidence: float
    bbox_xyxy: List[float]


@dataclass
class SARResult:
    success: bool
    detections: List[SARDetection] = field(default_factory=list)
    num_detections: int = 0
    inference_time_ms: float = 0
    gpu_vram_mb: float = 0
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        if not self.success:
            return f"SAR analysis failed: {self.error}"
        if self.num_detections == 0:
            return "No maritime targets detected in this SAR image."
        lines = [f"Detected {self.num_detections} maritime target(s):"]
        for i, d in enumerate(self.detections, 1):
            lines.append(
                f"  {i}. {d.class_name.title()} "
                f"(confidence: {d.confidence:.1%})"
            )
        return "\n".join(lines)


def _find_sar_venv_python() -> Optional[str]:
    """Find the Python executable in the isolated sar_venv."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Windows
    win_path = os.path.join(project_root, "sar_venv", "Scripts", "python.exe")
    if os.path.exists(win_path):
        return win_path

    # Unix
    unix_path = os.path.join(project_root, "sar_venv", "bin", "python")
    if os.path.exists(unix_path):
        return unix_path

    return None


def run_sar_detection(
    image_path: str,
    conf: float = 0.25,
    timeout: int = 30,
) -> SARResult:
    """Run SAR detection via isolated subprocess. Never breaks the main env."""
    venv_python = _find_sar_venv_python()

    if venv_python is None:
        return SARResult(
            success=False,
            error="sar_venv not found. Run 'python -m venv sar_venv' in project root first."
        )

    if not os.path.exists(image_path):
        return SARResult(success=False, error=f"Image not found: {image_path}")

    # Resolve absolute path for image
    image_path = os.path.abspath(image_path)

    # Build command
    cmd = [
        venv_python, "-X", "utf8",
        "-m", "satquery.sar_infer",
        image_path,
        "--conf", str(conf),
    ]

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Extract just the error, not the full traceback
            error_line = [l for l in stderr.split("\n") if "Error" in l or "error" in l]
            msg = error_line[-1] if error_line else stderr[:200]
            return SARResult(success=False, error=f"SAR subprocess failed: {msg}")

        # Parse JSON output (first line that is valid JSON)
        stdout = result.stdout.strip()
        # Find JSON block (may have ultralytics log lines before it)
        lines = stdout.split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        if json_start is None:
            return SARResult(success=False, error="No JSON in SAR subprocess output")

        json_str = "\n".join(lines[json_start:])
        data = json.loads(json_str)

        if not data.get("success"):
            return SARResult(success=False, error=data.get("error", "Unknown SAR error"))

        detections = []
        for d in data.get("detections", []):
            detections.append(SARDetection(
                class_name=d["class"],
                confidence=d["confidence"],
                bbox_xyxy=d["bbox_xyxy"],
            ))

        return SARResult(
            success=True,
            detections=detections,
            num_detections=data.get("num_detections", len(detections)),
            inference_time_ms=data.get("inference_time_ms", 0),
            gpu_vram_mb=data.get("gpu_vram_mb", 0),
        )

    except subprocess.TimeoutExpired:
        return SARResult(success=False, error="SAR detection timed out (>30s)")
    except json.JSONDecodeError as e:
        return SARResult(success=False, error=f"Failed to parse SAR output: {e}")
    except Exception as e:
        return SARResult(success=False, error=f"Unexpected error: {e}")


def format_sar_response(result: SARResult) -> str:
    """Format SAR result as markdown suitable for Gradio display."""
    if not result.success:
        return f"**SAR Analysis Error:** {result.error}"

    if result.num_detections == 0:
        return (
            "**SAR Maritime Analysis:**\n\n"
            "No ships or maritime vessels detected with sufficient confidence in this SAR image.\n\n"
            "_Note: This detector identifies ships and maritime targets only. "
            "It does not analyze terrain, vegetation, or urban features in SAR imagery._"
        )

    lines = [
        "**SAR Maritime Analysis**",
        "",
        f"Detected **{result.num_detections}** maritime target(s) in SAR imagery:\n",
        "| # | Object | Confidence | Bounding Box |",
        "|---|--------|-----------|-------------|",
    ]

    for i, d in enumerate(result.detections, 1):
        bb = d.bbox_xyxy
        lines.append(
            f"| {i} | {d.class_name.title()} | {d.confidence:.1%} | "
            f"[{bb[0]:.0f}, {bb[1]:.0f}, {bb[2]:.0f}, {bb[3]:.0f}] |"
        )

    lines.extend([
        "",
        f"_Inference: {result.inference_time_ms:.0f}ms, "
        f"GPU VRAM: {result.gpu_vram_mb:.0f} MB_",
        "",
        "_Note: This detector uses a YOLOv8 model trained on SAR vessel detection data. "
        "It provides object-level detection only, not natural-language scene understanding "
        "of SAR imagery._",
    ])

    return "\n".join(lines)
