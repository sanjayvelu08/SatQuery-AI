"""
SatQuery AI — EarthDial Subprocess Bridge.

Invokes EarthDial inference in the isolated earthdial_test_venv via subprocess,
similar to how sar_tool.py invokes SAR detection in sar_venv.

This design:
- Keeps transformers==4.37.2 isolated from the main environment (4.42.4)
- Prevents VRAM conflicts (EarthDial needs ~2.85 GB)
- Only loads EarthDial when actually needed
- Returns structured JSON results to the main pipeline
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class EarthDialResult:
    """Structured result from EarthDial subprocess inference."""
    success: bool
    answer: str = ""
    load_ms: float = 0
    inference_ms: float = 0
    total_ms: float = 0
    vram_used_mb: float = 0
    error: Optional[str] = None
    # Which LoRA adapter was actually used (None = pretrained base model).
    adapter_used: Optional[str] = None
    # Base precision used in the worker: "4bit" (default) or "bf16" fallback.
    precision: Optional[str] = None


def _find_earthdial_venv_python() -> Optional[str]:
    """Find the Python executable in the isolated earthdial_test_venv."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check multiple possible venv locations
    candidates = [
        os.path.join(project_root, "changemodel_test", "earthdial_test_venv", "Scripts", "python.exe"),
        os.path.join(project_root, "changemodel_test", "earthdial_test_venv", "bin", "python"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def run_earthdial(
    image_path: str,
    prompt: str,
    max_tokens: int = 200,
    num_beams: int = 5,
    timeout: int = 300,
    adapter: Optional[str] = None,
) -> EarthDialResult:
    """Run EarthDial inference via isolated subprocess. Never breaks the main env.

    adapter: explicit LoRA adapter dir to pass to the worker. None leaves the
    decision to the worker (env SATQUERY_EARTHDIAL_ADAPTER, else the default
    VRSBench-adapted checkpoint, else the pretrained base model). An empty
    string forces the pretrained base model.
    """
    venv_python = _find_earthdial_venv_python()

    if venv_python is None:
        return EarthDialResult(
            success=False,
            error="earthdial_test_venv not found. Run the venv setup first.",
        )

    if not os.path.exists(image_path):
        return EarthDialResult(success=False, error=f"Image not found: {image_path}")

    image_path = os.path.abspath(image_path)

    # Build command
    cmd = [
        venv_python, "-X", "utf8",
        "-m", "satquery.earthdial_infer",
        image_path,
        prompt,
        "--max_tokens", str(max_tokens),
        "--num_beams", str(num_beams),
    ]

    # Explicit adapter override (None = let the worker auto-resolve).
    if adapter is not None:
        cmd += ["--adapter", adapter]

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
            error_line = [l for l in stderr.split("\n") if "Error" in l or "error" in l or "Traceback" in l]
            msg = error_line[-1] if error_line else stderr[:300]
            return EarthDialResult(success=False, error=f"EarthDial subprocess failed: {msg}")

        # Parse JSON output
        stdout = result.stdout.strip()
        lines = stdout.split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        if json_start is None:
            return EarthDialResult(success=False, error="No JSON in EarthDial subprocess output")

        json_str = "\n".join(lines[json_start:])
        data = json.loads(json_str)

        if not data.get("success"):
            return EarthDialResult(success=False, error=data.get("error", "Unknown EarthDial error"))

        return EarthDialResult(
            success=True,
            answer=data.get("answer", ""),
            load_ms=data.get("load_ms", 0),
            inference_ms=data.get("inference_ms", 0),
            total_ms=data.get("total_ms", 0),
            vram_used_mb=data.get("vram_used_mb", 0),
            adapter_used=data.get("adapter_used"),
            precision=data.get("precision"),
        )

    except subprocess.TimeoutExpired:
        return EarthDialResult(success=False, error=f"EarthDial timed out (>{timeout}s)")
    except json.JSONDecodeError as e:
        return EarthDialResult(success=False, error=f"Failed to parse EarthDial output: {e}")
    except Exception as e:
        return EarthDialResult(success=False, error=f"Unexpected error: {e}")
