"""
SatQuery AI — Gradio Web UI (v3, Loop 6).

Changes from v2:
  - Annotated image output (SAR bounding boxes, optical grounding)
  - Result panel shows: analysis type, model used, answer, visual evidence
  - Robust edge-case handling (8 scenarios)
  - Improved demo scenarios with annotated images
  - Clearer layout with separate image and annotated image outputs

Run:  python -X utf8 -m satquery.app
URL:  http://localhost:7860
"""

from __future__ import annotations

import os
import sys

import gradio as gr

# Ensure satquery is importable
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from satquery.pipeline import SatQueryPipeline, PipelineResult
from satquery.demos import get_demo_list, get_demo_by_name
from satquery.visualize import create_annotated_image

# ── Global state ──────────────────────────────────────────────────
pipeline: SatQueryPipeline | None = None


def get_pipeline() -> SatQueryPipeline:
    global pipeline
    if pipeline is None:
        pipeline = SatQueryPipeline(max_history=5)
    return pipeline


# ── Intent badge mapping ──────────────────────────────────────────

_INTENT_BADGES = {
    "caption": "📝 Captioning",
    "vqa": "❓ Visual QA",
    "detect": "🎯 Detection",
    "grounding": "📍 Grounding",
    "classification": "🏷️ Classification",
    "change": "🔄 Change Detection",
    "sar": "📡 SAR Analysis",
    "general": "💬 General",
}

_MODEL_LABELS = {
    "caption": "EarthDial 4B RGB (VLM)",
    "vqa": "EarthDial 4B RGB (VLM)",
    "detect": "EarthDial 4B RGB (VLM + Grounding)",
    "grounding": "EarthDial 4B RGB (VLM + Grounding)",
    "classification": "EarthDial 4B RGB (VLM)",
    "change": "— (not implemented)",
    "sar": "YOLOv8 SAR Vessel Detector",
    "general": "EarthDial 4B RGB (VLM)",
}


def _format_answer(result: PipelineResult) -> str:
    """Format pipeline result as rich Markdown with model info."""
    badge = _INTENT_BADGES.get(result.intent, result.intent)
    model = result.model_used or _MODEL_LABELS.get(result.intent, "Unknown")

    if not result.supported:
        return (
            f"### {badge}\n\n"
            f"⚠️ **This feature is not yet available**\n\n"
            f"{result.unsupported_reason}\n\n"
            f"---\n"
            f"*Try a different query type, such as describing the image "
            f"or asking a question about what you see.*"
        )

    # Header with model info
    header = f"### {badge}\n*Model: {model}*\n\n"

    # Count bounding boxes in answer
    bbox_count = 0
    if result.answer:
        import re
        bbox_count = len(re.findall(r'\[\[\d', result.answer))

    bbox_note = ""
    if bbox_count > 0:
        bbox_note = f"\n\n---\n*📍 {bbox_count} region(s) with bounding coordinates — see annotated image →*"

    # SAR-specific extras
    sar_note = ""
    if result.sar_result and result.sar_result.success:
        sr = result.sar_result
        if sr.num_detections > 0:
            sar_note = (
                f"\n\n---\n**Detection Summary:** {sr.num_detections} target(s) "
                f"in {sr.inference_time_ms:.0f}ms "
                f"({sr.gpu_vram_mb:.0f} MB VRAM)"
            )

    return f"{header}{result.answer}{bbox_note}{sar_note}"


def _format_timing(result: PipelineResult) -> str:
    """Format timing info."""
    if result.elapsed_vlm_s == 0 and result.elapsed_total_s == 0:
        return "⚡ Instant (pre-computed demo)"
    parts = []
    if result.elapsed_route_ms > 0:
        parts.append(f"Route: {result.elapsed_route_ms:.0f}ms")
    if result.elapsed_vlm_s > 0:
        parts.append(f"VLM: {result.elapsed_vlm_s:.1f}s")
    if result.elapsed_total_s > 0:
        parts.append(f"Total: {result.elapsed_total_s:.1f}s")
    return "⏱️ " + " | ".join(parts) if parts else ""


def _format_status(result: PipelineResult) -> str:
    """Format status line."""
    if not result.supported:
        return f"⚠️ {result.intent} — feature coming soon"
    model = result.model_used or "unknown"
    return f"✅ Analyzed ({result.intent}) via {model}"


# ── Edge-case validation ──────────────────────────────────────────

def _validate_inputs(image, query: str) -> tuple[bool, PipelineResult | None]:
    """
    Validate inputs and return error result if invalid.
    Returns (is_valid, error_result_or_None).
    """
    # No image
    if image is None:
        return False, PipelineResult(
            query=query or "", image_path="", intent="general",
            all_intents=[], supported=False,
            unsupported_reason=(
                "**No image provided.**\n\n"
                "Please upload a satellite image (Sentinel-2, Landsat, SAR, etc.) "
                "or select a demo scenario from the dropdown."
            ),
        )

    # Empty query
    if not query or not query.strip():
        return False, PipelineResult(
            query="", image_path=image, intent="general",
            all_intents=[], supported=False,
            unsupported_reason=(
                "**No query entered.**\n\n"
                "Please type a question about the image. Examples:\n"
                "- *\"Describe this satellite image\"*\n"
                "- *\"Are there buildings here?\"*\n"
                "- *\"Detect ships in this SAR image\"*"
            ),
        )

    # Corrupt / unreadable image
    try:
        from PIL import Image
        img = Image.open(image)
        img.verify()
    except Exception:
        return False, PipelineResult(
            query=query, image_path=str(image), intent="general",
            all_intents=[], supported=False,
            unsupported_reason=(
                "**Could not read the image file.**\n\n"
                "The file may be corrupt or in an unsupported format. "
                "Please upload a valid JPEG or PNG image."
            ),
        )

    return True, None


# ── Core analyze function ─────────────────────────────────────────

def analyze(image, query: str, use_demo: str, history_state: list) -> tuple:
    """
    Run the SatQuery pipeline on an image + query.

    Returns:
        (answer_md, timing, status, image_preview, annotated_preview,
         history_md, history_state)
    """
    # ── Demo mode ─────────────────────────────────────────────────
    if use_demo and use_demo != "None":
        demo = get_demo_by_name(use_demo)
        if demo:
            result = PipelineResult(
                query=demo["query"],
                image_path=demo["image"],
                intent=demo["intent"],
                all_intents=demo["all_intents"],
                supported=demo["supported"],
                answer=demo["answer"],
                model_used=demo.get("model_used", ""),
                elapsed_vlm_s=0.0,
                elapsed_total_s=0.0,
            )
            history_entry = f"**Demo:** {use_demo}\n*Intent:* {result.intent}"
            new_history = (history_state or []) + [history_entry]
            new_history = new_history[-5:]

            # Try to create annotated image for demo
            annotated = demo.get("annotated_image")
            if annotated is None and demo["supported"] and demo["answer"]:
                try:
                    annotated = create_annotated_image(
                        demo["image"], demo["answer"], demo["intent"]
                    )
                except Exception:
                    pass

            return (
                _format_answer(result),
                _format_timing(result),
                f"✅ Demo loaded: {result.intent}",
                demo["image"],
                annotated,
                _format_history(new_history),
                new_history,
            )

    # ── Input validation ──────────────────────────────────────────
    is_valid, error_result = _validate_inputs(image, query)
    if not is_valid:
        return (
            _format_answer(error_result),
            "",
            "⚠️ Validation error",
            image,
            None,
            _format_history(history_state or []),
            history_state or [],
        )

    # ── Live mode ─────────────────────────────────────────────────
    try:
        result = get_pipeline().run(image, query)
    except Exception as e:
        err = PipelineResult(
            query=query, image_path=image, intent="error",
            all_intents=[], supported=False,
            unsupported_reason=f"Pipeline error: {e}",
        )
        return (
            f"### ❌ Error\n\n{e}", "", "❌ Error",
            image, None,
            _format_history(history_state or []), history_state or [],
        )

    # Build history
    badge = _INTENT_BADGES.get(result.intent, result.intent)
    history_entry = f"{badge}\n> {query[:80]}"
    new_history = (history_state or []) + [history_entry]
    new_history = new_history[-5:]

    return (
        _format_answer(result),
        _format_timing(result),
        _format_status(result),
        image,
        result.annotated_image,
        _format_history(new_history),
        new_history,
    )


def _format_history(history: list[str]) -> str:
    """Format history entries as Markdown."""
    if not history:
        return "*No queries yet.*"
    entries = list(reversed(history))
    return "\n\n---\n\n".join(entries)


# ── Demo loader ───────────────────────────────────────────────────

def load_demo(demo_name: str):
    """Load a demo scenario into the image and query fields."""
    if not demo_name or demo_name == "None":
        return None, ""

    demo = get_demo_by_name(demo_name)
    if demo:
        return demo["image"], demo["query"]
    return None, ""


# ── Build Gradio UI ───────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    demo_names = ["None"] + get_demo_list()

    with gr.Blocks(
        title="SatQuery AI — Remote Sensing Assistant",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🛰️ SatQuery AI\n"
            "**Interactive Vision-Language Assistant for Remote Sensing Image Analysis**\n"
            "*ISRO Problem Statement SIH26167 — Smart India Hackathon 2026*\n"
        )

        with gr.Row():
            # ── Left column: inputs ───────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Quick Demo")
                demo_dropdown = gr.Dropdown(
                    choices=demo_names,
                    value="None",
                    label="Select a pre-computed scenario",
                    info="Instant results — no VLM loading required",
                )

                gr.Markdown("---")

                image_input = gr.Image(
                    type="filepath",
                    label="🛰️ Upload satellite image (Sentinel-2, Landsat, SAR)",
                    height=280,
                )

                query_input = gr.Textbox(
                    label="💬 Ask a question",
                    placeholder=(
                        "e.g., Describe this image, Are there buildings?, "
                        "Detect ships in this SAR image"
                    ),
                    lines=2,
                )

                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze Image",
                        variant="primary",
                        size="lg",
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="sm",
                    )

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

            # ── Right column: results ─────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Result")
                answer_output = gr.Markdown(
                    value=(
                        "*Upload a satellite image and ask a question, "
                        "or select a demo scenario from the left panel.*\n\n"
                        "**Supported query types:**\n"
                        "- 📝 *\"Describe this satellite image\"* — Captioning\n"
                        "- ❓ *\"Are there buildings here?\"* — Visual QA\n"
                        "- 🎯 *\"Find all structures\"* — Detection\n"
                        "- 🏷️ *\"Classify the land cover\"* — Classification\n"
                        "- 📍 *\"Locate key features\"* — Grounding\n"
                        "- 📡 *\"Detect ships in this SAR image\"* — SAR Vessel Detection"
                    ),
                )
                timing_output = gr.Textbox(
                    label="Timing",
                    interactive=False,
                )

                # Annotated image output
                gr.Markdown("### 🖼️ Visual Evidence")
                annotated_output = gr.Image(
                    label="Annotated image (bounding boxes)",
                    height=280,
                    interactive=False,
                )

        # ── History section ───────────────────────────────────────
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📜 Query History (Last 5)")
                history_output = gr.Markdown(
                    value="*No queries yet.*",
                )
            with gr.Column(scale=1):
                gr.Markdown("### ℹ️ About")
                gr.Markdown(
                    "**SatQuery AI** uses:\n"
                    "- **EarthDial 4B** (InternVL + Phi-3) for optical RS analysis\n"
                    "- **YOLOv8** for SAR vessel detection\n\n"
                    "**Hardware:** NVIDIA RTX 3050 (4 GB VRAM)\n"
                    "**Imagery:** Optical (Sentinel-2, Landsat) + SAR\n\n"
                    "**Architecture:**\n"
                    "Query → Intent Router → VLM/SAR Tool → Structured Result\n"
                    "with annotated visual evidence"
                )

        # ── State ─────────────────────────────────────────────────
        history_state = gr.State([])

        # ── Wire events ───────────────────────────────────────────
        demo_dropdown.change(
            fn=load_demo,
            inputs=[demo_dropdown],
            outputs=[image_input, query_input],
        )

        analyze_args = dict(
            fn=analyze,
            inputs=[image_input, query_input, demo_dropdown, history_state],
            outputs=[answer_output, timing_output, status_output,
                     image_input, annotated_output, history_output, history_state],
        )

        analyze_btn.click(**analyze_args)
        query_input.submit(**analyze_args)

        def clear_all():
            return (
                None,           # image
                "",             # query
                "None",         # demo
                "*No queries yet.*",  # history
                [],             # history state
                "*Ready for analysis.*",  # answer
                "",             # timing
                "",             # status
                None,           # annotated image
            )

        clear_btn.click(
            fn=clear_all,
            outputs=[image_input, query_input, demo_dropdown,
                     history_output, history_state,
                     answer_output, timing_output, status_output,
                     annotated_output],
        )

    return app


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
