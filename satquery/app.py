"""
SatQuery AI — Gradio Web UI (v4, Loop 8).

UI/UX Redesign:
  - Professional scientific/ISRO branding with custom CSS
  - Compact header with branding + demo selector
  - Image + query + analyze on left; results + visual evidence on right
  - Inline timing/status in result header (no separate textboxes)
  - Compact horizontal query history
  - Conditional annotated image (shown when available, empty state otherwise)

Backend is FROZEN — no changes to pipeline, router, VLM, SAR, or demo data.

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
from satquery.bit_tool import get_bit_tool, unload_bit_tool

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

_INTENT_COLORS = {
    "caption": "#2563EB",
    "vqa": "#7C3AED",
    "detect": "#DC2626",
    "grounding": "#EA580C",
    "classification": "#D97706",
    "change": "#6B7280",
    "sar": "#059669",
    "general": "#475569",
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
        bbox_note = f"\n\n---\n*📍 {bbox_count} region(s) with bounding coordinates — see Visual Evidence panel →*"

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
    """Format timing info as compact inline text."""
    if result.elapsed_vlm_s == 0 and result.elapsed_total_s == 0:
        return "⚡ Instant"
    parts = []
    if result.elapsed_route_ms > 0:
        parts.append(f"Route {result.elapsed_route_ms:.0f}ms")
    if result.elapsed_vlm_s > 0:
        parts.append(f"VLM {result.elapsed_vlm_s:.1f}s")
    if result.elapsed_total_s > 0:
        parts.append(f"Total {result.elapsed_total_s:.1f}s")
    return " · ".join(parts) if parts else ""


def _format_status(result: PipelineResult) -> str:
    """Format status line."""
    if not result.supported:
        return f"⚠️ {result.intent} — coming soon"
    model = result.model_used or "unknown"
    return f"✅ {result.intent} via {model}"


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

def analyze(image, image_t2, image_sar, query: str, use_demo: str, history_state: list) -> tuple:
    """
    Run the SatQuery pipeline on an image + query.

    Args:
        image: Primary satellite image (optical, or T1 for change detection).
        image_t2: Second image for change detection (optional).
        image_sar: SAR image for joint analysis (optional).
        query: User query string.
        use_demo: Demo dropdown selection.
        history_state: Current query history.

    Returns:
        (answer_md, timing_md, status_md, image_preview, annotated_preview,
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
        result = get_pipeline().run(
            image, query,
            image_t2_path=image_t2 or None,
            image_sar_path=image_sar or None,
        )
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
    """Format history entries as compact horizontal chips."""
    if not history:
        return "*No queries yet.*"
    entries = list(reversed(history))
    return " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(entries)


# ── Demo loader ───────────────────────────────────────────────────

def load_demo(demo_name: str):
    """Load a demo scenario into the image and query fields."""
    if not demo_name or demo_name == "None":
        return None, ""

    demo = get_demo_by_name(demo_name)
    if demo:
        return demo["image"], demo["query"]
    return None, ""


# ── Custom CSS ────────────────────────────────────────────────────

_CSS = """
/* ── Header ──────────────────────────────────────────────── */
.satquery-header {
    background: linear-gradient(135deg, #0F2B3D 0%, #0D7377 100%);
    color: white;
    padding: 16px 24px;
    border-radius: 10px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.satquery-header .title-area {
    display: flex;
    align-items: center;
    gap: 12px;
}
.satquery-header .logo {
    font-size: 1.6em;
    line-height: 1;
}
.satquery-header h1 {
    margin: 0;
    font-size: 1.4em;
    font-weight: 700;
    color: white;
    letter-spacing: -0.02em;
}
.satquery-header .subtitle {
    font-size: 0.82em;
    color: rgba(255,255,255,0.75);
    margin-top: 2px;
}
.satquery-header .controls {
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Cards ───────────────────────────────────────────────── */
.input-card, .result-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px;
}
.input-card {
    border-top: 3px solid #0D7377;
}
.result-card {
    border-top: 3px solid #1B2A4A;
}

/* ── Section labels ──────────────────────────────────────── */
.section-label {
    font-size: 0.9em;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #E2E8F0;
}

/* ── Visual Evidence placeholder ─────────────────────────── */
.evidence-placeholder {
    background: #F8FAFC;
    border: 2px dashed #CBD5E1;
    border-radius: 8px;
    padding: 40px 20px;
    text-align: center;
    color: #94A3B8;
    font-size: 0.9em;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
}
.evidence-placeholder .icon {
    font-size: 2em;
    opacity: 0.5;
}

/* ── Timing chip ─────────────────────────────────────────── */
.timing-chip {
    display: inline-block;
    background: #F1F5F9;
    color: #475569;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.82em;
    font-family: 'SF Mono', 'Fira Code', monospace;
}

/* ── Status chip ─────────────────────────────────────────── */
.status-chip {
    display: inline-block;
    background: #ECFDF5;
    color: #065F46;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.82em;
}

/* ── History ─────────────────────────────────────────────── */
.history-bar {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.88em;
    color: #64748B;
}

/* ── Footer ──────────────────────────────────────────────── */
.footer-bar {
    text-align: center;
    padding: 10px 0;
    font-size: 0.8em;
    color: #94A3B8;
    border-top: 1px solid #E2E8F0;
    margin-top: 8px;
}

/* ── Analyze button ──────────────────────────────────────── */
.analyze-btn {
    background: #0D7377 !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
}
.analyze-btn:hover {
    background: #0A5E61 !important;
}
"""


# ── Build Gradio UI ───────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    demo_names = ["None"] + get_demo_list()

    with gr.Blocks(
        title="SatQuery AI — Remote Sensing Assistant",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as app:

        # ── Branded Header ───────────────────────────────────
        gr.HTML("""
        <div class="satquery-header">
            <div class="title-area">
                <span class="logo">🛰️</span>
                <div>
                    <h1>SatQuery AI</h1>
                    <div class="subtitle">ISRO SIH26167 · Remote Sensing Vision-Language Assistant</div>
                </div>
            </div>
            <div class="controls">
                <span style="font-size:0.82em; color:rgba(255,255,255,0.6);">
                    EarthDial 4B + YOLOv8 SAR + BIT-CD · RTX 3050
                </span>
            </div>
        </div>
        """)

        # ── Demo Quick-Select ────────────────────────────────
        with gr.Row():
            demo_dropdown = gr.Dropdown(
                choices=demo_names,
                value="None",
                label="📋 Quick Demo — pre-computed scenarios",
                info="Instant results · no model loading required",
                scale=4,
            )
            gr.HTML("", scale=1)  # spacer

        # ── Main Two-Column Layout ──────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT: Inputs ────────────────────────────────
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML('<div class="section-label">🛰️ Satellite Image</div>')
                    image_input = gr.Image(
                        type="filepath",
                        label=None,
                        height=300,
                    )
                    gr.HTML(
                        '<div class="section-label" style="margin-top:8px;">'
                        '🔄 Change Detection (optional)</div>'
                    )
                    image_t2_input = gr.Image(
                        type="filepath",
                        label=None,
                        height=200,
                        placeholder="Upload a second (later) image for change detection",
                    )
                    gr.HTML(
                        '<div class="section-label" style="margin-top:8px;">'
                        '📡 SAR Image (optional — for joint optical+SAR analysis)</div>'
                    )
                    image_sar_input = gr.Image(
                        type="filepath",
                        label=None,
                        height=200,
                        placeholder="Upload a SAR image for joint analysis with the optical image",
                    )

                with gr.Group():
                    gr.HTML('<div class="section-label">💬 Query</div>')
                    query_input = gr.Textbox(
                        label=None,
                        placeholder=(
                            "Describe this image · Are there buildings? · "
                            "Detect ships in SAR"
                        ),
                        lines=2,
                    )
                    with gr.Row():
                        analyze_btn = gr.Button(
                            "🔍 Analyze",
                            variant="primary",
                            elem_classes=["analyze-btn"],
                        )
                        clear_btn = gr.Button(
                            "Clear",
                            variant="secondary",
                        )

                # ── Visual Evidence (always visible) ─────────
                with gr.Group():
                    gr.HTML('<div class="section-label">🖼️ Visual Evidence</div>')
                    annotated_output = gr.Image(
                        label=None,
                        height=300,
                        interactive=False,
                        value=None,
                    )

            # ── RIGHT: Results ──────────────────────────────
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML('<div class="section-label">📊 Analysis Result</div>')
                    status_output = gr.Markdown(
                        value="*Upload a satellite image and ask a question, "
                              "or select a demo scenario above.*"
                    )
                    timing_output = gr.Markdown(value="")
                    answer_output = gr.Markdown(
                        value=(
                            "**Supported query types:**\n\n"
                            "| Type | Example |\n"
                            "|------|---------|\n"
                            "| 📝 Captioning | *\"Describe this satellite image\"* |\n"
                            "| ❓ Visual QA | *\"Are there buildings here?\"* |\n"
                            "| 🎯 Detection | *\"Find all structures\"* |\n"
                            "| 🏷️ Classification | *\"Classify the land cover\"* |\n"
                            "| 📍 Grounding | *\"Locate key features\"* |\n"
                            "| 📡 SAR Detection | *\"Detect ships in this SAR image\"* |\n"
                            "| 🔄 Change Detection | *\"What changed between these two images?\"* (requires T2 image) |\n"
                            "| 🔗 Joint Analysis | *\"Analyze optical and SAR images together\"* (requires SAR image) |"
                        ),
                    )

        # ── History Bar ─────────────────────────────────────
        gr.HTML('<div class="section-label" style="margin-top:8px;">📜 Query History</div>')
        history_output = gr.Markdown(
            value="*No queries yet.*",
            elem_classes=["history-bar"],
        )

        # ── Footer ──────────────────────────────────────────
        gr.HTML("""
        <div class="footer-bar">
            🛰️ SatQuery AI · EarthDial 4B (InternVL + Phi-3) · YOLOv8 SAR ·
            NVIDIA RTX 3050 (4 GB) ·
            <a href="https://github.com/sanjayvelu08/SatQuery-AI"
               style="color:#0D7377;">GitHub</a>
        </div>
        """)

        # ── State ───────────────────────────────────────────
        history_state = gr.State([])

        # ── Wire events ─────────────────────────────────────
        demo_dropdown.change(
            fn=load_demo,
            inputs=[demo_dropdown],
            outputs=[image_input, query_input],
        )

        analyze_args = dict(
            fn=analyze,
            inputs=[image_input, image_t2_input, image_sar_input, query_input, demo_dropdown, history_state],
            outputs=[answer_output, timing_output, status_output,
                     image_input, annotated_output, history_output, history_state],
        )

        analyze_btn.click(**analyze_args)
        query_input.submit(**analyze_args)

        def clear_all():
            return (
                None,           # image
                None,           # image_t2
                None,           # image_sar
                "",             # query
                "None",         # demo
                "*No queries yet.*",  # history
                [],             # history state
                (
                    "**Supported query types:**\n\n"
                    "| Type | Example |\n"
                    "|------|---------|\n"
                    "| 📝 Captioning | *\"Describe this satellite image\"* |\n"
                    "| ❓ Visual QA | *\"Are there buildings here?\"* |\n"
                    "| 🎯 Detection | *\"Find all structures\"* |\n"
                    "| 🏷️ Classification | *\"Classify the land cover\"* |\n"
                    "| 📍 Grounding | *\"Locate key features\"* |\n"
                    "| 📡 SAR Detection | *\"Detect ships in this SAR image\"* |\n"
                    "| 🔄 Change Detection | *\"What changed between these two images?\"* |\n"
                    "| 🔗 Joint Analysis | *\"Analyze optical and SAR images together\"* |"
                ),
                "",             # timing
                "*Upload a satellite image and ask a question, "
                "or select a demo scenario above.*",  # status
                None,           # annotated image
            )

        clear_btn.click(
            fn=clear_all,
            outputs=[image_input, image_t2_input, image_sar_input, query_input, demo_dropdown,
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
