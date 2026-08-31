"""
SatQuery AI — Image visualization utilities (Loop 6).

Draws bounding boxes, labels, and confidence scores on SAR and optical images.
Used for both SAR vessel detection and EarthDial grounding output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont


@dataclass
class BBox:
    """A single bounding box with label and confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float = 0.0


# Color palette for different object types
_COLORS = [
    (0, 120, 255),    # Blue
    (255, 60, 60),    # Red
    (0, 200, 100),    # Green
    (255, 165, 0),    # Orange
    (180, 0, 255),    # Purple
    (0, 200, 200),    # Cyan
    (255, 255, 0),    # Yellow
    (255, 0, 128),    # Magenta
]


def _get_color(index: int) -> tuple[int, int, int]:
    """Get a color from the palette by index."""
    return _COLORS[index % len(_COLORS)]


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """Get a font for labels. Falls back to default if truetype unavailable."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def draw_bboxes(
    image_path: str,
    bboxes: List[BBox],
    output_path: str | None = None,
    title: str = "",
) -> str:
    """
    Draw bounding boxes on an image and save the annotated version.

    Args:
        image_path: Path to the original image.
        bboxes: List of BBox objects to draw.
        output_path: Where to save the annotated image. If None, auto-generates.
        title: Optional title to draw at the top of the image.

    Returns:
        Path to the annotated image.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(14)
    font_small = _get_font(11)
    w, h = img.size

    # Title bar
    if title:
        draw.rectangle([(0, 0), (w, 28)], fill=(0, 0, 0, 180))
        draw.text((8, 5), title, fill=(255, 255, 255), font=font)

    # Draw each bounding box
    for i, bbox in enumerate(bboxes):
        color = _get_color(i)
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Draw rectangle (3px thick)
        for offset in range(3):
            draw.rectangle(
                [(x1 - offset, y1 - offset), (x2 + offset, y2 + offset)],
                outline=color,
            )

        # Label background
        label_text = f"{bbox.label} {bbox.confidence:.0%}" if bbox.confidence > 0 else bbox.label
        bbox_text = draw.textbbox((0, 0), label_text, font=font_small)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        # Position label above or below the box
        label_y = max(0, y1 - text_h - 6) if y1 > text_h + 10 else y2 + 3
        label_x = x1

        draw.rectangle(
            [(label_x, label_y), (label_x + text_w + 8, label_y + text_h + 4)],
            fill=color,
        )
        draw.text(
            (label_x + 4, label_y + 2),
            label_text,
            fill=(255, 255, 255),
            font=font_small,
        )

    # Legend at bottom if multiple detections
    if len(bboxes) > 1:
        legend_y = h - 22 * min(len(bboxes), 5) - 5
        draw.rectangle([(5, legend_y), (200, h - 2)], fill=(0, 0, 0))
        for i, bbox in enumerate(bboxes[:5]):
            color = _get_color(i)
            y_pos = legend_y + 5 + i * 18
            draw.rectangle([(10, y_pos), (22, y_pos + 12)], fill=color)
            draw.text(
                (28, y_pos),
                f"{bbox.label}: {bbox.confidence:.0%}",
                fill=(255, 255, 255),
                font=font_small,
            )

    # Save
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_annotated{ext}"

    img.save(output_path, quality=95)
    return output_path


def parse_grounding_output(answer: str, image_width: int, image_height: int) -> List[BBox]:
    """
    Parse EarthDial grounding output to extract bounding boxes.

    EarthDial returns coordinates normalized 0-100 in format:
    [[x1, y1, x2, y2, confidence]]

    Returns:
        List of BBox objects with pixel coordinates.
    """
    import re

    bboxes = []

    # Find all [[...]] patterns
    pattern = r'\[\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(?:\s*,\s*(\d+(?:\.\d+)?))?\]\]'
    matches = re.findall(pattern, answer)

    for match in matches:
        try:
            x1_norm, y1_norm, x2_norm, y2_norm = float(match[0]), float(match[1]), float(match[2]), float(match[3])
            conf = float(match[4]) / 100.0 if match[4] else 0.0

            # Convert from normalized 0-100 to pixel coordinates
            x1 = x1_norm / 100.0 * image_width
            y1 = y1_norm / 100.0 * image_height
            x2 = x2_norm / 100.0 * image_width
            y2 = y2_norm / 100.0 * image_height

            bboxes.append(BBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                label="detected",
                confidence=conf,
            ))
        except (ValueError, IndexError):
            continue

    return bboxes


def create_annotated_image(
    image_path: str,
    answer: str,
    intent: str,
    output_path: str | None = None,
) -> str | None:
    """
    Create an annotated image from pipeline output.
    Handles both SAR detections and optical grounding.

    Returns:
        Path to annotated image, or None if no visualization possible.
    """
    try:
        img = Image.open(image_path)
        w, h = img.size
    except Exception:
        return None

    bboxes = []

    if intent == "sar":
        # Parse SAR detection output — bboxes are in pixel coords
        import re
        pattern = r'\|\s*\d+\s*\|\s*(\w+)\s*\|\s*(\d+\.?\d*)%\s*\|\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]\s*\|'
        matches = re.findall(pattern, answer)
        for match in matches:
            try:
                bboxes.append(BBox(
                    x1=float(match[1]),
                    y1=float(match[2]),
                    x2=float(match[3]),
                    y2=float(match[4]),
                    label=match[0],
                    confidence=float(match[0 + 1]) / 100.0 if False else int(match[1]) / 100.0,
                ))
            except (ValueError, IndexError):
                continue

        # Fix confidence parsing
        bboxes = []
        for match in matches:
            try:
                label = match[0]  # "Ship"
                conf_pct = float(match[1])  # e.g. 76.3
                x1, y1, x2, y2 = float(match[2]), float(match[3]), float(match[4]), float(match[5])
                bboxes.append(BBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    label=label,
                    confidence=conf_pct / 100.0,
                ))
            except (ValueError, IndexError):
                continue

        if bboxes:
            title = f"SAR Vessel Detection — {len(bboxes)} target(s)"
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_sar_annotated{ext}"
            return draw_bboxes(image_path, bboxes, output_path, title=title)

    elif intent in ("detect", "grounding"):
        # Parse EarthDial grounding output
        bboxes = parse_grounding_output(answer, w, h)
        if bboxes:
            title = f"Feature Detection — {len(bboxes)} region(s)"
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_grounding_annotated{ext}"
            return draw_bboxes(image_path, bboxes, output_path, title=title)

    return None
