"""Tests for query-conditioned text-guided grounding (SIH 26167).

Covers: target phrase extraction, prompt conditioning, strict bbox parsing,
the verified squash-resize inverse coordinate mapping (non-square images),
structured grounding evidence on PipelineResult, zero-detection handling, the
generic-detect fallback, and API serialization.
"""

import os
import tempfile
import types
import unittest

from PIL import Image

from satquery.grounding import (
    build_grounding_prompt,
    extract_target,
    normalized_to_pixel,
    parse_grounding_bboxes,
)
from satquery.pipeline import SatQueryPipeline
from satquery.router import classify


def _tmp_image(name="gnd.png", size=(800, 600), color=(90, 120, 90)):
    p = os.path.join(tempfile.gettempdir(), name)
    Image.new("RGB", size, color).save(p)
    return p


class FakeVLM:
    """Records the prompt actually sent to EarthDial."""

    def __init__(self, answer="[[20, 30, 60, 70, 85]]"):
        self.calls = []
        self.answer = answer

    def query(self, image_path, prompt, max_tokens=50, num_beams=2):
        self.calls.append({"image_path": image_path, "prompt": prompt})
        return types.SimpleNamespace(
            answer=self.answer, model_loaded=True, elapsed_s=0.01)


class TestExtractTarget(unittest.TestCase):
    def test_highlight_with_reference(self):
        self.assertEqual(
            extract_target("Highlight the water body referred to in the query."),
            "water body")

    def test_locate_simple(self):
        self.assertEqual(extract_target("Locate the airport."), "airport")

    def test_show_me(self):
        self.assertEqual(extract_target("Show me the buildings"), "buildings")

    def test_where_is(self):
        self.assertEqual(extract_target("Where is the river?"), "river")

    def test_find_any(self):
        self.assertEqual(extract_target("Find any vehicles"), "vehicles")

    def test_mark_all_the(self):
        self.assertEqual(
            extract_target("Mark all the water bodies in the image"),
            "water bodies")

    def test_point_out(self):
        self.assertEqual(
            extract_target("Please point out the industrial area on the image"),
            "industrial area")

    def test_quoted_target(self):
        self.assertEqual(
            extract_target('Highlight "agricultural fields"'), "agricultural fields")

    def test_no_target_returns_none(self):
        self.assertIsNone(extract_target("Detect all key objects and features"))
        self.assertIsNone(extract_target("Describe the image"))
        self.assertIsNone(extract_target(""))
        self.assertIsNone(extract_target("Show me the objects"))
        self.assertIsNone(extract_target("Highlight it"))


class TestPromptConditioning(unittest.TestCase):
    def test_prompt_contains_target(self):
        prompt = build_grounding_prompt("water body")
        self.assertIn("water body", prompt)
        self.assertIn("[[x1, y1, x2, y2, confidence]]", prompt)
        self.assertIn("normalized 0-100", prompt)

    def test_prompt_contract_parseable(self):
        prompt = build_grounding_prompt("airport")
        # The requested contract must match the strict parser.
        boxes = parse_grounding_bboxes(
            "The box is [[10, 20, 30, 40, 90]] as requested.")
        self.assertEqual(len(boxes), 1)
        self.assertIn("airport", prompt)


class TestParseBBoxes(unittest.TestCase):
    def test_valid_box_with_confidence(self):
        boxes = parse_grounding_bboxes("[[10, 20, 30, 40, 90]]")
        self.assertEqual(len(boxes), 1)
        b = boxes[0]
        self.assertEqual((b["x1"], b["y1"], b["x2"], b["y2"]),
                         (10.0, 20.0, 30.0, 40.0))
        self.assertAlmostEqual(b["confidence"], 0.9)

    def test_valid_box_without_confidence(self):
        boxes = parse_grounding_bboxes("[[5, 6, 7, 8]]")
        self.assertEqual(len(boxes), 1)
        self.assertIsNone(boxes[0]["confidence"])  # never fabricated

    def test_multiple_boxes(self):
        boxes = parse_grounding_bboxes(
            "One: [[1, 2, 3, 4, 50]] and another [[10, 10, 20, 20]]")
        self.assertEqual(len(boxes), 2)

    def test_prose_ignored(self):
        boxes = parse_grounding_bboxes(
            "No water bodies are visible in this image.")
        self.assertEqual(boxes, [])

    def test_out_of_range_dropped(self):
        boxes = parse_grounding_bboxes("[[150, 20, 30, 40, 90]]")
        self.assertEqual(boxes, [])

    def test_inverted_box_dropped(self):
        boxes = parse_grounding_bboxes("[[50, 50, 10, 10, 90]]")
        self.assertEqual(boxes, [])

    def test_empty(self):
        self.assertEqual(parse_grounding_bboxes(""), [])
        self.assertEqual(parse_grounding_bboxes(None), [])


class TestCoordinateMapping(unittest.TestCase):
    def test_normalized_to_pixel_linear(self):
        # 800x600 non-square image: mapping must be linear per axis because
        # EarthDial's eval transform is a squash resize (no letterbox padding).
        px = normalized_to_pixel(
            {"x1": 50, "y1": 50, "x2": 100, "y2": 100}, 800, 600)
        self.assertEqual((px["x1"], px["y1"]), (400.0, 300.0))
        self.assertEqual((px["x2"], px["y2"]), (800.0, 600.0))

    def test_asymmetric_scaling(self):
        px = normalized_to_pixel(
            {"x1": 25, "y1": 10, "x2": 75, "y2": 90}, 800, 600)
        self.assertEqual((px["x1"], px["y1"]), (200.0, 60.0))
        self.assertEqual((px["x2"], px["y2"]), (600.0, 540.0))

    def test_frame_independence_of_normalized_coords(self):
        # Normalized 0-100 coords are frame-independent: mapping from the
        # original 800x600 image equals mapping from the 448x448 model frame
        # scaled by the per-axis dimension ratio (squash, no padding).
        box = {"x1": 20, "y1": 30, "x2": 60, "y2": 70}
        model_px = normalized_to_pixel(box, 448, 448)
        orig_px = normalized_to_pixel(box, 800, 600)
        self.assertAlmostEqual(orig_px["x1"], model_px["x1"] * (800 / 448))
        self.assertAlmostEqual(orig_px["y1"], model_px["y1"] * (600 / 448))
        self.assertAlmostEqual(orig_px["x2"], model_px["x2"] * (800 / 448))
        self.assertAlmostEqual(orig_px["y2"], model_px["y2"] * (600 / 448))


class TestPipelineGrounding(unittest.TestCase):
    def _run(self, query, answer, image=None):
        img = image or _tmp_image()
        fake = FakeVLM(answer=answer)
        pipe = SatQueryPipeline(vlm=fake)
        result = pipe.run(img, query)
        return fake, result

    def test_grounding_prompt_conditioned_on_target(self):
        fake, result = self._run(
            "Highlight the water body referred to in the query.",
            "The water body is [[20, 30, 60, 70, 85]].")
        self.assertEqual(result.intent, "grounding")
        self.assertTrue(result.supported)
        # The prompt sent to EarthDial contains the extracted target.
        self.assertIn("water body", fake.calls[0]["prompt"])
        self.assertNotIn("key objects and features", fake.calls[0]["prompt"])

    def test_structured_detections_with_pixel_coords(self):
        _, result = self._run(
            "Locate the airport.",
            "Airport box: [[25, 10, 75, 90, 80]].")
        self.assertEqual(len(result.grounding_detections), 1)
        det = result.grounding_detections[0]
        self.assertEqual(det["target"], "airport")
        # 800x600 image -> linear squash inverse mapping.
        self.assertEqual(det["x1"], 200.0)   # 25/100 * 800
        self.assertEqual(det["y1"], 60.0)    # 10/100 * 600
        self.assertEqual(det["x2"], 600.0)
        self.assertEqual(det["y2"], 540.0)
        self.assertAlmostEqual(det["confidence"], 0.8)

    def test_confidence_none_when_not_provided(self):
        _, result = self._run(
            "Find the ship.", "Ship: [[10, 10, 20, 20]].")
        self.assertEqual(len(result.grounding_detections), 1)
        self.assertIsNone(result.grounding_detections[0]["confidence"])

    def test_zero_detections(self):
        fake, result = self._run(
            "Highlight the water body referred to in the query.",
            "No water body is visible in this image.")
        self.assertTrue(result.supported)
        self.assertEqual(result.grounding_detections, [])
        step = next(s for s in result.trace if s.name == "grounding")
        self.assertEqual(step.status, "skipped")
        self.assertIn("0 box(es)", step.output_summary)

    def test_generic_detect_falls_back_to_generic_prompt(self):
        fake, result = self._run(
            "Detect all key objects and features",
            "Box: [[5, 5, 50, 50, 90]].")
        self.assertEqual(result.intent, "detect")
        # No extractable target -> generic detection prompt used.
        self.assertIn("main features", fake.calls[0]["prompt"])
        self.assertEqual(result.grounding_detections[0]["target"], "detected")

    def test_trace_records_target_frame_params_count(self):
        _, result = self._run(
            "Highlight the water body referred to in the query.",
            "[[20, 30, 60, 70, 85]]")
        vlm_step = next(s for s in result.trace if s.name == "vlm_infer")
        self.assertIn("target=water body", vlm_step.input_summary)
        self.assertIn("frame=squash-448", vlm_step.input_summary)
        self.assertIn("contract=[[x1,y1,x2,y2,conf]]", vlm_step.input_summary)
        gnd_step = next(s for s in result.trace if s.name == "grounding")
        self.assertEqual(gnd_step.tool, "earthdial_grounding")
        self.assertIn("1 box(es) parsed for 'water body'", gnd_step.output_summary)

    def test_visual_evidence_still_created(self):
        _, result = self._run(
            "Locate the airport.",
            "Airport: [[25, 10, 75, 90, 80]].")
        self.assertIsNotNone(result.annotated_image)
        self.assertTrue(os.path.isfile(result.annotated_image))

    def test_to_dict_serializes_detections(self):
        _, result = self._run(
            "Locate the airport.", "[[25, 10, 75, 90, 80]]")
        d = result.to_dict()
        self.assertIn("grounding_detections", d)
        self.assertEqual(d["grounding_detections"][0]["target"], "airport")

    def test_vqa_untouched(self):
        # Non-grounding intents must not gain grounding machinery.
        fake, result = self._run(
            "Is there a river in this image?", "Yes, a river is visible.")
        self.assertEqual(result.intent, "vqa")
        self.assertIn("Is there a river in this image?", fake.calls[0]["prompt"])
        self.assertEqual(result.grounding_detections, [])
        self.assertNotIn("grounding", [s.name for s in result.trace])


class TestRouting(unittest.TestCase):
    def test_highlight_routes_to_grounding(self):
        r = classify("Highlight the water body referred to in the query.")
        self.assertEqual(r.primary_intent, "grounding")

    def test_locate_routes_to_detect(self):
        # Existing routing behavior preserved: "locate" matches detect first.
        r = classify("Locate the airport.")
        self.assertEqual(r.primary_intent, "detect")


if __name__ == "__main__":
    unittest.main()