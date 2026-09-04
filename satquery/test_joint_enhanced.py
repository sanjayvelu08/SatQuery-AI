"""
SatQuery AI — mocked tests for the enhanced Optical + SAR joint workflow.

Covers:
  1. Pair-aware routing: image_sar supplied => joint workflow even without
     the literal words "optical" and "SAR" (pipeline-level reroute).
  2. SAR evidence: scene_scores + intensity_indicators fields + honest
     limitations.
  3. Fusion: capabilities/gaps derived from SAR-CLIP scores and indicators;
     joint prompt separates OPTICAL / SAR / DERIVED INDICATORS / UNCERTAIN.
  4. Joint execution: trace steps (sarclip_scene, otsu_indicators,
     pair_composite), composite image passed to the interpretation call.

All specialists are mocked — no real models, no GPU, no subprocesses.

Run:  python -X utf8 -m unittest -v satquery.test_joint_enhanced
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from satquery.router import classify  # noqa: E402
from satquery.evidence import (  # noqa: E402
    SAREvidence, OpticalEvidence, FusedEvidence, JointAnalysisResult)

from satquery.fusion import (  # noqa: E402
    build_optical_sar_composite,
    build_joint_interpretation_prompt,
    fuse_evidence,
    run_joint_interpretation,
)
from satquery.pipeline import SatQueryPipeline  # noqa: E402


def _tmp_image(name="t.png", size=(64, 64), color=(120, 120, 120)):
    p = os.path.join(tempfile.gettempdir(), name)
    Image.new("RGB", size, color).save(p)
    return p


class FakeVLM:
    """Stand-in for SatQueryVLM.query() that records calls."""

    def __init__(self, answer="joint answer"):
        self.calls = []
        self.answer = answer

    def query(self, image_path, prompt, max_tokens=50, num_beams=2):
        self.calls.append({"image_path": image_path, "prompt": prompt,
                           "path_ok_at_call": os.path.isfile(image_path)})
        return types.SimpleNamespace(
            answer=self.answer, model_loaded=True, elapsed_s=0.01)


FAKE_SAR_RAW = types.SimpleNamespace(
    success=True, num_detections=2, inference_time_ms=50.0,
    gpu_vram_mb=21.0, error=None)


def fake_sarclip(image_path):
    return {
        "success": True,
        "scores": {
            "coarse": {"water": 0.70, "urban or built-up area": 0.20,
                       "vegetation": 0.05, "agriculture": 0.05},
            "fine": {"water or river": 0.60, "building roof or house": 0.25,
                     "background": 0.15},
        },
        "total_ms": 3000.0,
    }


class TestPairAwareRouting(unittest.TestCase):
    def test_classify_unchanged(self):
        r = classify("Identify built-up and water-covered regions in this pair.")
        self.assertNotIn("joint_analysis", r.all_intents)

    def test_pipeline_reroutes_when_sar_supplied(self):
        opt = _tmp_image("r_opt.png", color=(30, 90, 30))
        sar = _tmp_image("r_sar.png", color=(60, 60, 60))
        pipe = SatQueryPipeline(vlm=FakeVLM())
        with mock.patch("satquery.pipeline.run_sar_detection",
                        return_value=FAKE_SAR_RAW), \
             mock.patch("satquery.pipeline.format_sar_response",
                        return_value="2 vessel(s) detected."), \
             mock.patch("satquery.pipeline.run_sarclip_scene",
                        side_effect=fake_sarclip), \
             mock.patch("satquery.pipeline.otsu_intensity_indicators",
                        return_value={"threshold": 95,
                                      "dark_fraction": 0.7,
                                      "bright_fraction": 0.1,
                                      "mid_fraction": 0.2,
                                      "dark_components_ge50": 1,
                                      "usable_mask": True}):
            result = pipe.run(
                opt, "Identify built-up and water-covered regions in this pair.",
                image_sar_path=sar)
        self.assertEqual(result.intent, "joint_analysis")
        self.assertTrue(result.supported)
        trace_names = [t.name for t in result.trace]
        self.assertIn("pair_composite", trace_names)
        self.assertIn("sarclip_scene", trace_names)
        self.assertIn("otsu_indicators", trace_names)
        # joint interpretation must have been called on the composite image
        vlm_calls = pipe.vlm.calls
        self.assertEqual(len(vlm_calls), 2)  # optical + joint interpretation
        interp_call = vlm_calls[1]
        self.assertTrue(interp_call["path_ok_at_call"])  # composite existed when queried
        self.assertTrue(interp_call["image_path"].endswith(".png"))
        self.assertIn("left half = OPTICAL", interp_call["prompt"])
        self.assertIn("right half = SAR", interp_call["prompt"])
        self.assertIn("=== OPTICAL EVIDENCE", interp_call["prompt"])
        self.assertIn("=== SAR EVIDENCE", interp_call["prompt"])
        self.assertIn("=== DERIVED INDICATORS", interp_call["prompt"])
        self.assertIn("=== UNCERTAIN OR INFERRED", interp_call["prompt"])
        # the joint composite is now persisted visual evidence — tidy up
        jr = result.joint_result
        if jr and jr.composite_path and os.path.isfile(jr.composite_path):
            os.remove(jr.composite_path)


class TestSAREvidence(unittest.TestCase):
    def test_extra_fields_and_limitations(self):
        ev = SAREvidence(
            source="yolov8_sar_vessel", image_path="s.png",
            num_detections=2, inference_time_ms=50.0, gpu_vram_mb=21.0,
            success=True,
            scene_scores={"coarse": {"water": 0.7}},
            intensity_indicators={"threshold": 95, "dark_fraction": 0.7,
                                  "bright_fraction": 0.1},
        )
        self.assertEqual(ev.scene_scores["coarse"]["water"], 0.7)
        self.assertEqual(ev.intensity_indicators["dark_fraction"], 0.7)
        limits = " ".join(ev.limitations).lower()
        self.assertIn("zero-shot", limits)
        self.assertIn("not verified pixel-level semantic segmentation", limits)
        self.assertIn("image-relative statistics", limits)
        caps = " ".join(ev.capabilities).lower()
        self.assertIn("zero-shot sar scene labels", caps)
        self.assertIn("intensity indicators", caps)


class TestFusion(unittest.TestCase):
    def _evidence(self):
        opt = OpticalEvidence(
            source="earthdial", answer="Agricultural fields with scattered "
            "buildings and no large water body.", intent="vqa",
            prompt_sent="p", image_path="o.png", elapsed_s=1.0, success=True)
        sar = SAREvidence(
            source="yolov8_sar_vessel", image_path="s.png",
            num_detections=0, inference_time_ms=50.0, gpu_vram_mb=21.0,
            success=True,
            scene_scores={"coarse": {"water": 0.8, "urban or built-up area": 0.1,
                                     "vegetation": 0.05, "agriculture": 0.05}},
            intensity_indicators={"threshold": 95, "dark_fraction": 0.6,
                                  "bright_fraction": 0.15, "mid_fraction": 0.25,
                                  "dark_components_ge50": 2},
        )
        return opt, sar

    def test_fuse_capabilities_from_new_evidence(self):
        opt, sar = self._evidence()
        fused = fuse_evidence(opt, sar)
        caps = " ".join(fused.joint_capabilities)
        self.assertIn("dominant label 'water'", caps)
        self.assertIn("dark/water-like", caps)
        self.assertIn("bright/built-up-like", caps)
        gaps = " ".join(fused.unresolved_gaps).lower()
        self.assertIn("not verified pixel-level semantic segmentation", gaps)

    def test_joint_prompt_sections(self):
        p = build_joint_interpretation_prompt(
            original_query="Identify built-up and water-covered regions.",
            optical_answer="Some built-up and water visible.",
            sar_detection_summary="0 vessels.",
            sar_limitations=["zero-shot labels", "no segmentation"],
            sar_scene_scores={"coarse": {"water": 0.8}},
            sar_intensity_indicators={"dark_fraction": 0.6,
                                      "bright_fraction": 0.1,
                                      "threshold": 95},
        )
        self.assertIn("=== OPTICAL EVIDENCE", p)
        self.assertIn("=== SAR EVIDENCE", p)
        self.assertIn("=== DERIVED INDICATORS", p)
        self.assertIn("=== UNCERTAIN OR INFERRED", p)
        self.assertIn("not pixel-level semantic segmentation", p)
        self.assertIn("image-relative statistics", p)

    def test_composite_builder(self):
        opt = _tmp_image("c_opt.png", color=(10, 120, 10))
        sar = _tmp_image("c_sar.png", color=(80, 80, 80))
        out = build_optical_sar_composite(opt, sar)
        try:
            im = Image.open(out)
            im.load()
            self.assertEqual(im.size, (512, 256))
            im.close()
        finally:
            if os.path.exists(out):
                os.remove(out)


class TestJointAnswerFormatting(unittest.TestCase):
    """The final answer must deterministically separate the four evidence
    categories (optical / SAR / derived indicators / joint conclusion)
    regardless of the model's prose, without claiming segmentation."""

    def _joint_result(self):
        opt = OpticalEvidence(
            source="earthdial",
            answer="Agricultural fields with a dark elongated water body; "
            "no dense settlement visible.",
            intent="joint_context", prompt_sent="p", image_path="o.png",
            elapsed_s=1.0, success=True)
        sar = SAREvidence(
            source="yolov8_sar_vessel", image_path="s.png",
            num_detections=0, inference_time_ms=50.0, gpu_vram_mb=21.0,
            success=True, detection_summary="0 vessels detected.",
            scene_scores={"coarse": {"urban or built-up area": 0.91,
                                      "water": 0.06},
                          "fine": {"water or river": 0.90,
                                    "cropland": 0.05}},
            intensity_indicators={"threshold": 132,
                                  "dark_fraction": 0.42,
                                  "bright_fraction": 0.32,
                                  "dark_components_ge50": 18},
        )
        return JointAnalysisResult(
            query="Use the optical and SAR images together to identify "
                  "built-up and water-covered regions.",
            optical_evidence=opt, sar_evidence=sar, fused_evidence=None,
            joint_answer="Dark smooth SAR returns align with the optical water "
                         "body; built-up evidence is weak.",
            confidence=0.8, confidence_reasoning="ok", trace=[])

    def test_answer_separates_four_evidence_sections(self):
        md = self._joint_result().format_markdown()
        markers = ["### 🔍 OPTICAL EVIDENCE",
                   "### 🛰️ SAR EVIDENCE",
                   "### 📊 DERIVED INTENSITY INDICATORS",
                   "### 🤝 JOINT CONCLUSION"]
        positions = [md.index(m) for m in markers]
        self.assertEqual(positions, sorted(positions),  # order preserved
                         "evidence sections must appear in a fixed order")
        # each category carries its real evidence
        self.assertIn("dark elongated water body", md)
        self.assertIn("0 vessels detected", md)
        self.assertIn("water or river 90.0%", md)   # fine native labels
        self.assertIn("dark/water-like returns 42.0%", md)
        # honesty: never presented as segmentation
        low = md.lower()
        self.assertIn("not pixel-level semantic segmentation", low)
        self.assertIn("not semantic segmentation", low)
        self.assertIn("what remains uncertain", low)


class TestJointExecution(unittest.TestCase):
    def test_run_joint_interpretation_uses_composite(self):
        opt = _tmp_image("j_opt.png", color=(10, 120, 10))
        sar = _tmp_image("j_sar.png", color=(80, 80, 80))
        composite = build_optical_sar_composite(opt, sar)
        fake = FakeVLM(answer="water dominates; built-up is minimal.")
        optical = OpticalEvidence(
            source="earthdial", answer="Some water and built-up.",
            intent="joint_context", prompt_sent="p", image_path=opt,
            elapsed_s=1.0, success=True)
        sarv = SAREvidence(
            source="yolov8", image_path=sar, num_detections=0,
            inference_time_ms=50.0, gpu_vram_mb=21.0, success=True,
            detection_summary="0 vessels.",
            scene_scores={"coarse": {"water": 0.8}},
            intensity_indicators={"dark_fraction": 0.6,
                                  "bright_fraction": 0.1, "threshold": 95},
        )
        try:
            answer, elapsed = run_joint_interpretation(
                fake, "Identify water regions.", optical, sarv,
                composite_path=composite)
            self.assertEqual(answer, "water dominates; built-up is minimal.")
            self.assertEqual(fake.calls[0]["image_path"], composite)
        finally:
            if os.path.exists(composite):
                os.remove(composite)


class TestJointEvidencePersistence(unittest.TestCase):
    """The OPTICAL|SAR composite must be persisted as visual evidence and
    surfaced as composite_url, not deleted after interpretation."""

    def _run_joint(self):
        opt = _tmp_image("p_opt.png", color=(20, 90, 20))
        sar = _tmp_image("p_sar.png", color=(70, 70, 70))
        pipe = SatQueryPipeline(vlm=FakeVLM())
        with mock.patch("satquery.pipeline.run_sar_detection",
                        return_value=FAKE_SAR_RAW), \
             mock.patch("satquery.pipeline.format_sar_response",
                        return_value="1 vessel(s) detected."), \
             mock.patch("satquery.pipeline.run_sarclip_scene",
                        side_effect=fake_sarclip), \
             mock.patch("satquery.pipeline.otsu_intensity_indicators",
                        return_value={"threshold": 95, "dark_fraction": 0.7,
                                      "bright_fraction": 0.1,
                                      "dark_components_ge50": 1,
                                      "usable_mask": True}):
            return pipe.run(
                opt, "Use the optical and SAR images together to identify "
                "built-up and water-covered regions.",
                image_sar_path=sar)

    def test_composite_persisted_and_url(self):
        r = self._run_joint()
        jr = r.joint_result
        self.assertTrue(jr.composite_path)
        self.assertTrue(os.path.isfile(jr.composite_path),
                        "composite must persist after interpretation")
        try:
            d = jr.to_dict()
            self.assertTrue(d["composite_url"].startswith("/changes/"))
            # reliability mirrors the joint confidence, deterministically
            self.assertIsNotNone(r.evidence_reliability)
            self.assertGreaterEqual(r.evidence_reliability, 0.0)
            self.assertLessEqual(r.evidence_reliability, 1.0)
            self.assertTrue(r.reliability_reasoning)
        finally:
            os.remove(jr.composite_path)

    def test_trace_records_configured_parameters(self):
        r = self._run_joint()
        jr = r.joint_result
        try:
            by_name = {t.name: t for t in r.trace}
            self.assertIn("conf=0.25", by_name["sar_detect"].input_summary)
            self.assertIn("labels=coarse+fine",
                          by_name["sarclip_scene"].input_summary)
            self.assertIn("tokens=50 beams=2",
                          by_name["optical_analyze"].input_summary)
            self.assertIn("pair_composite", by_name)
            self.assertIn("composite", by_name["pair_composite"].output_summary)
        finally:
            if jr.composite_path and os.path.isfile(jr.composite_path):
                os.remove(jr.composite_path)


if __name__ == "__main__":
    unittest.main()