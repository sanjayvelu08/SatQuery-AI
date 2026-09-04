"""
SatQuery AI — agentic auditability unit tests (mocked / pure, no GPU).

Covers the SIH 26167 auditability milestone:
  1. trace serialization preserves 'error' on failed steps (pipeline + api shape)
  2. change-path evidence reliability: deterministic, bounded, rule-driven
  3. stale SAR router state fixed (supported=True, current reason text)
  4. build_result_summary derives the compact API summary correctly

Run:  python -X utf8 -m unittest -v satquery.test_agentic_audit
"""

import os
import sys
import types
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from satquery.router import classify  # noqa: E402
from satquery.pipeline import (  # noqa: E402
    PipelineResult, build_result_summary, compute_change_reliability,
)
from satquery.evidence import ExecutionTraceStep  # noqa: E402
from satquery.bit_tool import ChangeDetectionResult, ChangeRegion  # noqa: E402


def _change_result(success=True, detected=True, regions=2, overlay=True,
                   error=None):
    return ChangeDetectionResult(
        success=success,
        change_detected=detected,
        change_pct=3.4 if detected else 0.0,
        regions=[ChangeRegion(region_id=1, bbox=[1, 1, 8, 8],
                              area_pixels=100, area_pct=1.0,
                              width=8, height=8)] * regions,
        num_regions=regions,
        summary="Detected changes." if detected else "No significant changes.",
        overlay_path="/tmp/ov.png" if overlay else None,
        postprocessing_ms=12.0, inference_time_ms=40.0, total_ms=300.0,
        vram_peak_mb=81.0, img_size=256,
        error=error)


def _trace_step(name, status="ok", error=None, input_summary="in",
                output_summary="out"):
    return ExecutionTraceStep(
        step=1, name=name, tool="tool", status=status, duration_ms=1.0,
        input_summary=input_summary, output_summary=output_summary,
        error=error)


class TestTraceSerialization(unittest.TestCase):
    def test_pipeline_to_dict_preserves_error(self):
        step = _trace_step("validate", status="error", error="corrupt file")
        pr = PipelineResult(
            query="q", image_path="i.png", intent="vqa", all_intents=["vqa"],
            supported=False, unsupported_reason="corrupt file",
            trace=[step])
        d = pr.to_dict()
        self.assertEqual(d["trace"][0]["error"], "corrupt file")
        self.assertEqual(d["trace"][0]["tool"], "tool")
        self.assertEqual(d["trace"][0]["status"], "error")

    def test_step_without_error_serializes_null(self):
        pr = PipelineResult(
            query="q", image_path="i.png", intent="caption",
            all_intents=["caption"], supported=True,
            trace=[_trace_step("route")])
        self.assertIsNone(pr.to_dict()["trace"][0]["error"])


class TestChangeReliability(unittest.TestCase):
    def test_deterministic_and_bounded(self):
        cr = _change_result()
        a = compute_change_reliability(cr, interpretation_requested=True,
                                       interpretation_produced=True,
                                       pair_verdict={"status": "compatible",
                                                     "co_registration": "unverified"})
        b = compute_change_reliability(cr, interpretation_requested=True,
                                       interpretation_produced=True,
                                       pair_verdict={"status": "compatible",
                                                     "co_registration": "unverified"})
        self.assertEqual(a, b)  # reproducible
        self.assertGreaterEqual(a[0], 0.0)
        self.assertLessEqual(a[0], 1.0)
        self.assertTrue(a[1])

    def test_high_when_all_evidence_present(self):
        cr = _change_result()
        score, reason = compute_change_reliability(
            cr, interpretation_requested=True, interpretation_produced=True,
            pair_verdict={"status": "compatible", "co_registration": "verified"})
        self.assertEqual(score, 1.0)
        self.assertIn("co-registration verified", reason)

    def test_failure_returns_zero(self):
        cr = _change_result(success=False, error="boom")
        score, reason = compute_change_reliability(cr)
        self.assertEqual(score, 0.0)
        self.assertIn("no reliability", reason)

    def test_missing_interpretation_lowers_score(self):
        cr = _change_result()
        with_interp = compute_change_reliability(
            cr, interpretation_requested=True, interpretation_produced=True)
        without = compute_change_reliability(
            cr, interpretation_requested=True, interpretation_produced=False)
        self.assertGreater(with_interp[0], without[0])

    def test_no_change_detected_still_reliable_absence(self):
        cr = _change_result(detected=False, regions=0)
        score, reason = compute_change_reliability(
            cr, interpretation_requested=False, interpretation_produced=False)
        self.assertEqual(score, 1.0)  # absent-change is itself a valid result
        self.assertIn("no change detected", reason)

    def test_incompatible_pair_penalizes(self):
        cr = _change_result()
        score, _ = compute_change_reliability(
            cr, interpretation_requested=False, interpretation_produced=False,
            pair_verdict={"status": "incompatible", "co_registration": "unverified"})
        self.assertLess(score, 1.0)


class TestSarRouterState(unittest.TestCase):
    def test_sar_supported_and_current(self):
        r = classify("Analyze the SAR backscatter in this image.")
        self.assertEqual(r.primary_intent, "sar")
        self.assertTrue(r.supported)
        self.assertIn("YOLOv8 SAR vessel", r.reason)
        self.assertNotIn("not yet integrated", r.reason)
        self.assertNotIn("only optical", r.reason)


class TestResultSummary(unittest.TestCase):
    def test_change_summary_fields(self):
        pr = PipelineResult(
            query="What changed?", image_path="t1.png", image_t2_path="t2.png",
            intent="change", all_intents=["change"], supported=True,
            model_used="BIT-CD (LEVIR-CD pretrained)",
            evidence_reliability=0.9, reliability_reasoning="BIT-CD ok",
            pair_compat={"warnings": ["co-registration unverified — "
                                      "insufficient geospatial metadata"]},
            trace=[_trace_step("route"), _trace_step("validate")])
        s = build_result_summary(pr)
        self.assertEqual(s["query"], "What changed?")
        self.assertEqual(s["intent"], "change")
        self.assertIn("BIT-CD", s["models_used"])
        self.assertEqual(s["evidence_reliability"], 0.9)
        self.assertEqual(s["reliability_reasoning"], "BIT-CD ok")
        self.assertEqual(s["trace_step_count"], 2)
        self.assertTrue(s["warnings"])
        self.assertIsNone(s["reliability_note"])

    def test_joint_summary_uses_models_used_list(self):
        jr = types.SimpleNamespace(models_used=["YOLOv8 SAR Vessel Detector",
                                                "AlignEarth-SAR-ViT-B-16 "
                                                "(zero-shot scene)",
                                                "EarthDial 4B RGB"])
        pr = PipelineResult(
            query="Use the optical and SAR images together to identify "
                  "built-up and water-covered regions.",
            image_path="o.tif", image_sar_path="s.tif",
            intent="joint_analysis", all_intents=["joint_analysis"],
            supported=True, model_used="EarthDial 4B + YOLOv8 SAR (Joint)",
            joint_result=jr, evidence_reliability=0.86,
            reliability_reasoning="both specialists succeeded", trace=[])
        s = build_result_summary(pr)
        self.assertIn("EarthDial 4B RGB", s["models_used"])
        self.assertIn("YOLOv8", s["models_used"])
        self.assertEqual(s["evidence_reliability"], 0.86)

    def test_single_qualitative_note_not_fake_number(self):
        pr = PipelineResult(
            query="Describe this image", image_path="i.png",
            intent="caption", all_intents=["caption"], supported=True,
            model_used="EarthDial 4B RGB", trace=[])
        s = build_result_summary(pr)
        self.assertIsNone(s["evidence_reliability"])
        self.assertIn("not quantified", s["reliability_note"])

    def test_unsupported_warnings(self):
        pr = PipelineResult(
            query="q", image_path="i.png", intent="vqa", all_intents=["vqa"],
            supported=False, unsupported_reason="No image provided\nline2",
            trace=[])
        s = build_result_summary(pr)
        self.assertEqual(s["warnings"], ["No image provided", "line2"])


if __name__ == "__main__":
    unittest.main()
