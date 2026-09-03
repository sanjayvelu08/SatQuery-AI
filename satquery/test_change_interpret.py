"""
Change-interpretation unit tests — mocked (no real EarthDial / BIT-CD).

Run:  python -m unittest -v satquery.test_change_interpret

Covers:
  A. "What changed?"                    → change + BIT-CD + interpretation
  B. "Where did changes occur?"         → change, interpretation skipped
  C. "Has built-up area increased?"     → rerouted to change + interpretation
  D. BIT-CD failure                     → interpretation NOT invoked
  E. EarthDial failure                  → BIT-CD result preserved, trace degraded
  F. Single-image request               → never becomes change
  G. Joint optical+SAR request          → never becomes change
  H. T1|T2 composite                    → both halves present, deterministic, cleaned
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from satquery.pipeline import SatQueryPipeline
from satquery.sar_tool import SARResult
from satquery.bit_tool import ChangeDetectionResult, ChangeRegion
from satquery.change_interpret import (
    build_change_prompt,
    build_t1_t2_composite,
    run_change_interpretation,
    should_interpret,
)


class FakeVLM:
    """Stand-in for SatQueryVLM.query()."""

    def __init__(self, answer="", model_loaded=True, raise_error=False):
        self.calls = []
        self.answer = answer
        self.model_loaded = model_loaded
        self.raise_error = raise_error

    def query(self, image_path, prompt, max_tokens=50, num_beams=2):
        if self.raise_error:
            raise RuntimeError("simulated EarthDial subprocess crash")
        self.calls.append({
            "image_path": image_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "num_beams": num_beams,
        })
        return types.SimpleNamespace(
            answer=self.answer, model_loaded=self.model_loaded, elapsed_s=0.2
        )

    def load(self):
        pass

    def unload(self):
        pass

    @property
    def is_loaded(self):
        return True


def make_change_result(success=True, detected=True, overlay_path=None):
    """A realistic ChangeDetectionResult without running BIT-CD.

    overlay_path: if provided (a real file), the pipeline records a
    visual_evidence trace step exactly like a real BIT-CD run.
    """
    if detected:
        regions = [
            ChangeRegion(region_id=1, bbox=[10, 20, 60, 80],
                         area_pixels=1200, area_pct=1.8, width=51, height=61),
            ChangeRegion(region_id=2, bbox=[150, 30, 220, 90],
                         area_pixels=900, area_pct=1.4, width=71, height=61),
        ]
    else:
        regions = []
    return ChangeDetectionResult(
        success=success,
        change_detected=detected,
        change_pct=3.4 if detected else 0.0,
        regions=regions,
        num_regions=len(regions),
        summary=(
            "Detected 2 spatially distinct changed region(s) covering 3.4% of the image."
            if detected else "No significant changes detected between the two images."
        ),
        overlay_path=overlay_path,
        postprocessing_ms=12.0,
        inference_time_ms=40.0,
        total_ms=300.0,
        vram_peak_mb=81.0,
        img_size=256,
        error=None if success else "simulated BIT-CD model failure",
    )


class FakeBitTool:
    def __init__(self, result=None):
        self.result = result
        self.last_call = None

    def detect(self, image_t1_path, image_t2_path, output_dir, **kwargs):
        self.last_call = (image_t1_path, image_t2_path, output_dir)
        return self.result or make_change_result()


class ChangeInterpretationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="satq_change_test_")
        cls.t1 = cls._img("t1.png", (120, 80, 40))
        cls.t2 = cls._img("t2.png", (40, 80, 200))
        cls.sar_img = cls._img("sar.png", (30, 30, 30))
        cls.overlay = cls._img("overlay.png", (10, 10, 10))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    @classmethod
    def _img(cls, name, color):
        p = os.path.join(cls._tmp, name)
        Image.new("RGB", (64, 48), color).save(p)
        return p

    def _run(self, query, vlm=None, t2=True, bit=None):
        pipe = SatQueryPipeline(vlm=vlm or FakeVLM())
        with mock.patch("satquery.bit_tool.get_bit_tool", return_value=bit or FakeBitTool()):
            return pipe.run(
                self.t1, query,
                image_t2_path=self.t2 if t2 else None,
                image_sar_path=None,
            )

    def step(self, result, name):
        return next((t for t in result.trace if t.name == name), None)

    # ── A. "What changed?" ───────────────────────────────────────
    def test_a_what_changed_runs_interpretation(self):
        vlm = FakeVLM(answer="The later image appears to show new structures in R1, "
                             "consistent with new construction.")
        # Overlay present (as in a real BIT-CD run) -> visual_evidence step
        bit = FakeBitTool(make_change_result(overlay_path=self.overlay))
        r = self._run("What changed between these two images?", vlm=vlm, bit=bit)

        self.assertEqual(r.intent, "change")
        self.assertTrue(r.supported)
        self.assertIsNotNone(r.change_result)
        names = [t.name for t in r.trace]
        self.assertEqual(names, [
            "route", "validate", "specialist_selection", "bit_cd_detect",
            "region_extraction", "visual_evidence", "change_interpret",
            "final_answer",
        ])
        vis = self.step(r, "visual_evidence")
        self.assertEqual(vis.status, "ok")
        self.assertEqual(vis.tool, "bit_cd")
        interp = self.step(r, "change_interpret")
        self.assertEqual(interp.status, "ok")
        self.assertEqual(interp.tool, "earthdial_change_interpreter")
        self.assertGreater(interp.duration_ms, 0)
        self.assertLess(len(interp.output_summary), 200)  # no huge VLM dump

        # Exactly one EarthDial call, with the interpretation token budget
        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(vlm.calls[0]["max_tokens"], 180)
        prompt = vlm.calls[0]["prompt"]
        self.assertIn("CHANGE STATISTICS", prompt)
        self.assertIn("3.4%", prompt)
        self.assertIn("Do NOT invent object counts", prompt)
        self.assertIn("What changed between these two images?", prompt)
        self.assertIn("Region R1", prompt)
        self.assertIn("appears consistent with", prompt)

        # Answer combines BIT-CD evidence + interpretation
        self.assertIn("Change Detection Analysis", r.answer)
        self.assertIn("What the changes appear to be", r.answer)
        self.assertIn("consistent with new construction", r.answer)

        # Composite temp artifact was cleaned up after the VLM call
        comp = vlm.calls[0]["image_path"]
        self.assertTrue(comp.endswith("t1_t2_composite.png"))
        self.assertFalse(os.path.exists(comp))
        self.assertNotIn(comp, (self.t1, self.t2))

    # ── B. "Where did changes occur?" → skipped interpretation ──
    def test_b_where_only_skips_interpretation(self):
        vlm = FakeVLM()
        r = self._run("Where did the changes occur between these two images?", vlm=vlm)

        self.assertTrue(r.supported)
        interp = self.step(r, "change_interpret")
        self.assertIsNotNone(interp)
        self.assertEqual(interp.status, "skipped")
        self.assertIn("location", interp.output_summary)
        self.assertEqual(len(vlm.calls), 0)  # no EarthDial call
        # BIT statistics / regions preserved
        self.assertEqual(r.change_result.change_pct, 3.4)
        self.assertEqual(r.change_result.num_regions, 2)
        self.assertNotIn("What the changes appear to be", r.answer)

    # ── C. "Has built-up area increased?" → reroute + interpret ──
    def test_c_semantic_question_reroutes_to_change(self):
        vlm = FakeVLM(answer="The changed regions are visually consistent with an "
                             "increase in built-up area; I cannot quantify it.")
        r = self._run("Has built-up area increased?", vlm=vlm)

        self.assertEqual(r.intent, "change")
        route_step = self.step(r, "route")
        self.assertIn("rerouted_to=change", route_step.output_summary)
        interp = self.step(r, "change_interpret")
        self.assertEqual(interp.status, "ok")
        self.assertEqual(len(vlm.calls), 1)
        prompt = vlm.calls[0]["prompt"]
        # Cautious wording is enforced in the prompt
        for clause in (
            "Do NOT estimate physical area",
            "Do NOT treat the change percentage",
            "Do NOT provide geospatial coordinates",
            "Do NOT invent object counts",
            "appears consistent with",
        ):
            self.assertIn(clause, prompt)
        self.assertIn("consistent with", r.answer)

    # ── D. BIT-CD failure → no interpretation ────────────────────
    def test_d_bit_failure_skips_interpretation(self):
        vlm = FakeVLM()
        r = self._run("What changed between these two images?", vlm=vlm,
                      bit=FakeBitTool(make_change_result(success=False)))
        self.assertFalse(r.supported)
        self.assertIsNone(self.step(r, "change_interpret"))
        self.assertEqual(len(vlm.calls), 0)
        detect_step = self.step(r, "bit_cd_detect")
        self.assertEqual(detect_step.status, "error")
        self.assertIn("simulated BIT-CD model failure", r.unsupported_reason)

    # ── E. EarthDial failure → BIT-CD result preserved ───────────
    def test_e_interpretation_failure_degrades_gracefully(self):
        # Case 1: model returns no usable answer (model_loaded=False)
        vlm = FakeVLM(model_loaded=False)
        r = self._run("What changed between these two images?", vlm=vlm)
        self.assertTrue(r.supported)
        self.assertIsNotNone(r.change_result)
        self.assertEqual(r.change_result.change_pct, 3.4)
        interp = self.step(r, "change_interpret")
        self.assertEqual(interp.status, "failed")
        self.assertIn("BIT-CD statistics preserved", interp.output_summary)
        self.assertIn("Change Detection Analysis", r.answer)
        self.assertNotIn("What the changes appear to be", r.answer)

        # Case 2: VLM raises → still degrades, never a 500
        vlm2 = FakeVLM(raise_error=True)
        r2 = self._run("What changed between these two images?", vlm=vlm2)
        self.assertTrue(r2.supported)
        self.assertEqual(self.step(r2, "change_interpret").status, "failed")
        self.assertIsNotNone(r2.change_result)

    # ── F. Single-image request → NOT change ─────────────────────
    def test_f_single_image_never_becomes_change(self):
        vlm = FakeVLM(answer="coastal scene")
        pipe = SatQueryPipeline(vlm=vlm)
        r = pipe.run(self.t1, "What is the land cover type of this area?")
        self.assertNotEqual(r.intent, "change")
        self.assertIsNone(r.change_result)
        self.assertIsNone(self.step(r, "change_interpret"))
        self.assertEqual(len(vlm.calls), 1)
        # The VLM saw the original single image, not a composite
        self.assertEqual(vlm.calls[0]["image_path"], self.t1)

    # ── G. Joint optical+SAR request → NOT change ────────────────
    def test_g_joint_request_not_rerouted_to_change(self):
        vlm = FakeVLM(answer="port scene with water")
        pipe = SatQueryPipeline(vlm=vlm)
        sar_ok = SARResult(success=True, detections=[], num_detections=0,
                           inference_time_ms=10.0, gpu_vram_mb=21.0)
        with mock.patch("satquery.pipeline.run_sar_detection", return_value=sar_ok):
            r = pipe.run(
                self.t1, "analyze optical and sar imagery together",
                image_sar_path=self.sar_img,
            )
        self.assertEqual(r.intent, "joint_analysis")
        self.assertIsNone(self.step(r, "change_interpret"))
        names = [t.name for t in r.trace]
        self.assertIn("sar_detect", names)
        self.assertIn("fuse", names)
        self.assertIn("confidence", names)
        self.assertNotIn("bit_cd_detect", names)
        # optical + joint interpretation = exactly two EarthDial calls
        self.assertEqual(len(vlm.calls), 2)

    # ── H. Composite generation ───────────────────────────────────
    def test_h_composite_has_both_halves_and_is_deterministic(self):
        red = os.path.join(self._tmp, "comp_red.png")
        blue = os.path.join(self._tmp, "comp_blue.png")
        Image.new("RGB", (64, 64), (200, 0, 0)).save(red)
        Image.new("RGB", (64, 64), (0, 0, 200)).save(blue)

        out1 = os.path.join(self._tmp, "c1.png")
        out2 = os.path.join(self._tmp, "c2.png")
        p1 = build_t1_t2_composite(red, blue, out_path=out1)
        p2 = build_t1_t2_composite(red, blue, out_path=out2)

        im = Image.open(p1)
        self.assertEqual(im.size, (512, 256))  # deterministic layout
        arr = im.convert("RGB")

        # Left half = T1 (red), right half = T2 (blue)
        left = arr.crop((0, 0, 256, 256))
        right = arr.crop((256, 0, 512, 256))
        lpx = list(left.getdata())
        rpx = list(right.getdata())
        self.assertGreater(sum(p[0] for p in lpx), sum(p[2] for p in lpx))
        self.assertGreater(sum(p[2] for p in rpx), sum(p[0] for p in rpx))

        # Deterministic: identical bytes for identical inputs
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())
        for f in (p1, p2):
            os.remove(f)
            self.assertFalse(os.path.exists(f))

    # ── Prompt / heuristic unit checks ────────────────────────────
    def test_should_interpret_heuristic(self):
        run_cases = [
            "What changed?",
            "Describe the changes between the images",
            "Has construction increased?",
            "Has built-up area increased?",
            "Did anything change between these dates?",
            "What does the change look like?",
        ]
        skip_cases = [
            "Where did the changes occur?",
            "How much of the image changed?",
            "What percentage of the area changed?",
            "Locate the changed regions",
        ]
        for q in run_cases:
            self.assertTrue(should_interpret(q)[0], q)
        for q in skip_cases:
            self.assertFalse(should_interpret(q)[0], q)

    def test_build_change_prompt_includes_constraints(self):
        res = make_change_result()
        prompt = build_change_prompt(
            query="What changed?", change_pct=res.change_pct,
            num_regions=res.num_regions, regions=res.regions,
        )
        for token in (
            "BIT-CD", "authoritative", "3.4%", "Region R1",
            "T1 (before / earlier image)", "T2 (after / later image)",
            "Do NOT invent object counts",
            "Do NOT estimate physical area in m²",
            "Do NOT claim that buildings, roads, vegetation, or water changed",
            "Do NOT treat the change percentage",
            "Do NOT provide geospatial coordinates",
            "appears consistent with",
        ):
            self.assertIn(token, prompt)


if __name__ == "__main__":
    unittest.main()
