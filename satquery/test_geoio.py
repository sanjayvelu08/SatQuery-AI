"""
SatQuery AI — GeoTIFF/TIFF probe, render, and pair-compatibility tests.

Pure unittest (no pytest). Generates tiny synthetic TIFF/GeoTIFF fixtures with
tifffile + numpy so no large dataset is needed:

  rgb8.tif      3-band uint8 RGB (Photometric=RGB)
  ms13.tif      13-band int16 multispectral (Sentinel-2-like layout)
  sars2.tif     2-band float32 (Sentinel-1-like VV/VH)
  gray16.tif    1-band int16 with GDAL nodata
  nan32.tif     1-band float32 containing NaN
  geo_4326.tif  32x32 georeferenced (EPSG:4326, pixel scale, tiepoint)
  geo_4326_b.tif  identical grid copy of geo_4326 (distinct path)
  geo_4326_far.tif same CRS, disjoint spatial extent
  geo_32643.tif projected EPSG:32643, 10 m pixels
  geo_diffres.tif same CRS/extent as geo_4326 but coarser pixels
  plain.tif     metadata-free TIFF

Also covers the pipeline integration (GeoTIFF prepare step producing
geo_probe + pair_compat_check trace steps) with specialists mocked.

Run:  python -X utf8 -m unittest -v satquery.test_geoio
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import tifffile
from PIL import Image

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from satquery.geoio import (  # noqa: E402
    probe_image, render_rgb, suggest_rgb_bands,
    check_pair_compat, format_meta_line,
)
from satquery.pipeline import SatQueryPipeline  # noqa: E402


# ── synthetic GeoTIFF fixture writers ────────────────────────────

def _geokeys(crs_type: str, epsg: int) -> list:
    model = 1 if crs_type == "projected" else 2
    key = 3072 if crs_type == "projected" else 2048
    return [1, 1, 0, 2, 1024, 0, 1, model, key, 0, 1, epsg]


def _extratags(keys, scale, tiepoint, nodata=None):
    tags = [
        (34735, "H", len(keys), keys),
        (33550, "d", 3, [float(scale), float(scale), 0.0]),
        (33922, "d", 6, [float(v) for v in tiepoint]),
    ]
    if nodata is not None:
        tags.append((42113, "s", len(str(nodata)), str(nodata).encode("ascii")))
    return tags


class GeoIoFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="satq_geoio_")
        W = cls._tmp

        # 1. 3-band uint8 RGB
        rgb = np.zeros((64, 64, 3), np.uint8)
        rgb[..., 0] = 200; rgb[..., 1] = 80; rgb[..., 2] = 30
        cls.rgb8 = os.path.join(W, "rgb8.tif")
        tifffile.imwrite(cls.rgb8, rgb, photometric="rgb")

        # 2. 13-band int16, S2-like (indices 3/2/1 = "true colour")
        ms = np.full((13, 64, 64), 1000, np.int16)
        ms[3] = np.linspace(0, 20000, 64 * 64, dtype=np.int16).reshape(64, 64)  # red-ish
        ms[2] = np.linspace(20000, 0, 64 * 64, dtype=np.int16).reshape(64, 64)  # green-ish
        ms[1] = np.full((64, 64), 4000, np.int16)                                # blue-ish
        cls.ms13 = os.path.join(W, "ms13.tif")
        tifffile.imwrite(cls.ms13, ms, photometric="minisblack",
                         planarconfig="separate")

        # 3. 2-band float32 SAR-like VV/VH
        yy, xx = np.mgrid[0:64, 0:64] / 64.0
        sar = np.stack([yy.astype(np.float32), (0.5 * xx).astype(np.float32)])
        cls.sars2 = os.path.join(W, "sars2.tif")
        tifffile.imwrite(cls.sars2, sar, photometric="minisblack",
                         planarconfig="separate")

        # 4. 1-band int16 with nodata
        g16 = np.arange(64 * 64, dtype=np.int16).reshape(64, 64) % 2000
        g16[0:8, 0:8] = -9999
        cls.gray16 = os.path.join(W, "gray16.tif")
        tifffile.imwrite(cls.gray16, g16, photometric="minisblack",
                         extratags=_extratags(_geokeys("geographic", 4326),
                                              scale=1e-4, tiepoint=(0, 0, 0, 10.0, 20.0, 0.0),
                                              nodata=-9999))

        # 5. 1-band float32 gradient with NaN + extreme outlier
        nan = np.linspace(0.0, 900.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        nan[0:10, 0:10] = np.nan
        nan[30:34, 30:34] = 1e9  # extreme outlier -> stretch must not blow out
        cls.nan32 = os.path.join(W, "nan32.tif")
        tifffile.imwrite(cls.nan32, nan, photometric="minisblack")

        # 6/7/8. georeferenced 32x32 grids
        gray = (np.arange(32 * 32, dtype=np.uint16) % 1000).reshape(32, 32)
        cls.geo4326 = os.path.join(W, "geo_4326.tif")
        tifffile.imwrite(cls.geo4326, gray, photometric="minisblack",
                         extratags=_extratags(_geokeys("geographic", 4326),
                                              scale=1e-4,
                                              tiepoint=(0, 0, 0, 10.0, 20.0, 0.0)))
        cls.geo4326_b = os.path.join(W, "geo_4326_b.tif")
        tifffile.imwrite(cls.geo4326_b, gray.copy(), photometric="minisblack",
                         extratags=_extratags(_geokeys("geographic", 4326),
                                              scale=1e-4,
                                              tiepoint=(0, 0, 0, 10.0, 20.0, 0.0)))
        cls.geo4326_far = os.path.join(W, "geo_4326_far.tif")
        tifffile.imwrite(cls.geo4326_far, gray.copy(), photometric="minisblack",
                         extratags=_extratags(_geokeys("geographic", 4326),
                                              scale=1e-4,
                                              tiepoint=(0, 0, 0, 80.0, 90.0, 0.0)))
        cls.geo32643 = os.path.join(W, "geo_32643.tif")
        tifffile.imwrite(cls.geo32643, gray.copy(), photometric="minisblack",
                         extratags=_extratags(_geokeys("projected", 32643),
                                              scale=10.0,
                                              tiepoint=(0, 0, 0, 500000.0, 4000000.0, 0.0)))
        cls.geo_diffres = os.path.join(W, "geo_diffres.tif")
        tifffile.imwrite(cls.geo_diffres, gray.copy(), photometric="minisblack",
                         extratags=_extratags(_geokeys("geographic", 4326),
                                              scale=2e-4,
                                              tiepoint=(0, 0, 0, 10.0, 20.0, 0.0)))

        # 9. metadata-free TIFF
        cls.plain = os.path.join(W, "plain.tif")
        tifffile.imwrite(cls.plain, gray.copy(), photometric="minisblack")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _open_rgb(self, path):
        im = Image.open(path)
        arr = np.asarray(im.convert("RGB"))
        im.close()
        return arr


class TestProbe(GeoIoFixtures):
    def test_rgb8_probe(self):
        m = probe_image(self.rgb8)
        self.assertTrue(m["is_tiff"])
        self.assertEqual(m["width"], 64)
        self.assertEqual(m["height"], 64)
        self.assertEqual(m["bands"], 3)
        self.assertEqual(m["dtype"], "uint8")
        self.assertEqual(m["format"], "tiff")
        self.assertIsNone(m["epsg"])

    def test_ms13_probe(self):
        m = probe_image(self.ms13)
        self.assertEqual(m["bands"], 13)
        self.assertEqual(m["dtype"], "int16")
        self.assertEqual(m["width"], 64)

    def test_sars2_probe(self):
        m = probe_image(self.sars2)
        self.assertEqual(m["bands"], 2)
        self.assertEqual(m["dtype"], "float32")

    def test_gray16_nodata_probe(self):
        m = probe_image(self.gray16)
        self.assertEqual(m["bands"], 1)
        self.assertEqual(m["dtype"], "int16")
        self.assertEqual(m["nodata"], -9999)

    def test_geo4326_epsg_and_bounds(self):
        m = probe_image(self.geo4326)
        self.assertEqual(m["format"], "geotiff")
        self.assertEqual(m["epsg"], 4326)
        self.assertEqual(m["crs_type"], "geographic")
        self.assertAlmostEqual(m["pixel_scale"][0], 1e-4)
        b = m["bounds"]
        self.assertAlmostEqual(b["left"], 10.0)
        self.assertAlmostEqual(b["top"], 20.0)
        self.assertAlmostEqual(b["right"], 10.0 + 32 * 1e-4)
        self.assertAlmostEqual(b["bottom"], 20.0 - 32 * 1e-4)

    def test_geo32643_projected(self):
        m = probe_image(self.geo32643)
        self.assertEqual(m["epsg"], 32643)
        self.assertEqual(m["crs_type"], "projected")
        self.assertAlmostEqual(m["pixel_scale"][0], 10.0)
        self.assertAlmostEqual(m["bounds"]["right"], 500000.0 + 32 * 10.0)

    def test_plain_metadata_free(self):
        m = probe_image(self.plain)
        self.assertEqual(m["format"], "tiff")
        self.assertIsNone(m["epsg"])
        self.assertIsNone(m["bounds"])
        self.assertIsNone(m["pixel_scale"])

    def test_unknowns_are_none_not_guessed(self):
        m = probe_image(self.plain)
        for key in ("epsg", "crs_type", "bounds", "pixel_scale",
                    "nodata", "date_time"):
            self.assertIsNone(m[key])

    def test_probe_missing_file_raises(self):
        with self.assertRaises(ValueError):
            probe_image(os.path.join(self._tmp, "does_not_exist.tif"))


class TestRender(GeoIoFixtures):
    def test_rgb_preserves_channels(self):
        out = render_rgb(self.rgb8)
        arr = self._open_rgb(out)
        self.assertEqual(arr.shape, (64, 64, 3))
        self.assertEqual(arr.dtype, np.uint8)
        # passthrough: uint8 RGB must not be stretched or shifted
        self.assertEqual(int(arr[0, 0, 0]), 200)
        self.assertEqual(int(arr[0, 0, 1]), 80)
        self.assertEqual(int(arr[0, 0, 2]), 30)

    def test_13band_render_not_all_white(self):
        out = render_rgb(self.ms13)
        arr = self._open_rgb(out)
        self.assertEqual(arr.shape, (64, 64, 3))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertLess(float((arr > 250).mean()), 0.5,
                        "multispectral render must not be the old all-white failure")
        self.assertGreater(float(arr.std()), 1.0)

    def test_band_selection_configurable(self):
        a = self._open_rgb(render_rgb(self.ms13))                       # S2 (3,2,1)
        b = self._open_rgb(render_rgb(self.ms13, band_indices=(0, 1, 2)))
        self.assertFalse(np.array_equal(a, b),
                         "changing band_indices must change the render")

    def test_2band_sar_render_uses_vv_vh(self):
        out = render_rgb(self.sars2)                                    # R=VV G=VH B=VV
        arr = self._open_rgb(out)
        self.assertEqual(arr.shape, (64, 64, 3))
        # R and B both come from band 0 -> identical after identical stretch
        self.assertTrue(np.array_equal(arr[..., 0], arr[..., 2]))
        # G comes from band 1 (VH, half amplitude) -> must differ somewhere
        self.assertFalse(np.array_equal(arr[..., 0], arr[..., 1]))

    def test_1band_render_grayscale(self):
        out = render_rgb(self.gray16)
        arr = self._open_rgb(out)
        self.assertEqual(arr.shape, (64, 64, 3))
        self.assertTrue(np.array_equal(arr[..., 0], arr[..., 1]))
        self.assertTrue(np.array_equal(arr[..., 1], arr[..., 2]))

    def test_nan_handled_and_black(self):
        out = render_rgb(self.nan32)
        arr = self._open_rgb(out)
        # NaN patch rendered black; rest finite and not blown out by 1e9 outlier
        self.assertEqual(int(arr[5, 5, 0]), 0)
        self.assertGreater(float(arr[40, 10, 0]), 0)
        self.assertLess(float(arr.max()), 256)
        self.assertTrue(np.all(np.isfinite(arr)))

    def test_nodata_pixels_black(self):
        out = render_rgb(self.gray16)
        arr = self._open_rgb(out)
        self.assertEqual(int(arr[3, 3, 0]), 0)     # nodata region
        self.assertGreater(int(arr[40, 40, 0]), 0)  # valid region stretched

    def test_suggest_rgb_bands_heuristics(self):
        self.assertEqual(suggest_rgb_bands(1, None), (0, 0, 0))
        self.assertEqual(suggest_rgb_bands(2, None), (0, 1, 0))
        self.assertEqual(suggest_rgb_bands(13, None), (3, 2, 1))
        self.assertEqual(suggest_rgb_bands(3, 2), (0, 1, 2))  # native RGB


class TestPairCompatibility(GeoIoFixtures):
    def test_identical_geogrid_verified(self):
        v = check_pair_compat(probe_image(self.geo4326),
                              probe_image(self.geo4326_b))
        self.assertEqual(v["status"], "compatible")
        self.assertEqual(v["co_registration"], "verified")
        self.assertIn("identical georeferenced pixel grid", v["summary"])

    def test_crs_mismatch_incompatible(self):
        v = check_pair_compat(probe_image(self.geo4326),
                              probe_image(self.geo32643))
        self.assertEqual(v["status"], "incompatible")
        self.assertIn("CRS mismatch", v["summary"])

    def test_disjoint_extents_incompatible(self):
        v = check_pair_compat(probe_image(self.geo4326),
                              probe_image(self.geo4326_far))
        self.assertEqual(v["status"], "incompatible")
        self.assertIn("do not overlap", v["summary"])

    def test_equal_dims_no_metadata_unverified(self):
        rgb_copy = os.path.join(self._tmp, "rgb8_copy.tif")
        tifffile.imwrite(rgb_copy, np.zeros((64, 64, 3), np.uint8),
                         photometric="rgb")
        v = check_pair_compat(probe_image(self.rgb8), probe_image(rgb_copy))
        self.assertEqual(v["status"], "compatible")
        # equal dimensions must NEVER imply co-registration
        self.assertEqual(v["co_registration"], "unverified")
        self.assertIn("insufficient geospatial metadata", v["summary"])
        os.remove(rgb_copy)

    def test_different_dims_warn_not_resample(self):
        v = check_pair_compat(probe_image(self.plain),   # 32x32
                              probe_image(self.rgb8))    # 64x64
        self.assertEqual(v["status"], "compatible")
        self.assertEqual(v["co_registration"], "unverified")
        self.assertIn("different pixel dimensions", v["summary"])
        self.assertIn("no resampling performed", v["summary"])

    def test_different_resolution_warns(self):
        v = check_pair_compat(probe_image(self.geo4326),
                              probe_image(self.geo_diffres))
        self.assertEqual(v["status"], "compatible")
        self.assertEqual(v["co_registration"], "unverified")
        self.assertIn("different pixel resolutions", v["summary"])

    def test_missing_metadata_warning_text(self):
        v = check_pair_compat(probe_image(self.rgb8), probe_image(self.rgb8))
        self.assertIn("co-registration unverified — insufficient geospatial metadata",
                      v["summary"])

    def test_format_meta_line(self):
        line = format_meta_line("Optical", probe_image(self.geo4326))
        self.assertIn("EPSG:4326", line)
        self.assertIn("32×32", line)


# ── pipeline integration (specialists mocked, CPU-only) ──────────

class FakeVLM:
    def __init__(self, answer="joint answer"):
        self.calls = []
        self.answer = answer

    def query(self, image_path, prompt, max_tokens=50, num_beams=2):
        self.calls.append({"image_path": image_path,
                           "path_ok_at_call": os.path.isfile(image_path)})
        return types.SimpleNamespace(answer=self.answer, model_loaded=True,
                                     elapsed_s=0.01)


FAKE_SAR_RAW = types.SimpleNamespace(
    success=True, num_detections=1, inference_time_ms=40.0,
    gpu_vram_mb=21.0, error=None)


def _fake_sarclip(image_path):
    return {"success": True,
            "scores": {"coarse": {"water": 0.7, "urban or built-up area": 0.2},
                       "fine": {"water or river": 0.6}},
            "total_ms": 100.0}


class TestPipelineGeoTiffIntegration(GeoIoFixtures):
    """GeoTIFF inputs must be probed + rendered to RGB PNGs before any
    specialist runs; paired flows must add pair_compat_check + verdicts."""

    def test_joint_geotiff_pair_verified(self):
        pipe = SatQueryPipeline(vlm=FakeVLM())
        with mock.patch("satquery.pipeline.run_sar_detection",
                        return_value=FAKE_SAR_RAW), \
             mock.patch("satquery.pipeline.format_sar_response",
                        return_value="1 vessel(s) detected."), \
             mock.patch("satquery.pipeline.run_sarclip_scene",
                        side_effect=_fake_sarclip), \
             mock.patch("satquery.pipeline.otsu_intensity_indicators",
                        return_value={"threshold": 90, "dark_fraction": 0.5,
                                      "bright_fraction": 0.2,
                                      "dark_components_ge50": 1,
                                      "usable_mask": True}):
            r = pipe.run(
                self.geo4326,
                "Use the optical and SAR images together to identify "
                "built-up and water-covered regions.",
                image_sar_path=self.geo4326_b)

        self.assertTrue(r.supported)
        names = [t.name for t in r.trace]
        self.assertEqual(names.count("geo_probe"), 2)
        self.assertIn("pair_compat_check", names)
        jr = r.joint_result
        self.assertEqual(jr.pair_compat["co_registration"], "verified")
        self.assertEqual(jr.optical_evidence.image_meta["epsg"], 4326)
        # optical call saw the rendered RGB PNG, never the raw TIFF; the
        # second call is the joint interpretation on the composite
        first = pipe.vlm.calls[0]
        self.assertTrue(first["image_path"].endswith("_geo_rgb.png"),
                        first["image_path"])
        self.assertTrue(first["path_ok_at_call"])
        self.assertTrue(pipe.vlm.calls[1]["image_path"].endswith(".png"))
        for c in pipe.vlm.calls:
            self.assertTrue(c["path_ok_at_call"])
        self.assertIn("SPATIAL PAIR COMPATIBILITY", r.answer)
        self.assertIn("verified", r.answer)

    def test_change_geotiff_metadata_free_unverified(self):
        from satquery.bit_tool import ChangeDetectionResult, ChangeRegion

        def fake_detect(image_t1_path, image_t2_path, output_dir, **kw):
            return ChangeDetectionResult(
                success=True, change_detected=False, change_pct=0.0,
                regions=[], num_regions=0,
                summary="No significant changes detected between the two images.",
                overlay_path=None, postprocessing_ms=1.0,
                inference_time_ms=1.0, total_ms=2.0,
                vram_peak_mb=0.0, img_size=256, error=None)

        pipe = SatQueryPipeline(vlm=FakeVLM())
        with mock.patch("satquery.bit_tool.get_bit_tool") as gb:
            gb.return_value.detect = fake_detect
            r = pipe.run(self.plain, "What changed between these two images?",
                         image_t2_path=self.gray16)

        self.assertTrue(r.supported)
        names = [t.name for t in r.trace]
        self.assertEqual(names.count("geo_probe"), 2)
        self.assertIn("pair_compat_check", names)
        self.assertEqual(r.pair_compat["co_registration"], "unverified")
        self.assertIn("Pair compatibility", r.answer)
        self.assertIn("insufficient geospatial metadata", r.answer)


if __name__ == "__main__":
    unittest.main()
