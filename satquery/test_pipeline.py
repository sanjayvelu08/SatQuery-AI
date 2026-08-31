"""
Loop 2: End-to-end pipeline test.

Tests:
  1. Captioning on sentinel2_optical.jpg
  2. VQA on sentinel2_optical.jpg
  3. Grounding on urban_optical.jpg
  4. Classification on sentinel2_optical.jpg
  5. Unsupported intent: change detection
  6. Unsupported intent: SAR query
  7. General / unknown intent on sentinel2_optical.jpg

Must be run with: python -X utf8 satquery/test_pipeline.py
"""

import os
import sys
import json
import time

# Ensure satquery package is importable
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from satquery.pipeline import SatQueryPipeline

# ── Test definitions ──────────────────────────────────────────────
IMG_S2 = os.path.join(ROOT, "test_images", "sentinel2_optical.jpg")
IMG_URBAN = os.path.join(ROOT, "test_images", "urban_optical.jpg")

TESTS = [
    # (image_path, query, expected_intent, expected_supported)
    (IMG_S2, "Please describe this satellite image.", "caption", True),
    (IMG_S2, "Are there any buildings visible in this image?", "vqa", True),
    (IMG_URBAN, "[grounding]Locate the main features in this image.", "grounding", True),
    (IMG_S2, "What is the land cover type of this area?", "classification", True),
    (IMG_S2, "What changed between these two images?", "change", False),
    (IMG_S2, "Analyze the SAR backscatter in this image.", "sar", False),
    (IMG_S2, "Tell me about this area.", "general", True),
]


def main():
    print("=" * 70)
    print("LOOP 2: SatQuery Pipeline End-to-End Test")
    print("=" * 70)
    print()

    pipeline = SatQueryPipeline()

    # Show VRAM before loading
    print(f"VRAM before model load: {pipeline.vlm.vram_info()}")
    print()

    results = []
    for i, (img, query, exp_intent, exp_supported) in enumerate(TESTS, 1):
        print(f"--- Test {i}/{len(TESTS)} ---")
        print(f"  Image:  {os.path.basename(img)}")
        print(f"  Query:  {query}")
        print(f"  Expect: intent={exp_intent}, supported={exp_supported}")

        if not os.path.exists(img):
            print(f"  SKIP: image not found at {img}")
            continue

        result = pipeline.run(img, query)
        results.append(result.to_dict())

        print()
        print(result.format())
        print()

        # Verify intent
        ok_intent = result.intent == exp_intent
        ok_support = result.supported == exp_supported
        status = "PASS" if (ok_intent and ok_support) else "FAIL"
        print(f"  Status: {status} (intent={'ok' if ok_intent else 'MISMATCH'}, "
              f"supported={'ok' if ok_support else 'MISMATCH'})")
        print()

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tests run:    {len(results)}")
    print(f"  VRAM after:   {pipeline.vlm.vram_info()}")

    # Write results to JSON
    out_path = os.path.join(ROOT, "loop2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}")
    print()

    # Verify all results
    passed = 0
    for r, (_, _, exp_intent, exp_sup) in zip(results, TESTS):
        ok = r["intent"] == exp_intent and r["supported"] == exp_sup
        if ok:
            passed += 1
    print(f"  Passed: {passed}/{len(results)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
