# 🎤 SatQuery AI — Judge Q&A

Likely evaluator questions and technically honest answers.

---

## Q1: What is your technical contribution?

**A**: Our contribution is **system-level integration**, not a new model. We built:
- A deterministic intent router that classifies natural-language RS queries into 7 categories in <1ms
- An orchestration pipeline that routes to the right model (EarthDial for optical, YOLOv8 for SAR)
- A visualization engine that draws bounding boxes and confidence scores on images
- A unified Gradio UI that makes multiple pretrained models feel like one application
- An isolated subprocess architecture that lets incompatible model environments coexist on one machine

The insight is that deploying pretrained models effectively — with proper routing, visualization, and error handling — is itself valuable engineering.

---

## Q2: Did you train any of the models?

**A**: No. All models are pretrained:
- **EarthDial 4B**: Trained by MBZUAI/EarthDial team, published at CVPR 2025, Apache 2.0 license
- **YOLOv8 SAR**: Fine-tuned by a third party on SAR vessel detection data, hosted on HuggingFace

We did not train, fine-tune, or modify any model weights. Our work is entirely at the system integration level.

---

## Q3: Why not use SARDet-100K?

**A**: We investigated SARDet-100K extensively. It failed on our platform for three reasons:
1. Its dependency stack (mmcv/mmdet) requires C++ compilation that fails on Windows + Python 3.12
2. Its pretrained weights require Kaggle authentication or Baidu Disk access, which we couldn't automate
3. Even if installed, SARDet-100K is an **object detector with 6 fixed categories** (ship, aircraft, car, bridge, harbour, tank) — not a VLM that can answer natural-language questions about SAR imagery

We replaced it with YOLOv8 SAR vessel detection, which installs cleanly, runs reliably, and provides actual useful output (ship detection with bounding boxes).

---

## Q4: Can your system handle SAR scene understanding?

**A**: Not yet. Our SAR capability is limited to **vessel detection** — identifying ships in SAR imagery with bounding boxes and confidence scores. We cannot:
- Describe what a SAR scene shows in natural language
- Classify terrain types from SAR imagery
- Answer questions like "What is in this SAR image?"
- Perform general SAR VQA

This is documented honestly in the UI and README. General SAR understanding would require a SAR-specific VLM, which is an active research area.

---

## Q5: What about change detection?

**A**: Change detection is listed as future work. We investigated lightweight approaches (pixel differencing) and found them unreliable for real remote sensing imagery — they require same sensor, same season, same time of day, and precise sub-pixel registration. Without these, pixel differences are noise, not semantic changes. Real change detection requires a trained bi-temporal model (like ChangeChat or DeltaVLM), which we don't have.

---

## Q6: How does the intent router work?

**A**: It's a deterministic keyword-based classifier. We maintain 7 intent categories, each with a list of trigger keywords. The first matching intent wins. This gives us:
- <1ms latency (no model inference needed)
- Zero VRAM usage
- Deterministic, explainable behavior
- No false positives from ML uncertainty

For a production system, you'd replace this with an LLM-based planner. For a hackathon prototype, deterministic routing is more reliable.

---

## Q7: Why use a subprocess for SAR instead of integrating directly?

**A**: The EarthDial environment and the YOLOv8 environment have incompatible Python dependency trees. EarthDial requires specific versions of transformers, peft, and sentencepiece. YOLOv8 requires ultralytics. Installing both in the same environment would break EarthDial. The subprocess approach:
- Keeps each environment isolated and working
- Adds ~2s overhead (acceptable for a demo)
- Prevents VRAM conflicts (only one model loads at a time)
- Is the standard approach for multi-model systems with incompatible dependencies

---

## Q8: Can this run on a machine without a GPU?

**A**: Partially. The keyword router and visualization engine work on CPU. EarthDial can run on CPU but inference would be extremely slow (minutes per query instead of seconds). YOLOv8 SAR works on CPU in ~66ms. For a practical demo, a GPU is strongly recommended.

---

## Q9: What are the VRAM requirements?

**A**: 
- EarthDial 4B (4-bit quantized): ~2.85 GB
- YOLOv8 SAR: ~21 MB
- Total: ~2.9 GB
- Our GPU: RTX 3050 Laptop (4 GB VRAM)
- Headroom: ~1.2 GB

Both models can coexist in VRAM if needed, but we use subprocess isolation so only one loads at a time.

---

## Q10: How does this compare to existing systems like GeoChat or Earth-Agent?

**A**: Our system is simpler but more practical:
- **GeoChat** is a research model that handles RS VQA but requires 7B+ VRAM
- **Earth-Agent** is an agentic framework but requires cloud infrastructure
- We built a working prototype that runs on a laptop, combines optical + SAR, and produces visual evidence

Our advantage is practical deployability — not theoretical capability. A working demo on a laptop is more useful for ISRO than a research paper that requires cloud GPUs.

---

## Q11: What would you improve with more time?

**A**:
1. **Change detection**: Integrate a trained bi-temporal model
2. **SAR VQA**: Use a SAR-specific VLM for natural-language SAR analysis
3. **Multi-image support**: Compare images side-by-side
4. **LLM router**: Replace keyword matching with an LLM-based intent classifier
5. **Offline deployment**: Optimize for edge devices (RISC-V, mobile)

---

## Q12: Is this suitable for commercial deployment?

**A**: As a prototype, no. Limitations:
- Inference is slow (50-260s per query)
- SAR is ship-only
- All models have non-commercial licenses (Apache 2.0 for EarthDial, CC BY-NC for some weights)
- No error recovery for adversarial inputs

For commercial deployment, you'd need: faster inference (model distillation), broader SAR capabilities, commercial licensing, and rigorous testing.
