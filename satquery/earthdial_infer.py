"""
EarthDial Inference Script — runs in isolated earthdial_test_venv.
Called via subprocess from the main SatQuery pipeline.

Usage:
    python -m satquery.earthdial_infer <image_path> <prompt> \
        [--max_tokens 200] [--num_beams 5] [--adapter <dir|"">]

Adapter support (VRSBench QLoRA, default ON when the artifact is present)
-------------------------------------------------------------------------
EarthDial here is the QLoRA domain-adapted model (fine-tuned on VRSBench
RS-VQA with fused Phi-3 LoRA targets). The adapter is loaded onto the SAME
4-bit NF4 base used for training/validation:

  1. explicit  --adapter <dir>          ("" forces the base model)
  2. env       SATQUERY_EARTHDIAL_ADAPTER  ("" forces the base model)
  3. default   changemodel_test/qlora_smoke/runs/adapt_exp/ckpt-499

Resolution order is 1 > 2 > 3. If the resolved adapter directory is missing
or invalid the script logs a warning and falls back to the clean pretrained
base model — it never crashes because an adapter is unavailable.

Loading mirrors the validated recipe (see changemodel_test/qlora_smoke):
4-bit NF4 whole model (compute bf16), the accelerate-dispatch workaround for
transformers 4.37 + accelerate 1.14, then PeftModel.from_pretrained on the
language model (fused self_attn.qkv_proj/o_proj, mlp.gate_up_proj/down_proj).
If bitsandbytes is not importable the original bf16 device_map="auto" base
path is used instead (no adapter).

Output: JSON with answer, adapter_used, precision, timing, success status.
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")


# Default adapted-checkpoint artifact (kept OUTSIDE the model checkpoints dir;
# a separate ~100 MB file set — not copied into the repo).
def _default_adapter_dir():
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, "changemodel_test", "qlora_smoke",
                        "runs", "adapt_exp", "ckpt-499")


def _report(msg):
    print(f"[earthdial_infer] {msg}", file=sys.stderr, flush=True)


def resolve_adapter_dir(explicit):
    """Return adapter dir to use or None (base model).

    explicit is the CLI value (None = not given, "" = force base).
    Falls back to the SATQUERY_EARTHDIAL_ADAPTER env var, then to the
    default adapted artifact if it exists.
    """
    if explicit is not None:
        val = explicit
    elif os.environ.get("SATQUERY_EARTHDIAL_ADAPTER") is not None:
        val = os.environ["SATQUERY_EARTHDIAL_ADAPTER"]
    else:
        val = _default_adapter_dir()
    val = (val or "").strip()

    if not val:
        _report("adapter disabled by config -> using pretrained base model")
        return None
    if not os.path.isdir(val):
        _report(f"adapter dir not found ({val}) -> using pretrained base model")
        return None
    has_cfg = os.path.isfile(os.path.join(val, "adapter_config.json"))
    has_wts = any(
        f.startswith("adapter_model") and f.endswith((".bin", ".safetensors"))
        for f in os.listdir(val))
    if not (has_cfg and has_wts):
        _report(f"adapter dir invalid ({val}) -> using pretrained base model")
        return None
    _report(f"using LoRA adapter: {val}")
    return val


def _ensure_on_cuda(model):
    first_dev = next(model.parameters()).device
    if str(first_dev) != "cuda:0":
        _report("moving submodules to GPU0")
        model.mlp1.cuda()
        model.vision_model.cuda()
        model.language_model.cuda()
    else:
        _report("model already on GPU0")
    return model


def load_model_4bit(model_dir, report):
    """Validated QLoRA-era loader: 4-bit NF4 base on GPU0 (peak ~3.1 GiB)."""
    import torch
    import transformers.modeling_utils as _tmu
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from earthdial.model.internvl_chat import InternVLChatModel

    # accelerate 1.14's dispatch_model calls .to() on the model, which
    # transformers 4.37 forbids for quantized models. Weights already land on
    # GPU0 during load (set_module_tensor_to_device), so the dispatch step is
    # a no-op here — same workaround as the validated QLoRA smoke/train path.
    _tmu.dispatch_model = lambda *a, **k: None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = InternVLChatModel.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=None,
        trust_remote_code=True,
    )
    model = _ensure_on_cuda(model)
    report(f"4-bit NF4 model loaded in {time.time()-t0:.1f}s")
    return model


def load_model_bf16(model_dir, report):
    """Original fallback loader (no bitsandbytes): bf16 + device_map auto."""
    import torch
    from transformers import AutoTokenizer
    from earthdial.model.internvl_chat import InternVLChatModel

    t0 = time.time()
    model = InternVLChatModel.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto",
    )
    report(f"bf16 (device_map=auto) model loaded in {time.time()-t0:.1f}s")
    return model


def attach_adapter(model, adapter_dir):
    """Wrap the language model with the saved LoRA adapter (inference mode)."""
    from peft import PeftModel

    llm = PeftModel.from_pretrained(
        model.language_model, adapter_dir, is_trainable=False)
    model.language_model = llm
    return model


def run_earthdial_inference(
    image_path: str,
    prompt: str,
    max_tokens: int = 200,
    num_beams: int = 5,
    adapter: str | None = None,
) -> dict:
    """Run EarthDial inference on a single image. Returns structured result."""
    try:
        import torch
        from PIL import Image
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}"}

    # Find EarthDial source
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    earthdial_src = os.path.join(project_root, "EarthDial", "src")
    if not os.path.isdir(earthdial_src):
        return {"success": False, "error": f"EarthDial source not found: {earthdial_src}"}

    if earthdial_src not in sys.path:
        sys.path.insert(0, earthdial_src)

    # Find checkpoint
    model_dir = os.path.join(project_root, "checkpoints", "EarthDial_4B_RGB")
    if not os.path.isdir(model_dir):
        return {"success": False, "error": f"EarthDial checkpoint not found: {model_dir}"}

    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    t_total = time.time()
    report = _report

    try:
        from transformers import AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, use_fast=False)

        # ── base model: 4-bit NF4 (validated) with bf16 fallback ──────────
        precision = "4bit"
        try:
            import bitsandbytes  # noqa: F401
            model = load_model_4bit(model_dir, report)
        except ImportError:
            report("bitsandbytes unavailable -> bf16 base (no adapter)")
            precision = "bf16"
            model = load_model_bf16(model_dir, report)
        except Exception as e:  # quantized load failure: fall back to bf16
            report(f"4-bit load failed ({type(e).__name__}: {e}) -> bf16 fallback")
            precision = "bf16"
            model = load_model_bf16(model_dir, report)

        model = model.eval()

        # ── adapter (default: VRSBench QLoRA; robust base fallback) ────────
        adapter_dir = resolve_adapter_dir(explicit=adapter)
        if adapter_dir is not None:
            try:
                model = attach_adapter(model, adapter_dir)
                report("adapter attached")
            except Exception as e:
                report(f"adapter attach FAILED ({type(e).__name__}: {e}) "
                       f"-> continuing with pretrained base model")
                adapter_dir = None
        else:
            report("running pretrained base model (no adapter)")

        # Defensive: chat()/generate() expect the <IMG_CONTEXT> token id set.
        if model.img_context_token_id is None:
            model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")

        image_size = (model.config.force_image_size
                      or model.config.vision_config.image_size)
        from earthdial.train.dataset import build_transform
        transform = build_transform(
            is_train=False, input_size=image_size, normalize_type="imagenet")
        load_ms = (time.time() - t0) * 1000

        vram_used = (torch.cuda.memory_allocated(0) / 1024**2
                     if torch.cuda.is_available() else 0)

        # Preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).cuda().to(torch.bfloat16)

        # Inference
        gen_cfg = {
            "num_beams": num_beams,
            "max_new_tokens": max_tokens,
            "min_new_tokens": 1,
            "do_sample": False,
        }

        t0 = time.time()
        answer = model.chat(tok, pixel_values, prompt, gen_cfg, verbose=False)
        inference_ms = (time.time() - t0) * 1000

        # Cleanup
        del model, tok, transform
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_ms = (time.time() - t_total) * 1000

        return {
            "success": True,
            "answer": answer,
            "load_ms": round(load_ms, 0),
            "inference_ms": round(inference_ms, 0),
            "total_ms": round(total_ms, 0),
            "vram_used_mb": round(vram_used, 0),
            "model_dir": model_dir,
            "adapter_used": adapter_dir,
            "precision": precision,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "total_ms": round((time.time() - t_total) * 1000, 0),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to satellite image")
    parser.add_argument("prompt", help="Prompt to send to EarthDial")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument(
        "--adapter", default=None,
        help="LoRA adapter dir (default: env SATQUERY_EARTHDIAL_ADAPTER, then "
             "the standard VRSBench-adapted checkpoint). Empty string forces "
             "the pretrained base model.")
    args = parser.parse_args()

    result = run_earthdial_inference(
        image_path=args.image_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        num_beams=args.num_beams,
        adapter=args.adapter,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
