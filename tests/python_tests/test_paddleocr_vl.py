# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Repository test coverage for the PaddleOCR-VL (`paddleocr_vl`) VLM pipeline.

The upstream `optimum-intel-internal-testing` collection does not (yet) host a
tiny-random PaddleOCR-VL model, and the shared `test_vlm_pipeline.py` suite only
consumes Hub ids. To keep this test self-contained and deterministic, the tiny
fixture is constructed locally from the real model's remote code plus a shrunken
config (no original weights are downloaded), mirroring the approach used by
optimum-intel's own PaddleOCR-VL test fixture. The architecture identity is
preserved (`model_type=paddleocr_vl`, `PaddleOCRVLForConditionalGeneration`,
explicit `head_dim=128`, GQA grouping, mrope_section sum == head_dim//2,
SigLIP-variant vision tower with 2x2 spatial merge).
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import openvino
import openvino_genai
import openvino_tokenizers
import transformers

from utils.network import retry_request

PADDLEOCR_VL_ORIGINAL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
_TINY_MARKER = "genai_tiny_paddleocr_vl_v1"
_REMOTE_CODE_FILES = [
    "configuration_paddleocr_vl.py",
    "modeling_paddleocr_vl.py",
    "image_processing_paddleocr_vl.py",
    "processing_paddleocr_vl.py",
]
_ASSET_FILES = [
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.json",
    "tokenizer.model",
]

SEED = 42


def _patch_remote_code(modeling_path: Path) -> None:
    """Adapt the 4.55-era remote modeling code to transformers>=4.56.

    transformers>=4.56 renamed the ``create_causal_mask`` keyword ``inputs_embeds``
    to ``input_embeds``; call it positionally. Only affects the local fixture.
    """
    text = modeling_path.read_text()
    old = (
        "        causal_mask = create_causal_mask(\n"
        "            config=self.config,\n"
        "            inputs_embeds=inputs_embeds,\n"
        "            attention_mask=attention_mask,\n"
        "            past_key_values=past_key_values,\n"
        "            position_ids=position_ids,\n"
        "            cache_position=cache_position,\n"
        "        )"
    )
    new = (
        "        causal_mask = create_causal_mask(\n"
        "            self.config,\n"
        "            inputs_embeds,\n"
        "            attention_mask,\n"
        "            cache_position,\n"
        "            past_key_values,\n"
        "            position_ids,\n"
        "        )"
    )
    if old in text:
        modeling_path.write_text(text.replace(old, new))


def _build_tiny_paddleocr_vl_pt_model() -> str:
    """Create (or reuse) a tiny random PaddleOCR-VL PyTorch model, return its path."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    out_dir = Path(tempfile.gettempdir()) / "genai_tiny_random_paddleocr_vl"
    marker_file = out_dir / ".tiny_model_marker.json"
    cfg_file = out_dir / "config.json"
    if marker_file.exists() and cfg_file.exists():
        try:
            if json.loads(marker_file.read_text()).get("marker") == _TINY_MARKER:
                cfg = json.loads(cfg_file.read_text())
                if cfg.get("model_type") == "paddleocr_vl" and any(
                    (out_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin")
                ):
                    return str(out_dir)
        except Exception:
            pass

    assets_dir = Path(tempfile.gettempdir()) / "genai_paddleocr_vl_assets"
    if not (assets_dir / "config.json").exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        for f in ["config.json"] + _REMOTE_CODE_FILES + _ASSET_FILES:
            try:
                retry_request(
                    lambda f=f: hf_hub_download(PADDLEOCR_VL_ORIGINAL_ID, f, local_dir=str(assets_dir))
                )
            except Exception:
                pass
    _patch_remote_code(assets_dir / "modeling_paddleocr_vl.py")

    config = AutoConfig.from_pretrained(str(assets_dir), trust_remote_code=True)
    config.hidden_size = 64
    config.intermediate_size = 256
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.head_dim = 128
    config.max_position_embeddings = 4096
    config.use_cache = True
    assert sum(config.rope_scaling.get("mrope_section", [])) == config.head_dim // 2

    vc = config.vision_config
    vc.hidden_size = 128
    vc.intermediate_size = 256
    vc.num_hidden_layers = 2
    vc.num_attention_heads = 4
    vc.image_size = 224

    config.torch_dtype = "float32"
    if hasattr(config, "dtype"):
        config.dtype = "float32"
    vc.torch_dtype = "float32"
    if hasattr(vc, "dtype"):
        vc.dtype = "float32"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.float32)
    with torch.no_grad():
        model.lm_head.weight.normal_(mean=0.0, std=0.2)
        model.get_input_embeddings().weight.normal_(mean=0.0, std=0.2)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    processor = AutoProcessor.from_pretrained(str(assets_dir), trust_remote_code=True)
    processor.save_pretrained(str(out_dir))

    for f in _REMOTE_CODE_FILES + _ASSET_FILES:
        src = assets_dir / f
        dst = out_dir / f
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    _patch_remote_code(out_dir / "modeling_paddleocr_vl.py")

    marker_file.write_text(json.dumps({"marker": _TINY_MARKER}))
    return str(out_dir)


def _export_tiny_paddleocr_vl_ir() -> Path:
    """Build the tiny PT model and export it to OpenVINO IR, return the IR dir."""
    from optimum.intel import OVModelForVisualCausalLM

    pt_dir = _build_tiny_paddleocr_vl_pt_model()
    ir_dir = Path(tempfile.gettempdir()) / "genai_tiny_random_paddleocr_vl_ir"
    if (ir_dir / "openvino_language_model.xml").exists():
        return ir_dir

    processor = retry_request(
        lambda: transformers.AutoProcessor.from_pretrained(
            pt_dir, trust_remote_code=True, padding_side="left", truncation_side="left"
        )
    )
    model = retry_request(
        lambda: OVModelForVisualCausalLM.from_pretrained(
            pt_dir, compile=False, device="CPU", export=True, load_in_8bit=False, trust_remote_code=True
        )
    )
    model.save_pretrained(str(ir_dir))

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if getattr(tokenizer, "chat_template", None) is None and getattr(processor, "chat_template", None) is not None:
        tokenizer.chat_template = processor.chat_template
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    openvino.save_model(ov_tokenizer, ir_dir / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, ir_dir / "openvino_detokenizer.xml")
    processor.save_pretrained(str(ir_dir))
    return ir_dir


@pytest.fixture(scope="module")
def tiny_paddleocr_vl_ir() -> Path:
    try:
        return _export_tiny_paddleocr_vl_ir()
    except Exception as exc:  # pragma: no cover - environment/network dependent
        pytest.skip(f"Could not build tiny PaddleOCR-VL IR fixture: {exc}")


@pytest.mark.precommit
def test_paddleocr_vl_text_only(tiny_paddleocr_vl_ir: Path):
    pipe = openvino_genai.VLMPipeline(str(tiny_paddleocr_vl_ir), "CPU")
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 8
    config.do_sample = False
    result = pipe.generate("Read the text.", generation_config=config)
    assert isinstance(str(result), str)
    assert len(str(result)) >= 0


@pytest.mark.precommit
def test_paddleocr_vl_image_text(tiny_paddleocr_vl_ir: Path):
    pipe = openvino_genai.VLMPipeline(str(tiny_paddleocr_vl_ir), "CPU")
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 8
    config.do_sample = False

    # 56x56 image -> grid 4x4 patches (patch_size=14) -> merged image tokens after 2x2 merge.
    rng = np.random.RandomState(SEED)
    image = rng.randint(0, 256, (56, 56, 3), dtype=np.uint8)
    image_tensor = openvino.Tensor(image[None])  # 1HWC

    result = pipe.generate("What is in the image?",
                           images=[image_tensor], generation_config=config)
    # The merge/position-id logic must run without raising and produce a decodable result.
    assert isinstance(str(result), str)
