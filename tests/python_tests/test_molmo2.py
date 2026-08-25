# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Molmo2 (`allenai/MolmoWeb-4B`) VLMPipeline support.

Molmo2 export is not yet available in the released optimum-intel -- see
https://github.com/huggingface/optimum-intel/pull/1812 (open, unmerged at the
time this test was added). The whole module is skipped unless the installed
optimum-intel already recognizes "molmo2" as an exportable model type, so
this file is a no-op in CI today and becomes a real regression test the
moment that PR (or an equivalent) lands.

No HF-hosted tiny-random fixture exists yet for Molmo2 (this environment has
no HF Hub write access to publish one under optimum-intel-internal-testing,
where other tiny-random-* fixtures used by this test suite live), so this
test builds a tiny checkpoint on the fly instead: it downloads only the small
trust_remote_code/tokenizer/config files from the real `allenai/MolmoWeb-4B`
repo (no weights, a few hundred KB total), shrinks the config down to ~20M
params, and instantiates random weights -- matching the intent of the
pre-hosted tiny-random-* fixtures used elsewhere in this suite.

This covers pipeline wiring (config/inputs_embedder/vision_encoder dispatch
for VLMModelType::MOLMO2) with generate() smoke checks. It intentionally does
not assert exact accuracy against a PyTorch reference -- see this model's
accuracy report and enablement notebook (published under enabled-models/) for
the full manual accuracy validation against the real checkpoint.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from PIL import Image

MODEL_ID = "allenai/MolmoWeb-4B"

# 28x28 images with patch_size=14 -> a 2x2 patch grid (4 positions, a perfect
# square as required by Molmo2's positional-embedding reshape logic).
TINY_IMAGE_SIZE = 28
TINY_IMAGE_NUM_POS = 4

# Small trust_remote_code/tokenizer/config files copied verbatim from the real
# model snapshot. No weights are downloaded.
FILES_TO_COPY = [
    "config.json",
    "configuration_molmo2.py",
    "modeling_molmo2.py",
    "processing_molmo2.py",
    "image_processing_molmo2.py",
    "video_processing_molmo2.py",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
]


def _molmo2_export_supported() -> bool:
    try:
        from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING
    except ImportError:
        return False
    return "molmo2" in MODEL_TYPE_TO_CLS_MAPPING


pytestmark = pytest.mark.skipif(
    not _molmo2_export_supported(),
    reason=(
        "optimum-intel installed in this environment does not recognize 'molmo2' as an "
        "exportable model type -- requires https://github.com/huggingface/optimum-intel/pull/1812 "
        "(open, unmerged at the time this test was added)."
    ),
)


def _build_shrunk_config(src_config: dict) -> dict:
    cfg = json.loads(json.dumps(src_config))  # deep copy

    cfg["text_config"].update({
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 512,
    })

    cfg["vit_config"].update({
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "image_default_input_size": [TINY_IMAGE_SIZE, TINY_IMAGE_SIZE],
        "image_num_pos": TINY_IMAGE_NUM_POS,
    })

    # adapter_config's vit_layers indexes into the vision tower's hidden_states
    # list (embeddings + num_hidden_layers entries). With vit num_hidden_layers=2
    # that list has 3 entries, so only -1/-2/-3 are valid indices.
    cfg["adapter_config"].update({
        "hidden_size": 32,
        "text_hidden_size": 64,
        "head_dim": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 64,
        "vit_layers": [-1, -2],
    })
    return cfg


def _apply_rope_compat_shim() -> None:
    """Molmo2RotaryEmbedding.__init__ eagerly does
    ROPE_INIT_FUNCTIONS[self.rope_type] at construction time. Transformers
    versions newer than what the model card was tested against dropped the
    "default" key from ROPE_INIT_FUNCTIONS in favor of a class-based
    rope_parameters API, raising KeyError('default') the moment any Molmo2
    model (tiny or real) is instantiated. Re-registering the legacy default
    entry is a compatible, additive no-op on transformers versions that still
    have it.
    """
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _compute_default_rope_parameters_compat(config=None, device=None, seq_len=None, layer_type=None, **_kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0  # attention_factor=1.0, i.e. no post-scaling

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters_compat


@pytest.fixture(scope="module")
def tiny_molmo2_ov_model(tmp_path_factory) -> Path:
    """Build a tiny random-weight Molmo2 checkpoint and export it to OpenVINO IR.

    Equivalent in spirit to the "tiny-random-*" HF-hosted fixtures used by
    other tests in this suite, but built locally (see module docstring).
    """
    from huggingface_hub import hf_hub_download
    from optimum.commands.optimum_cli import main as optimum_cli_main

    _apply_rope_compat_shim()

    tmp_path = tmp_path_factory.mktemp("molmo2")
    src_dir = tmp_path / "source"
    ov_dir = tmp_path / "ov"
    src_dir.mkdir()

    for fname in FILES_TO_COPY:
        local_path = hf_hub_download(MODEL_ID, fname)
        shutil.copy(local_path, src_dir / fname)

    with open(src_dir / "config.json") as f:
        src_config = json.load(f)
    with open(src_dir / "config.json", "w") as f:
        json.dump(_build_shrunk_config(src_config), f, indent=2)

    with open(src_dir / "preprocessor_config.json") as f:
        pp = json.load(f)
    pp["size"] = {"height": TINY_IMAGE_SIZE, "width": TINY_IMAGE_SIZE}
    pp["max_crops"] = 1
    with open(src_dir / "preprocessor_config.json", "w") as f:
        json.dump(pp, f, indent=2)

    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    torch.manual_seed(0)
    tiny_config = AutoConfig.from_pretrained(str(src_dir), trust_remote_code=True)
    tiny_model = AutoModelForImageTextToText.from_config(tiny_config, trust_remote_code=True, dtype=torch.float32)
    tiny_model.save_pretrained(str(src_dir))

    # Invoke `optimum-cli export openvino` in-process: main_export() has no
    # weight_format parameter of its own (see export_model.py in this model's
    # enablement branch for the full writeup) -- the CLI command builds the
    # OVConfig/quantization_config correctly, main_export() alone does not.
    old_argv = sys.argv
    try:
        sys.argv = [
            "optimum-cli", "export", "openvino",
            "--model", str(src_dir),
            "--task", "image-text-to-text",
            "--framework", "pt",
            "--trust-remote-code",
            "--weight-format", "fp16",
            str(ov_dir),
        ]
        optimum_cli_main()
    finally:
        sys.argv = old_argv

    shutil.copy(src_dir / "preprocessor_config.json", ov_dir / "preprocessor_config.json")
    shutil.copy(src_dir / "video_preprocessor_config.json", ov_dir / "video_preprocessor_config.json")

    subprocess.run(  # nosec B603, B607
        ["convert_tokenizer", str(src_dir), "--with-detokenizer", "--trust-remote-code", "-o", str(ov_dir)],
        check=True,
    )
    return ov_dir


def test_molmo2_vlm_pipeline_image_and_text(tiny_molmo2_ov_model: Path):
    """VLMPipeline.generate() with an image + text prompt produces non-empty output."""
    import openvino_genai as ov_genai

    pipe = ov_genai.VLMPipeline(str(tiny_molmo2_ov_model), "CPU")
    image_arr = (np.random.default_rng(0).random((TINY_IMAGE_SIZE, TINY_IMAGE_SIZE, 3)) * 255).astype(np.uint8)
    image_tensor = ov.Tensor(np.array(Image.fromarray(image_arr))[None, ...])

    result = pipe.generate("Describe this image.", images=[image_tensor], max_new_tokens=10)
    assert len(str(result)) > 0


def test_molmo2_vlm_pipeline_text_only(tiny_molmo2_ov_model: Path):
    """VLMPipeline.generate() also supports text-only prompts (no image input)."""
    import openvino_genai as ov_genai

    pipe = ov_genai.VLMPipeline(str(tiny_molmo2_ov_model), "CPU")
    result = pipe.generate("Hello, how are you?", max_new_tokens=10)
    assert len(str(result)) > 0


def test_molmo2_vlm_pipeline_deterministic(tiny_molmo2_ov_model: Path):
    """Two independently constructed pipelines produce identical greedy output."""
    import openvino_genai as ov_genai

    def _run() -> str:
        pipe = ov_genai.VLMPipeline(str(tiny_molmo2_ov_model), "CPU")
        return str(pipe.generate("Hello, how are you?", max_new_tokens=10))

    assert _run() == _run()
