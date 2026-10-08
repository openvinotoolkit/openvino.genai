# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from whowhatbench import wwb
from whowhatbench.ltx2_gguf import convert_ltx2_state_dict, resolve_gguf_file

# Original LTX-2.3 tensor names and their diffusers names, as diffusers' scripts/convert_ltx2_to_diffusers.py maps them
TRANSFORMER_NAMES = {
    "patchify_proj.weight": "proj_in.weight",
    "audio_patchify_proj.bias": "audio_proj_in.bias",
    "adaln_single.emb.timestep_embedder.linear_1.weight": "time_embed.emb.timestep_embedder.linear_1.weight",
    "audio_adaln_single.linear.weight": "audio_time_embed.linear.weight",
    "prompt_adaln_single.linear.weight": "prompt_adaln.linear.weight",
    "audio_prompt_adaln_single.linear.bias": "audio_prompt_adaln.linear.bias",
    "av_ca_video_scale_shift_adaln_single.linear.weight": "av_cross_attn_video_scale_shift.linear.weight",
    "av_ca_audio_scale_shift_adaln_single.linear.weight": "av_cross_attn_audio_scale_shift.linear.weight",
    "av_ca_a2v_gate_adaln_single.linear.weight": "av_cross_attn_video_a2v_gate.linear.weight",
    "av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_2.bias": (
        "av_cross_attn_audio_v2a_gate.emb.timestep_embedder.linear_2.bias"
    ),
    "scale_shift_table": "scale_shift_table",
    "transformer_blocks.3.prompt_scale_shift_table": "transformer_blocks.3.prompt_scale_shift_table",
    "transformer_blocks.3.scale_shift_table_a2v_ca_video": (
        "transformer_blocks.3.video_a2v_cross_attn_scale_shift_table"
    ),
    "transformer_blocks.3.scale_shift_table_a2v_ca_audio": (
        "transformer_blocks.3.audio_a2v_cross_attn_scale_shift_table"
    ),
    "transformer_blocks.3.attn1.q_norm.weight": "transformer_blocks.3.attn1.norm_q.weight",
    "transformer_blocks.3.audio_to_video_attn.k_norm.weight": "transformer_blocks.3.audio_to_video_attn.norm_k.weight",
    "transformer_blocks.3.attn2.to_gate_logits.weight": "transformer_blocks.3.attn2.to_gate_logits.weight",
    "transformer_blocks.3.audio_ff.net.0.proj.weight": "transformer_blocks.3.audio_ff.net.0.proj.weight",
    "proj_out.bias": "proj_out.bias",
}
CONNECTOR_NAMES = {
    "video_embeddings_connector.learnable_registers": "video_connector.learnable_registers",
    "video_embeddings_connector.transformer_1d_blocks.0.attn1.q_norm.weight": (
        "video_connector.transformer_blocks.0.attn1.norm_q.weight"
    ),
    "audio_embeddings_connector.transformer_1d_blocks.2.attn1.k_norm.weight": (
        "audio_connector.transformer_blocks.2.attn1.norm_k.weight"
    ),
    "audio_embeddings_connector.transformer_1d_blocks.7.ff.net.2.weight": (
        "audio_connector.transformer_blocks.7.ff.net.2.weight"
    ),
    "text_embedding_projection.video_aggregate_embed.weight": "video_text_proj_in.weight",
    "text_embedding_projection.audio_aggregate_embed.bias": "audio_text_proj_in.bias",
    "text_embedding_projection.aggregate_embed.weight": "text_proj_in.weight",  # LTX-2.0
}


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
def test_convert_ltx2_state_dict(prefix):
    state_dict = {prefix + name: name for name in {**TRANSFORMER_NAMES, **CONNECTOR_NAMES}}
    transformer, connectors = convert_ltx2_state_dict(state_dict)
    assert transformer == {new: old for old, new in TRANSFORMER_NAMES.items()}
    assert connectors == {new: old for old, new in CONNECTOR_NAMES.items()}


def test_resolve_gguf_file(monkeypatch, tmp_path):
    local_file = tmp_path / "model.gguf"
    local_file.touch()
    assert resolve_gguf_file(str(local_file)) == str(local_file)

    downloads = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: downloads.append(kwargs) or "cached.gguf")
    assert resolve_gguf_file("unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-Q8_0.gguf") == "cached.gguf"
    assert downloads == [{"repo_id": "unsloth/LTX-2.3-GGUF", "filename": "ltx-2.3-22b-dev-Q8_0.gguf"}]

    for missing_file in ("missing.gguf", str(tmp_path / "models" / "missing.gguf"), "./models/ltx/missing.gguf"):
        with pytest.raises(FileNotFoundError, match="Pass a local path or <org>/<repo>/<file>.gguf"):
            resolve_gguf_file(missing_file)
    assert len(downloads) == 1


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--target-model", "model"], "requires --hf"),
        (["--base-model", "model", "--target-model", "model", "--hf"], "both --base-model and --target-model"),
    ],
)
def test_text2video_gguf_file_args(monkeypatch, args, message):
    argv = ["wwb", "--gt-data", "gt.csv", "--model-type", "text-to-video", "--gguf-file", "model.gguf", *args]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match=message):
        wwb.check_args(wwb.parse_args())
