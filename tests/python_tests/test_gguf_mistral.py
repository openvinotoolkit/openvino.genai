# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import sys
import pytest

import gguf

import openvino_genai as ov_genai

from utils.hugging_face import download_gguf_model
from utils.ov_genai_pipelines import (
    create_ov_pipeline,
    GGUF_PIPELINE_TYPES,
    PipelineType,
)

# A small, llama-architecture GGUF already used by the GGUF suite. Mistral is
# llama-family: identical tensor layout and NORM-type (interleaved) q/k RoPE
# weights. The mistral / mistral3 support added in this PR routes those arch
# names through the same `create_language_model` path and applies the same q/k
# de-interleave ("reorder") that llama already uses. Relabeling this model's
# architecture to mistral / mistral3 must therefore produce *identical* output
# to the llama load. This validates the fix (arch whitelist + reorder) without
# needing a real mistral3 model (Devstral is ~14 GB) and without an HF
# reference (transformers' GGUF loader does not support the mistral3 arch).
LLAMA_GGUF = {
    "gguf_model_id": "prithivMLmods/SmolLM2-135M-GGUF",
    "gguf_filename": "SmolLM2-135M.F16.gguf",
}


def _relabel_gguf_architecture(src_path: str, dst_path: str, new_arch: str) -> None:
    """Write a copy of ``src_path`` to ``dst_path`` with ``general.architecture``
    (and every ``<old_arch>.*`` hyperparameter key) rewritten to ``new_arch``.

    Tensors, tokenizer metadata and all other keys are copied verbatim, so the
    only behavioural difference is the architecture the GGUF reader dispatches
    on. Faithful field/type copying follows llama.cpp's ``gguf_new_metadata``.
    """
    reader = gguf.GGUFReader(src_path)
    old_arch = reader.get_field("general.architecture").contents()

    writer = gguf.GGUFWriter(dst_path, arch=new_arch)  # writes general.architecture
    try:
        for field in reader.fields.values():
            # general.architecture is written by the writer; GGUF.* are virtual.
            if field.name == "general.architecture" or field.name.startswith("GGUF."):
                continue

            name = field.name
            if name.startswith(old_arch + "."):
                name = new_arch + name[len(old_arch):]

            value_type = field.types[0]
            sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(name, field.contents(), value_type, sub_type=sub_type)

        for tensor in reader.tensors:
            writer.add_tensor_info(
                tensor.name,
                tensor.data.shape,
                tensor.data.dtype,
                tensor.data.nbytes,
                tensor.tensor_type,
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in reader.tensors:
            writer.write_tensor_data(tensor.data)
    finally:
        writer.close()


def _generate(models_path, pipeline_type: PipelineType, prompt: str) -> str:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 20
    config.apply_chat_template = False  # deterministic, no chat template

    pipe = create_ov_pipeline(models_path, pipeline_type=pipeline_type)
    try:
        result = pipe.generate(prompt, config)
    finally:
        del pipe
        gc.collect()
    # LLMPipeline.generate(str, ...) returns DecodedResults; str() yields the text.
    return str(result)


@pytest.mark.parametrize("pipeline_type", GGUF_PIPELINE_TYPES)
@pytest.mark.parametrize("relabel_arch", ["mistral", "mistral3"])
@pytest.mark.skipif(sys.platform == "win32", reason="CVS-174065")
def test_mistral_arch_matches_llama_reference(pipeline_type, relabel_arch, tmp_path):
    """mistral / mistral3 arch must be served identically to the equivalent
    llama-family model (arch whitelist + NORM-type q/k reorder)."""
    if sys.platform == "darwin":
        pytest.skip(reason="168882: Sporadic segmentation fault failure on MacOS.")

    prompt = "The capital of France is"

    llama_gguf = download_gguf_model(
        LLAMA_GGUF["gguf_model_id"], LLAMA_GGUF["gguf_filename"]
    )
    reference = _generate(llama_gguf, pipeline_type, prompt)

    relabeled_gguf = tmp_path / f"smollm2-135m-{relabel_arch}.gguf"
    _relabel_gguf_architecture(str(llama_gguf), str(relabeled_gguf), relabel_arch)

    actual = _generate(str(relabeled_gguf), pipeline_type, prompt)

    assert actual == reference, (
        f"{relabel_arch} output diverged from the llama reference; "
        f"the mistral arch path likely mis-handles NORM-type q/k weights."
    )
