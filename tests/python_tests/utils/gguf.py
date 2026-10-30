# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from gguf import GGUFReader
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.constants import get_default_llm_properties
from utils.hugging_face import OVConvertedModelSchema, convert_models
from utils.network import retry_request


def reference_properties():
    return {**get_default_llm_properties(), "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}


@dataclass
class TinyGGUFModel:
    gguf_path: Path
    reference: OVConvertedModelSchema


def load_tiny_gguf_model(model_description, tmp_path_factory):
    gguf_repo = model_description.gguf_repo
    gguf_revision = model_description.gguf_revision
    filename = model_description.filename
    hf_repo = model_description.hf_repo
    hf_revision = model_description.hf_revision
    gguf_dir = retry_request(
        lambda: snapshot_download(
            gguf_repo,
            revision=gguf_revision,
            allow_patterns=[filename],
        )
    )
    hf_dir = retry_request(
        lambda: snapshot_download(
            hf_repo,
            revision=hf_revision,
            allow_patterns=["*.json", "tokenizer.model", "*.txt"],
        )
    )
    # Use the source configuration/tokenizer: HF's GGUF loader omits some configuration
    # fields (e.g. Qwen3 head_dim) and does not reconstruct all SentencePiece tokenizers.
    config = AutoConfig.from_pretrained(hf_dir).get_text_config()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    gguf_path = Path(gguf_dir) / filename
    reader = GGUFReader(gguf_path)
    template = reader.fields.get("tokenizer.chat_template")
    hf_tokenizer.chat_template = template.contents() if template is not None else None
    del reader
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_model, loading_info = AutoModelForCausalLM.from_pretrained(
        gguf_dir,
        gguf_file=filename,
        config=config,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    assert not loading_info["missing_keys"], loading_info
    assert not loading_info["unexpected_keys"], loading_info
    assert not loading_info["mismatched_keys"], loading_info
    hf_model.eval()
    hf_model.generation_config.pad_token_id = hf_tokenizer.pad_token_id
    reference_dir = tmp_path_factory.mktemp("gguf_optimum_reference")
    hf_model.save_pretrained(reference_dir)
    hf_tokenizer.save_pretrained(reference_dir)
    del hf_model
    gc.collect()

    opt_model = OVModelForCausalLM.from_pretrained(
        reference_dir,
        export=True,
        load_in_8bit=False,
        ov_config=reference_properties(),
    )
    convert_models(opt_model, hf_tokenizer, reference_dir)
    return TinyGGUFModel(
        gguf_path,
        OVConvertedModelSchema(hf_repo, opt_model, hf_tokenizer, reference_dir),
    )
