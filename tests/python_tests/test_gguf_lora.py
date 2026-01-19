# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import gc
import openvino_genai as ov_genai
from utils.hugging_face import download_gguf_model

# Constants
GGUF_MODEL_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
GGUF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
ADAPTER_REPO_ID = "unclecode/tinyllama-function-call-lora-adapter-250424"
ADAPTER_FILENAME = "tinyllama-function-call-lora-adapter-250424-f16.gguf"


def _extract_generated_text(result) -> str:
    """
    Extract generated text from pipeline result.
    Handles both string returns and DecodedResults objects.
    """
    if isinstance(result, str):
        return result
    if hasattr(result, "texts") and result.texts:
        return result.texts[0]
    if hasattr(result, "text"):
        return result.text
    # Fallback to string representation
    return str(result)


@pytest.mark.nightly
@pytest.mark.skipif(sys.platform == "darwin", reason="Sporadic instability on Mac")
def test_gguf_lora_generation():
    """
    Test that GGUF LoRA adapters can be loaded and influence generation.
    
    This test:
    1. Downloads a GGUF model and compatible GGUF LoRA adapter
    2. Generates text without the adapter (baseline)
    3. Generates text with the adapter applied
    4. Verifies that the adapter changes the output
    """
    # Download model and adapter (uses HF cache, won't redownload)
    model_path = download_gguf_model(GGUF_MODEL_ID, GGUF_FILENAME)
    adapter_path = download_gguf_model(ADAPTER_REPO_ID, ADAPTER_FILENAME)

    device = "CPU"
    
    # Deterministic generation config
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 30
    config.do_sample = False  # Deterministic
    
    prompt = "The quick brown fox jumps over the lazy dog."
    
    # Baseline: Generate without adapter
    pipe_base = ov_genai.LLMPipeline(model_path, device)
    res_base = pipe_base.generate(prompt, config)
    base_text = _extract_generated_text(res_base)
    
    del pipe_base
    gc.collect()
    
    # With adapter: Initialize pipeline with adapter registered
    adapter = ov_genai.Adapter(adapter_path)
    adapter_config = ov_genai.AdapterConfig(adapter)
    
    pipe_lora = ov_genai.LLMPipeline(model_path, device, adapters=adapter_config)
    
    # Generate with adapter active (alpha=1.0)
    active_adapter = ov_genai.AdapterConfig(adapter, alpha=1.0)
    res_lora = pipe_lora.generate(prompt, config, adapters=active_adapter)
    lora_text = _extract_generated_text(res_lora)
    
    del pipe_lora
    gc.collect()
    
    # Verify: Adapter should change the output
    assert base_text != lora_text, (
        f"LoRA adapter did not change generation output.\n"
        f"Base: {base_text}\n"
        f"LoRA: {lora_text}"
    )
