# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
from pathlib import Path
import openvino_genai as ov_genai
from utils.hugging_face import download_gguf_model

# Constants
GGUF_MODEL_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
GGUF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
ADAPTER_REPO_ID = "unclecode/tinyllama-function-call-lora-adapter-250424"
ADAPTER_FILENAME = "tinyllama-function-call-lora-adapter-250424-f16.gguf"

@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.skipif(sys.platform == "darwin", reason="Sporadic instability on Mac")
def test_gguf_lora_generation():
    # 1. Download Model and Adapter
    model_path = download_gguf_model(GGUF_MODEL_ID, GGUF_FILENAME)
    adapter_path = download_gguf_model(ADAPTER_REPO_ID, ADAPTER_FILENAME)

    device = "CPU"
    
    # 2. Baseline Generation (No Adapter)
    pipe = ov_genai.LLMPipeline(model_path, device)
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 30
    config.do_sample = False
    
    prompt = "The quick brown fox jumps over the lazy dog."
    
    res_base = pipe.generate(prompt, config)
    
    del pipe 
    
    # 3. Generation with GGUF Adapter
    adapter = ov_genai.Adapter(adapter_path)
    adapter_config = ov_genai.AdapterConfig(adapter)
    
    # Initialize pipeline with adapter registered
    pipe_lora = ov_genai.LLMPipeline(model_path, device, adapters=adapter_config)
    
    # Activate adapter with alpha=1.0
    active_adapter = ov_genai.AdapterConfig(adapter, 1.0)
    res_lora = pipe_lora.generate(prompt, config, adapters=active_adapter)
    
    print(f"Base: {res_base}")
    print(f"LoRA: {res_lora}")
    
    # 4. Verify Difference
    assert res_base != res_lora
