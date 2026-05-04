# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tiny random model generator for gemma4-moe.
"""

from pathlib import Path

from utils.tiny_model_factory import register_generator


@register_generator("optimum-intel-internal-testing/tiny-random-gemma4-moe")
def generate_gemma4_moe(output_dir: Path) -> None:
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Gemma4ForConditionalGeneration

    config = AutoConfig.from_pretrained("google/gemma-4-26B-A4B-it")

    # Text config
    config.text_config.global_head_dim = 4
    config.text_config.head_dim = 4
    config.text_config.hidden_size = 32
    config.text_config.hidden_size_per_layer_input = 0
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["sliding_attention", "full_attention"]
    config.text_config.num_kv_shared_layers = 0
    config.text_config.intermediate_size = 64
    config.text_config.dtype = "float32"

    # MOE parameters scaled down to avoid CPU plugin crash on SPR
    config.text_config.num_experts = 4
    config.text_config.top_k_experts = 2
    config.text_config.moe_intermediate_size = 64
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.text_config.num_global_key_value_heads = 2

    # Vision config
    config.vision_config.head_dim = 4
    config.vision_config.hidden_size = 8
    config.vision_config.intermediate_size = 32
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_key_value_heads = 2

    model = Gemma4ForConditionalGeneration(config)
    model.eval()
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
    tokenizer.save_pretrained(str(output_dir))

    processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
    processor.save_pretrained(str(output_dir))
