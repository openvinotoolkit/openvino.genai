# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess  # nosec B404
from pathlib import Path

import pytest
import torch

from .atomic_download import AtomicDownloadManager
from .network import retry_request

logger = logging.getLogger(__name__)

BAGEL_TINY_MODEL_SEED = 0
BAGEL_EXPORT_TIMEOUT_SECONDS = 600
# Qwen2 tokenizer with the <|im_start|>, <|vision_start|> and <|image_pad|> special tokens BAGEL uses
BAGEL_TOKENIZER_ID = "optimum-intel-internal-testing/tiny-random-qwen2vl"


def generate_tiny_bagel_model(output_dir: Path) -> Path:
    """Generate a tiny random BAGEL checkpoint with its processor.

    4 text layers, so the f16 rescaling of the export (from layer 3 on) is exercised. Images are resized to 4x4 ViT
    patches and to 32x32 pixels (8x8 latent tokens) for the VAE.
    """
    try:
        from transformers import (
            AutoTokenizer,
            BagelConfig,
            BagelForConditionalGeneration,
            BagelImageProcessor,
            BagelProcessor,
        )
        from transformers.models.bagel.convert_bagel_weights_to_hf import CHAT_TEMPLATE
    except ImportError as error:
        pytest.skip(f"transformers with BAGEL support is required: {error}")

    tokenizer = retry_request(lambda: AutoTokenizer.from_pretrained(BAGEL_TOKENIZER_ID))
    tokenizer.chat_template = CHAT_TEMPLATE
    image_processor = BagelImageProcessor(
        size={"longest_edge": 56, "shortest_edge": 28},
        max_num_patches_per_side=4,
        max_pixels=56 * 56,
        vae_max_size=32,
        vae_min_size=16,
    )
    processor = BagelProcessor(image_processor=image_processor, tokenizer=tokenizer, chat_template=CHAT_TEMPLATE)
    token_id = tokenizer.convert_tokens_to_ids
    config = BagelConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 512,
            # as the converted checkpoint: every prompt is wrapped in <|im_start|> ... <|im_end|>
            "bos_token_id": token_id("<|im_start|>"),
            "eos_token_id": token_id("<|im_end|>"),
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 14,
            "image_size": 56,
            "max_num_patches_per_side": 4,
        },
        vae_config={"base_channels": 32, "channel_multiplier": [1, 2], "num_res_blocks": 1, "latent_channels": 4},
        max_latent_size=8,
        start_of_image_token_id=token_id("<|vision_start|>"),
        end_of_image_token_id=token_id("<|vision_end|>"),
        image_token_id=token_id("<|image_pad|>"),
    )
    torch.manual_seed(BAGEL_TINY_MODEL_SEED)
    model = BagelForConditionalGeneration(config).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Generated tiny BAGEL model at %s", output_dir)
    return output_dir


def export_bagel_model_to_openvino(local_model_path: Path, output_path: Path) -> Path:
    """Export a local BAGEL checkpoint to OpenVINO IR via optimum-cli."""
    command = [
        "optimum-cli",
        "export",
        "openvino",
        "-m",
        str(local_model_path),
        "--task",
        "image-text-to-text",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=BAGEL_EXPORT_TIMEOUT_SECONDS)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"BAGEL export failed with return code {error.returncode}\nstdout: {error.stdout}\nstderr: {error.stderr}"
        ) from error
    logger.info("Exported tiny BAGEL model to %s", output_path)
    return output_path


def prepare_tiny_bagel_ov_path(converted_models_dir: Path) -> Path:
    """Prepare the tiny random BAGEL checkpoint and its OpenVINO export in cache and return the export path."""
    model_path = Path(converted_models_dir) / "tiny-random-bagel"
    AtomicDownloadManager(model_path).execute(generate_tiny_bagel_model)
    ov_path = Path(converted_models_dir) / "tiny-random-bagel-ov"
    AtomicDownloadManager(ov_path).execute(lambda temp_path: export_bagel_model_to_openvino(model_path, temp_path))
    return ov_path
