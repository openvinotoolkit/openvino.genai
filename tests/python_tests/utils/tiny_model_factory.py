# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Registry-based factory for generating tiny random VLM models at test time.

Models registered here are generated locally instead of being downloaded from
HuggingFace. This allows fixing model dimensions that trigger platform-specific
bugs (e.g., CPU plugin edge cases on SPR) without waiting for upstream updates.

The factory only handles PyTorch model generation. OV IR export is performed
by the existing _get_ov_model() conversion pipeline in test_vlm_pipeline.py.
Generation is on-demand: triggered when _get_ov_model() is called for a
registered model_id.

To add a new locally-generated model:

    @register_generator("optimum-intel-internal-testing/tiny-random-<name>")
    def _generate_<name>(output_dir: Path) -> None:
        # 1. Build config from a reference model
        # 2. Create model with random weights
        # 3. Save model + tokenizer to output_dir
        ...
"""

import logging
import time
from pathlib import Path
from typing import Callable

from utils.constants import get_ov_cache_dir

logger = logging.getLogger(__name__)

# Registry: model_id -> generator function(output_dir)
_GENERATORS: dict[str, Callable[[Path], None]] = {}


def register_generator(model_id: str) -> Callable:
    """Decorator to register a tiny model generator for a given model_id."""

    def decorator(fn: Callable[[Path], None]) -> Callable[[Path], None]:
        _GENERATORS[model_id] = fn
        return fn

    return decorator


def is_locally_generated(model_id: str) -> bool:
    """Return True if model_id should be generated locally instead of downloaded."""
    return model_id in _GENERATORS


def get_hf_model_path(model_id: str) -> Path:
    """Return the cache path where a locally-generated HF model is stored."""
    generated_dir = get_ov_cache_dir() / "generated_hf_models"
    generated_dir.mkdir(parents=True, exist_ok=True)
    dir_name = model_id.replace("/", "_")
    return generated_dir / dir_name


def generate_hf_model(model_id: str) -> Path:
    """
    Generate a tiny HF PyTorch model on demand. Returns path to the HF model directory.

    Skips generation if the model already exists at the expected path.
    """
    if model_id not in _GENERATORS:
        raise ValueError(f"No generator registered for model_id={model_id!r}")

    hf_path = get_hf_model_path(model_id)
    if hf_path.exists() and (hf_path / "config.json").exists():
        logger.info(f"[{model_id}] HF model already cached: {hf_path}")
        return hf_path

    hf_path.mkdir(parents=True, exist_ok=True)
    generator_fn = _GENERATORS[model_id]

    t0 = time.perf_counter()
    generator_fn(hf_path)
    elapsed = time.perf_counter() - t0
    logger.info(f"[{model_id}] HF model generated in {elapsed:.2f}s -> {hf_path}")

    return hf_path
