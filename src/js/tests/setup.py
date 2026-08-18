#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Setup script to download and convert models for JS tests.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from optimum.intel import (
    OVModelForCausalLM,
    OVFluxPipeline,
    OVModelForVisualCausalLM,
    OVModelForFeatureExtraction,
    OVModelForSequenceClassification,
    OVModelForSpeechSeq2Seq,
    OVModelForTextToSpeechSeq2Seq,
    OVModelForMultimodalLM,
)

# Add the Python tests utils directory to the path
tests_utils_path = Path(__file__).parent.parent.parent.parent / "tests" / "python_tests"
sys.path.insert(0, str(tests_utils_path))

from utils import hugging_face

TEST_MODELS = {
    "LLM": {
        "model_id": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
        "model_class": OVModelForCausalLM,
    },
    "VLM": {
        "model_id": "optimum-intel-internal-testing/tiny-random-qwen2vl",
        "model_class": OVModelForVisualCausalLM,
    },
    "OMNI": {
        "model_id": "optimum-intel-internal-testing/tiny-random-qwen3-omni",
        "model_class": OVModelForMultimodalLM,
    },
    "EMBEDDING_MODEL": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "model_class": OVModelForFeatureExtraction,
    },
    "RERANK_MODEL": {
        "model_id": "cross-encoder/ms-marco-TinyBERT-L2-v2",
        "model_class": OVModelForSequenceClassification,
    },
    "WHISPER_MODEL": {
        "model_id": "openai/whisper-tiny",
        "model_class": OVModelForSpeechSeq2Seq,
    },
    "TTS_MODEL": {
        "model_id": "hf-internal-testing/tiny-random-SpeechT5ForTextToSpeech",
        "model_class": OVModelForTextToSpeechSeq2Seq,
        "model_kwargs": {
            "vocoder": "fxmarty/speecht5-hifigan-tiny",
        },
    },
    "IMAGE_GENERATION_MODEL": {
        "model_id": "optimum-intel-internal-testing/tiny-random-flux",
        "model_class": OVFluxPipeline,
        "model_kwargs": {"has_tokenizer": False},
    },
}


def _patch_tiny_qwen3_omni(model_path: Path, hf_tokenizer) -> None:
    """Align the known-broken tiny checkpoint until it is regenerated upstream."""
    native_audio_tokens = ["<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"]
    missing_audio_tokens = [token for token in native_audio_tokens if token not in hf_tokenizer.get_vocab()]
    if missing_audio_tokens:
        added_tokens = hf_tokenizer.add_special_tokens({"additional_special_tokens": native_audio_tokens})
        if added_tokens != len(missing_audio_tokens):
            raise RuntimeError(
                f"Expected to add {len(missing_audio_tokens)} Qwen3-Omni audio tokens, added {added_tokens}"
            )
        hf_tokenizer.save_pretrained(model_path)
        hugging_face.convert_and_save_tokenizer(hf_tokenizer, model_path)

    def token_id(token: str) -> int:
        return int(hf_tokenizer.convert_tokens_to_ids(token))

    def first_token_id(text: str) -> int:
        tokens = hf_tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            raise RuntimeError(f"Qwen3-Omni tokenizer produced no tokens for role '{text}'")
        return int(tokens[0])

    config_path = model_path / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    role_token_ids = {role: first_token_id(role) for role in ("system", "user", "assistant")}
    expected_config = {
        "im_start_token_id": token_id("<|im_start|>"),
        "im_end_token_id": token_id("<|im_end|>"),
        "system_token_id": role_token_ids["system"],
        "user_token_id": role_token_ids["user"],
        "assistant_token_id": role_token_ids["assistant"],
    }
    expected_thinker_config = {
        "audio_start_token_id": token_id("<|audio_start|>"),
        "audio_token_id": token_id("<|audio_pad|>"),
        "image_token_id": token_id("<|image_pad|>"),
        "video_token_id": token_id("<|video_pad|>"),
        "vision_start_token_id": token_id("<|vision_start|>"),
        "user_token_id": role_token_ids["user"],
    }

    if all(config.get(key) == value for key, value in expected_config.items()) and all(
        config["thinker_config"].get(key) == value for key, value in expected_thinker_config.items()
    ):
        return

    config.update(expected_config)
    config["thinker_config"].update(expected_thinker_config)

    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")

if __name__ == "__main__":
    """Download and convert all models required for JS tests."""
    # Check if OV_CACHE environment variable is set
    if "OV_CACHE" not in os.environ:
        os.environ["OV_CACHE"] = "./ov_cache"
        print("OV_CACHE environment variable is not set. Using default './ov_cache' directory.")

    parser = argparse.ArgumentParser(description="Download and convert models for JS tests")
    parser.add_argument(
        "--to-env-file", type=str, help="Path to the .env file to save environment variables (default: test.env)"
    )
    args = parser.parse_args()

    env_vars = {}

    for model_name, model_info in TEST_MODELS.items():
        try:
            result = hugging_face.download_and_convert_model_class(**model_info)
            if model_name == "OMNI":
                _patch_tiny_qwen3_omni(result.models_path, result.hf_tokenizer)
            env_vars[f"{model_name}_PATH"] = str(result.models_path)
        except Exception as e:
            print(f"Error processing model '{model_name}': {e}")
            raise
    print(f"All models downloaded and converted successfully!")

    # Write environment variables to .env file
    result = [f"{var_name}={var_value}\n" for var_name, var_value in env_vars.items()]
    if args.to_env_file:
        with open(args.to_env_file, "w") as f:
            f.writelines(result)
        print(f"Environment variables saved to: {args.to_env_file}")
    else:
        print("\nPaths to the test models:")
        print("".join(result))
