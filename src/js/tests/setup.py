#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Setup script to download and convert models for JS tests.
This is a Python equivalent of setup.js
"""

import argparse
import os
import sys
from pathlib import Path
from optimum.intel import (
    OVModelForCausalLM,
    OVModelForVisualCausalLM,
    OVModelForFeatureExtraction,
    OVModelForSequenceClassification,
)

# Add the Python tests utils directory to the path
tests_utils_path = Path(__file__).parent.parent.parent.parent / "tests" / "python_tests"
sys.path.insert(0, str(tests_utils_path))

from utils import hugging_face, constants

TEST_MODELS = {
    "LLM": {
        "model_id": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
        "model_class": OVModelForCausalLM,
    },
    "VLM": {
        "model_id": "optimum-intel-internal-testing/tiny-random-qwen2vl",
        "model_class": OVModelForVisualCausalLM,
        "trust_remote_code": True,
    },
    "EMBEDDING_MODEL": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "model_class": OVModelForFeatureExtraction,
    },
    "RERANK_MODEL": {
        "model_id": "cross-encoder/ms-marco-TinyBERT-L2-v2",
        "model_class": OVModelForSequenceClassification,
    },
}

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
