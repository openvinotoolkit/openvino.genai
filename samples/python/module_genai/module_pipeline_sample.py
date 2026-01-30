#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Sample script for OpenVINO GenAI ModulePipeline.

This script demonstrates how to run a ModulePipeline with YAML configuration.

Usage:
    python module_pipeline_sample.py [--yaml <path>] [--prompt <text>] [--image <path>] [--debug]

Examples:
    # Run with default settings
    python module_pipeline_sample.py

    # Run with specific config
    python module_pipeline_sample.py --yaml ./qwen_2_5_val_pipeline.yaml

    # Run with custom prompt and image
    python module_pipeline_sample.py --yaml config.yaml --prompt "What is in this image?" --image ./my_image.png

    # Enable debug output
    python module_pipeline_sample.py --debug
"""

# Fix Python path priority: ensure pip-installed packages take precedence over
# OpenVINO toolkit bundled packages (which may be outdated)
import sys
import site

def _fix_python_path():
    """Ensure site-packages has higher priority than OpenVINO toolkit path."""
    site_packages = site.getsitepackages()
    user_site = site.getusersitepackages()
    priority_paths = [user_site] + site_packages if user_site else site_packages
    for path in priority_paths:
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

_fix_python_path()
# End of path fix

import argparse
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None
    print("Warning: PIL not available, image loading will be limited")

try:
    import openvino_genai
    from openvino import Tensor
except ImportError as e:
    print(f"Error: Failed to import openvino_genai: {e}")
    sys.exit(1)


# ============================================================================
# Image Utility Functions
# ============================================================================

def load_image(image_path: str) -> Tensor:
    """
    Load a single image from file.

    :param image_path: Path to the image file
    :return: ov.Tensor with shape [1, H, W, 3] in RGB format
    """
    if Image is None:
        raise RuntimeError("PIL is required for image loading. Install with: pip install Pillow")

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)

    # Add batch dimension: [H, W, 3] -> [1, H, W, 3]
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    return Tensor(img_array)


def load_images(input_path: str) -> List[Tensor]:
    """
    Load multiple images from a directory or a single image file.

    :param input_path: Path to a directory or single image file
    :return: List of ov.Tensor images
    """
    path = Path(input_path)
    images = []

    if path.is_file():
        images.append(load_image(str(path)))
    elif path.is_dir():
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() in image_extensions:
                images.append(load_image(str(file_path)))
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    return images


# ============================================================================
# Running Pipeline Helper
# ============================================================================

def run_module_pipeline(
    config_path: str,
    inputs: Dict[str, Any],
    output_names: List[str]
) -> List[Any]:
    """
    Running a ModulePipeline with given config and inputs.

    :param config_path: Path to the YAML config file
    :param inputs: Dictionary of input parameters
    :param output_names: List of output names to retrieve
    :return: List of outputs
    """
    print(f"  Creating ModulePipeline from: {config_path}")
    pipe = openvino_genai.ModulePipeline(config_path)

    start_time = time.time()
    pipe.generate(**inputs)
    elapsed_time = (time.time() - start_time) * 1000
    print(f"  Generate time: {elapsed_time:.0f} ms")

    outputs = []
    for name in output_names:
        try:
            output = pipe.get_output(name)
            outputs.append(output)
            print(f"  Got output '{name}': type={type(output).__name__}")
        except Exception as e:
            print(f"  Warning: Failed to get output '{name}': {e}")
            outputs.append(None)

    return outputs


def launch_module_pipeline(config_yaml: str = None, prompt: str = None, image_path: str = None):
    """
    Running Qwen2.5-VL module pipeline.

    :param config_yaml: Optional path to config YAML file
    :param prompt: Optional prompt text (overrides default)
    :param image_path: Optional path to input image (overrides default)
    """
    if config_yaml is None:
        config_yaml = "./qwen_2_5_val_pipeline.yaml"

    print(f"== Running: Qwen2.5-VL Module Pipeline ==")
    print(f"   Config: {config_yaml}")

    if not os.path.exists(config_yaml):
        print(f"   Error: Config file not found: {config_yaml}")
        os._exit(1)  # Fast exit to avoid slow module cleanup

    # Prepare inputs with defaults
    prompt_text = prompt if prompt else "Please describe this image"
    img_path = image_path if image_path else "./cat_120_100.png"
    print(f"   Prompt: {prompt_text}")
    print(f"   Image: {img_path}")

    inputs = {
        "prompts_data": [prompt_text],
        "img1": load_image(img_path)
    }

    # Run pipeline
    outputs = run_module_pipeline(config_yaml, inputs, ["generated_text"])

    if outputs and outputs[0] is not None:
        generated_text = outputs[0]
        print(f"  Generated Text: {generated_text}")

        # Optional validation
        # if "white cat" not in generated_text.lower():
        #     print("  Warning: Expected 'white cat' in output")

        return True
    else:
        print("  Error: No output generated")
        return False


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample pipelines for OpenVINO GenAI ModulePipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--yaml', default=None,
                        help="Path to YAML config file")
    parser.add_argument('--prompt', default=None,
                        help="Prompt text (overrides default)")
    parser.add_argument('--image', default=None,
                        help="Path to input image (overrides default)")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug output")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        print("\n[DEBUG] Environment Info:")
        print(f"  Python: {sys.version}")
        import openvino
        print(f"  OpenVINO: {openvino.__version__}")
        print(f"  openvino_genai location: {openvino_genai.__file__}")
        print(f"  Working directory: {os.getcwd()}")
        print()

    # Run modular pipeline
    print(f"\n{'='*60}")
    print(f"Running pipeline: qwen2_5_vl")
    print(f"{'='*60}\n")

    try:
        success = launch_module_pipeline(args.yaml, args.prompt, args.image)

        print(f"\n{'='*60}")
        print(f"Running result: {'PASS' if success else 'FAIL'}")
        print(f"{'='*60}\n")

        return 0 if success else 1

    except Exception as e:
        print(f"\n[ERROR] Running failed with exception:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
