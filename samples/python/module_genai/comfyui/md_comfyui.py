#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
OpenVINO GenAI ModulePipeline for ComfyUI

Standalone tool to run OpenVINO GenAI ModulePipeline from ComfyUI JSON workflow.
Supports both workflow and API JSON formats.

Usage:
    python ./samples/python/module_genai/comfyui/md_comfyui.py \
        --json <comfyui.json> --model_path <model_path> [options]

Examples:
    python ./samples/python/module_genai/comfyui/md_comfyui.py \
        --json ./samples/cpp/module_genai/comfyui/wan2.1_t2v.json \
        --model_path ./models/Wan2.1-T2V --device GPU
    python ./samples/python/module_genai/comfyui/md_comfyui.py \
        --json ./samples/cpp/module_genai/comfyui/z_image_turbo_2k_tiled.json \
        --model_path ./models/Z-Image-Turbo --device GPU
"""

# Fix Python path priority: ensure pip-installed packages take precedence over
# OpenVINO toolkit bundled packages (which may be outdated)
import sys
import site
import numpy as np
import argparse
import os
import time
from PIL import Image

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenVINO GenAI ModulePipeline for ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --json wan2.1_t2v.json --model_path ./models/Wan2.1-T2V --device GPU
  %(prog)s --json z_image_turbo_2k_tiled.json --model_path ./models/Z-Image-Turbo --device GPU
  %(prog)s --json z_image_turbo_2k_non_tiled_api.json --model_path ./models/Z-Image-Turbo --width 1024 --height 1024
        """
    )
    parser.add_argument('--json', required=True,
                        help="ComfyUI JSON file (API or workflow format)")
    parser.add_argument('--model_path', required=True,
                        help="Model path base (required)")
    parser.add_argument('--device', default="GPU",
                        help="Device to run on (default: GPU)")
    parser.add_argument('--prompt', default=None,
                        help="Text prompt for generation (overrides JSON value)")
    parser.add_argument('--output', default=None,
                        help="Output filename (auto-generated if not specified)")
    parser.add_argument('--width', type=int, default=None,
                        help="Image/video width (overrides JSON value)")
    parser.add_argument('--height', type=int, default=None,
                        help="Image/video height (overrides JSON value)")
    parser.add_argument('--num_frames', type=int, default=None,
                        help="Number of video frames (overrides JSON value)")
    parser.add_argument('--steps', type=int, default=None,
                        help="Number of inference steps (overrides JSON value)")
    parser.add_argument('--guidance', type=float, default=None,
                        help="Guidance scale (overrides JSON value)")
    parser.add_argument('--max-seq-len', dest='max_seq_len', type=int, default=None,
                        help="Max sequence length (overrides JSON value)")
    parser.add_argument('--tile_size', type=int, default=None,
                        help="VAE decoder tile size in pixels (default: 256)")
    parser.add_argument('--use_tiling', type=int, default=None, choices=[-1, 0, 1],
                        help="VAE tiling mode: -1=auto (default, enabled for Wan 2.1), 0=disable, 1=enable")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed (overrides JSON value)")
    parser.add_argument('--verbose', type=int, default=2,
                        help="Verbosity: 0=quiet, 1=error, 2=info (default), 3=debug")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug output (same as --verbose 3)")
    return parser.parse_args()


# ============================================================================
# Logging
# ============================================================================

from utils import Logger

log = Logger()


# ============================================================================
# ComfyUI JSON Conversion
# ============================================================================

def validate_and_print_config(yaml_content: str, config_name: str = "config") -> bool:
    """
    Validate YAML configuration and print results.

    :param yaml_content: YAML configuration string
    :param config_name: Name for logging purposes
    :return: True if valid, False otherwise
    """
    import openvino_genai

    log.info(f"Validating {config_name}...")
    result = openvino_genai.ModulePipeline.validate_config_string(yaml_content)

    log.info(f"Valid: {result.valid}")

    if result.errors:
        log.error(f"Errors ({len(result.errors)}):")
        for error in result.errors:
            log.error(f"  - {error}")

    if result.warnings:
        for warning in result.warnings:
            log.warning(warning)

    return result.valid


def load_from_comfyui_json(json_path: str, **kwargs) -> tuple:
    """
    Load pipeline configuration from ComfyUI JSON file.

    :param json_path: Path to ComfyUI JSON file
    :param kwargs: Optional parameters (model_path, device, tile_size, etc.)
    :return: Tuple of (yaml_content, extracted_params)
    """
    import openvino_genai

    log.info(f"Converting ComfyUI JSON: {json_path}")

    yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_to_yaml(
        json_path,
        **kwargs
    )

    log.info("Conversion successful!")
    log.debug("Extracted parameters:")
    for key, value in params.items():
        # Truncate long string values for cleaner log output
        str_value = str(value)
        if len(str_value) > 60:
            log.debug(f"  {key}: {str_value[:60]}...")
        else:
            log.debug(f"  {key}: {value}")

    return yaml_content, params


# ============================================================================
# Pipeline Execution
# ============================================================================

def build_pipeline_inputs(params: dict, args) -> dict:
    """
    Build pipeline inputs from extracted params and command line arguments.
    Command line arguments override extracted values.

    :param params: Parameters extracted from ComfyUI JSON
    :param args: Command line arguments
    :return: Dictionary of pipeline inputs
    """
    inputs = {}

    # Copy extracted inputs from ComfyUI JSON
    param_mappings = [
        ("prompt", str, ""),
        ("negative_prompt", str, ""),
        ("guidance_scale", float, 1.0),
        ("num_inference_steps", int, 4),
        ("width", int, 1024),
        ("height", int, 1024),
        ("num_frames", int, None),
        ("max_sequence_length", int, 512),
        ("batch_size", int, 1),
        ("seed", int, 42),
        ("tile_size", int, None),
    ]

    for key, dtype, default in param_mappings:
        if key in params:
            inputs[key] = params[key]
        elif default is not None:
            inputs[key] = default

    # Command line arguments override extracted values
    if args.prompt is not None:
        inputs["prompt"] = args.prompt
    if args.width is not None:
        inputs["width"] = args.width
    if args.height is not None:
        inputs["height"] = args.height
    if args.num_frames is not None:
        inputs["num_frames"] = args.num_frames
    if args.steps is not None:
        inputs["num_inference_steps"] = args.steps
    if args.guidance is not None:
        inputs["guidance_scale"] = args.guidance
    if args.max_seq_len is not None:
        inputs["max_sequence_length"] = args.max_seq_len
    if args.tile_size is not None:
        inputs["tile_size"] = args.tile_size
    if args.seed is not None:
        inputs["seed"] = args.seed

    return inputs


def print_pipeline_inputs(inputs: dict):
    """Print pipeline inputs for debugging."""
    log.info("Final pipeline inputs:")
    log.info(f"  - prompt: \"{inputs.get('prompt', '')[:60]}...\""
             if len(inputs.get('prompt', '')) > 60
             else f"  - prompt: \"{inputs.get('prompt', '')}\"")
    log.info(f"  - negative_prompt: \"{inputs.get('negative_prompt', '')[:40]}...\""
             if len(inputs.get('negative_prompt', '')) > 40
             else f"  - negative_prompt: \"{inputs.get('negative_prompt', '')}\"")
    log.info(f"  - width: {inputs.get('width')}")
    log.info(f"  - height: {inputs.get('height')}")
    if inputs.get('num_frames'):
        log.info(f"  - num_frames: {inputs.get('num_frames')}")
    log.info(f"  - batch_size: {inputs.get('batch_size', 1)}")
    log.info(f"  - seed: {inputs.get('seed', 42)}")
    log.info(f"  - steps: {inputs.get('num_inference_steps')}")
    log.info(f"  - guidance: {inputs.get('guidance_scale')}")
    log.info(f"  - max_seq_len: {inputs.get('max_sequence_length')}")
    if inputs.get('tile_size'):
        log.info(f"  - tile_size: {inputs.get('tile_size')}")


def handle_pipeline_output(pipe, output_file: str = None) -> bool:
    """
    Handle pipeline output - get and save/verify image or video.

    Priority: saved_video > saved_image > image tensor

    :param pipe: ModulePipeline instance
    :param output_file: Optional output filename
    :return: True if successful, False otherwise
    """

    log.info("Getting output...")

    # 1. Try to get saved_video path (from SaveVideoModule)
    try:
        output_path = pipe.get_output("saved_video")
        if output_path:
            log.info(f"Video saved by pipeline to: {output_path}")
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                log.success(f"Video saved successfully: {output_path} ({file_size} bytes)")
                return True
            else:
                log.warning(f"Saved video file does not exist: {output_path}")
    except Exception:
        pass  # saved_video output not available

    # 2. Try to get saved_image path (from SaveImageModule)
    try:
        output_path = pipe.get_output("saved_image")
        if output_path:
            log.info(f"Image saved by pipeline to: {output_path}")
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                log.success(f"Image saved successfully: {output_path} ({file_size} bytes)")
                return True
            else:
                log.warning(f"Saved image file does not exist: {output_path}")
    except Exception:
        pass  # saved_image output not available

    # 3. Fallback: try to get raw image tensor
    try:
        output = pipe.get_output("image")
        if output is not None:
            image_data = np.array(output.data)
            log.info(f"Output tensor shape: {image_data.shape}")
            log.info(f"Output tensor dtype: {image_data.dtype}")

            # Convert to image
            if image_data.ndim == 3:
                image_data = image_data[None, ...]

            # Generate output filename if not specified
            if not output_file:
                output_file = f"output_comfyui_{int(time.time())}.png"

            log.info(f"Saving output image to: {output_file}")
            pil_image = Image.fromarray(image_data[0].astype("uint8"))
            pil_image.save(output_file)
            log.success(f"Image saved successfully: {output_file}")
            return True
    except Exception as e:
        log.debug(f"Failed to get image tensor: {e}")

    log.error("No valid output found (tried: saved_video, saved_image, image)")
    return False


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Set log level
    if args.debug:
        log.level = Logger.DEBUG
    else:
        log.level = args.verbose

    # Import heavy modules after argument parsing
    import openvino_genai

    if args.debug:
        log.debug("Environment Info:")
        log.debug(f"  Python: {sys.version}")
        import openvino
        log.debug(f"  OpenVINO: {openvino.__version__}")
        log.debug(f"  openvino_genai location: {openvino_genai.__file__}")

    try:
        # ====================================================================
        # Step 1: Get YAML content from JSON conversion
        # ====================================================================

        if not os.path.exists(args.json):
            log.error(f"JSON file not found: {args.json}")
            return 1

        log.info(f"Loading ComfyUI JSON from: {args.json}")

        model_path = args.model_path

        # Build conversion kwargs
        conversion_kwargs = {
            "model_path": model_path,
            "device": args.device
        }
        if args.tile_size is not None:
            conversion_kwargs["tile_size"] = args.tile_size
        if args.use_tiling is not None:
            # -1=auto (default), 0=disable, 1=enable
            if args.use_tiling == -1:
                pass  # auto mode, don't set use_tiling
            else:
                conversion_kwargs["use_tiling"] = (args.use_tiling == 1)

        # Convert JSON to YAML
        yaml_content, params = load_from_comfyui_json(args.json, **conversion_kwargs)

        if not yaml_content:
            log.error("Failed to convert JSON to YAML")
            return 1

        log.info("Converted to YAML successfully")

        # Save generated YAML to file for debugging
        yaml_debug_file = args.json + ".generated.yaml"
        with open(yaml_debug_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        log.info(f"Generated YAML saved to: {yaml_debug_file}")

        # Print YAML content for debugging
        if log.level >= Logger.DEBUG:
            print(f"\n[DEBUG] ====== YAML Pipeline Config ======")
            print(yaml_content)
            print(f"[DEBUG] ==================================\n")

        # Validate configuration
        if not validate_and_print_config(yaml_content, "Converted ComfyUI Config"):
            log.warning("Configuration validation failed, proceeding anyway...")

        # ====================================================================
        # Step 2: Create ModulePipeline
        # ====================================================================

        log.info("Creating ModulePipeline...")
        pipe = openvino_genai.ModulePipeline(config_yaml_content=yaml_content)
        log.info("ModulePipeline created successfully!")

        # ====================================================================
        # Step 3: Prepare inputs and run
        # ====================================================================

        inputs = build_pipeline_inputs(params, args)
        print_pipeline_inputs(inputs)

        log.info("Running pipeline.generate()...")
        start_time = time.time()

        pipe.generate(**inputs)

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        log.info(f"Generation completed in {duration_ms:.0f} ms")

        # ====================================================================
        # Step 4: Get output and verify saved image/video
        # ====================================================================

        if not handle_pipeline_output(pipe, args.output):
            return 1

        log.success("Pipeline execution completed!")
        return 0

    except Exception as e:
        log.error(f"Exception: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
