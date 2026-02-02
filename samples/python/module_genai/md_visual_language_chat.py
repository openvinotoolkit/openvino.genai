# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from typing import Any
from openvino import Tensor
import openvino_genai

from utils.utils import load_image, load_video, get_parameter_module_outputs


def parse_vlm_inputs_from_yaml(cfg_yaml_path: Path, prompt: str, image_path: str, video_path: str) -> dict[str, Any]:
    outputs = get_parameter_module_outputs(cfg_yaml_path)
    inputs: dict[str, Any] = {}

    for out in outputs:
        name = out.get("name")
        typ = out.get("type")
        if "prompt" in name.lower() and typ == "String":
            inputs[name] = prompt
            continue

        if ("image" in name.lower() or "img" in name.lower()) and typ == "OVTensor":
            if image_path:
                inputs[name] = load_image(image_path)
            else:
                raise RuntimeError("Image path is empty.")
            continue

        if "video" in name.lower() and typ == "OVTensor":
            if video_path:
                inputs[name] = load_video(video_path)
            else:
                raise RuntimeError("Video path is empty.")
            continue

    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Module GenAI VLM sample (Python). "
            "Preferred usage: --cfg <config.yaml> --prompt <text> [--img <image>] [--video <frames_dir>]"
        )
    )

    # Preferred flags
    parser.add_argument("--cfg", dest="cfg", default="", help="Path to config.yaml")
    parser.add_argument("--prompt", dest="prompt", default="", help="Prompt text")
    parser.add_argument("--img", dest="img", default="", help="Optional image path")
    parser.add_argument("--video", dest="video", default="", help="Optional video path (dir of frames)")

    args = parser.parse_args()

    cfg_yaml_path = Path(args.cfg)
    if not cfg_yaml_path.exists():
        raise FileNotFoundError(str(cfg_yaml_path))

    inputs = parse_vlm_inputs_from_yaml(cfg_yaml_path, args.prompt, args.img, args.video)
    for key, value in inputs.items():
        print(f"[Input] {key}: {str(value)}")

    pipe = openvino_genai.ModulePipeline(str(cfg_yaml_path))
    pipe.generate(**inputs)

    generated = pipe.get_output("generated_text")
    print("Generation Result:", str(generated))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

