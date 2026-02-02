import numpy as np
import yaml
from pathlib import Path
from typing import Any
from PIL import Image
from openvino import Tensor
import openvino_genai

def load_image(image_path: str) -> Tensor:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    # Add batch dimension: [H, W, 3] -> [1, H, W, 3]
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    return Tensor(arr)


def load_video(video_path: str) -> Tensor:
    entry = Path(video_path)
    if not entry.exists():
        raise FileNotFoundError(video_path)

    if entry.is_file():
        # Minimal behavior: treat a single file as a 1-frame video
        return load_image(str(entry))

    frames: list[np.ndarray] = []
    for file in sorted(entry.iterdir()):
        if not file.is_file():
            continue
        try:
            pic = Image.open(file).convert("RGB")
        except Exception:
            continue
        frames.append(np.array(pic, dtype=np.uint8))

    if not frames:
        raise RuntimeError(f"No readable image frames in directory: {video_path}")

    # [N, H, W, 3]
    return Tensor(np.stack(frames, axis=0))

def get_parameter_module_outputs(config_yaml_path: Path) -> list[dict[str, Any]]:

    config = yaml.safe_load(config_yaml_path.read_text(encoding="utf-8"))
    pipeline_modules = config.get("pipeline_modules")
    if not isinstance(pipeline_modules, dict):
        raise RuntimeError("Invalid config: missing 'pipeline_modules'")

    for _, module in pipeline_modules.items():
        if isinstance(module, dict) and module.get("type") == "ParameterModule":
            outputs = module.get("outputs")
            if isinstance(outputs, list):
                return [x for x in outputs if isinstance(x, dict)]
            raise RuntimeError("Invalid config: ParameterModule.outputs is not a list")

    raise RuntimeError("Could not find ParameterModule in config YAML.")
