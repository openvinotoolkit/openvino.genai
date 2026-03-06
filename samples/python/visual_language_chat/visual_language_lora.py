import argparse
import numpy as np
import openvino_genai as ov_genai

from pathlib import Path
from PIL import Image
from openvino import Tensor


def streamer(subword: str) -> bool:
    """

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    """
    print(subword, end="", flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return openvino_genai.StreamingStatus.RUNNING".


def read_image(path: str) -> Tensor:
    """

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    """
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def parse_lora_pairs(raw):
    if len(raw) % 2 != 0:
        raise argparse.ArgumentTypeError(
            "LoRA args must come in pairs: <LORA_SAFETENSORS> <ALPHA> ..."
        )

    pairs = []
    for i in range(0, len(raw), 2):
        path = raw[i]
        try:
            alpha = float(raw[i + 1])
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid alpha '{raw[i+1]}' for LoRA '{path}'"
            ) from e
        pairs.append((path, alpha))
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(
        description="OpenVINO GenAI VLM sample: run with and without LoRA adapters.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("model_dir", help="Path to model directory")
    p.add_argument("images_path", help="Image file OR directory with images")
    p.add_argument("device", choices=["CPU", "GPU"], help='Device, e.g. "CPU", "GPU"')
    p.add_argument(
        "lora_pairs",
        nargs="+",
        metavar=("LORA", "ALPHA"),
        help="Pairs: <LORA_SAFETENSORS> <ALPHA> ...",
    )

    args = p.parse_args()
    loras = parse_lora_pairs(args.lora_pairs)

    rgbs = read_images(args.images_path)

    pipe_kwargs = {}
    if args.device == "GPU":
        pipe_kwargs["ov_config"] = {"CACHE_DIR": "vlm_cache"}

    # Configure LoRA adapters with weights (alphas)
    if loras:
        adapter_config = ov_genai.AdapterConfig()
        for lora_path, alpha in loras:
            adapter_config.add(ov_genai.Adapter(lora_path), alpha)
        pipe_kwargs["adapters"] = adapter_config

    pipe = ov_genai.VLMPipeline(args.model_dir, args.device, **pipe_kwargs)

    gen_cfg = pipe.get_generation_config()
    gen_cfg.max_new_tokens = 100

    prompt = input("question:\n")

    print("----------\nGenerating answer with LoRA adapters applied:\n")
    pipe.generate(
        prompt,
        images=rgbs,
        generation_config=gen_cfg,
        streamer=streamer,
    )

    print("\n----------\nGenerating answer without LoRA adapters applied:\n")
    pipe.generate(
        prompt,
        images=rgbs,
        generation_config=gen_cfg,
        adapters=ov_genai.AdapterConfig(),
        streamer=streamer,
    )

    print("\n----------")
    return 0


if __name__ == "__main__":
    main()
