from pathlib import Path
from typing import Any
import argparse
import shutil

import torch
from transformers import AutoProcessor
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from huggingface_hub import snapshot_download, hf_hub_download

from .compat import apply_dense_compat
from .convert_thinker_embedding import (
    convert_thinker_embedding,
)
from .convert_audio_encoder import convert_audio_encoder
from .convert_vision_encoder import (
    convert_vision_patcher,
    convert_vision_merger,
)
from .convert_thinker_language import (
    convert_thinker_language,
)
from .convert_talker import (
    convert_talker_embedding,
    convert_talker_language,
)
from .constants import ATTN_IMPLEMENTATION, WEIGHT_FORMAT_TO_NNCF
from .convert_code_predictor import convert_code_predictor
from .convert_code2wav import convert_code2wav


def _load_model(
    model_id: str, output_dir: Path, use_local_dir: bool = False
) -> tuple[Qwen3OmniMoeForConditionalGeneration, str | Path]:
    if use_local_dir:
        ckpt: str | Path = output_dir / "ckpt"
        if not ckpt.exists():
            snapshot_download(model_id, local_dir=ckpt, force_download=True)
    elif Path(model_id).is_dir():
        ckpt = Path(model_id)
    else:
        ckpt = model_id

    apply_dense_compat(ckpt)

    config = Qwen3OmniMoeConfig.from_pretrained(ckpt)
    config.thinker_config._attn_implementation_autoset = False
    config.thinker_config._attn_implementation = ATTN_IMPLEMENTATION
    config.talker_config._attn_implementation_autoset = False
    config.talker_config._attn_implementation = ATTN_IMPLEMENTATION

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        ckpt,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, ckpt


def _save_configs(
    model: Qwen3OmniMoeForConditionalGeneration,
    ckpt: str | Path,
    output_dir: Path,
) -> None:
    processor = AutoProcessor.from_pretrained(ckpt)
    model.config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    spk_path: Path | None = None
    local_spk = Path(ckpt) / "spk_dict.pt"
    if local_spk.exists():
        spk_path = local_spk
    else:
        try:
            spk_path = Path(hf_hub_download(str(ckpt), filename="spk_dict.pt"))
        except Exception:
            pass

    if spk_path and not (output_dir / "spk_dict.pt").exists():
        shutil.copy(spk_path, output_dir / "spk_dict.pt")


def convert_qwen3_omni_moe(
    model_id: str,
    output_dir: str | Path,
    weight_format: str = "fp16",
    use_local_dir: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantization_config = WEIGHT_FORMAT_TO_NNCF.get(weight_format)

    print(f"{model_id} conversion started. Be patient, it may take some time.")
    print("Loading original model...")
    model, ckpt = _load_model(model_id, output_dir, use_local_dir)
    print("Original model successfully loaded")

    _save_configs(model, ckpt, output_dir)

    convert_thinker_embedding(model, output_dir)
    convert_audio_encoder(model, output_dir)
    convert_vision_patcher(model, output_dir)
    convert_vision_merger(model, output_dir)
    convert_thinker_language(model, output_dir, quantization_config)

    if model.has_talker:
        convert_talker_embedding(model, output_dir)
        convert_talker_language(model, output_dir, quantization_config)
        convert_code_predictor(model, output_dir, quantization_config)
        convert_code2wav(model, output_dir)

    print(f"{model_id} model conversion finished. Results in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3-Omni-MOE model to OpenVINO IR")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted models",
    )
    parser.add_argument(
        "--weight_format",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "int8", "int4"],
        help="Weight compression format",
    )
    parser.add_argument(
        "--use_local_dir",
        action="store_true",
        help="Download model to output_dir/ckpt first",
    )
    args = parser.parse_args()

    convert_qwen3_omni_moe(
        model_id=args.model_id,
        output_dir=args.output_dir,
        weight_format=args.weight_format,
        use_local_dir=args.use_local_dir,
    )


if __name__ == "__main__":
    main()
