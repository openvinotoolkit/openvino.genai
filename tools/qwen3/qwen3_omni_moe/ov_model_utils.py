from pathlib import Path
from typing import Any

import gc
import openvino as ov

from .constants import PKV_INPUT_PREFIX, PKV_OUTPUT_PREFIX
from .utils import cleanup_torchscript_cache


def set_ov_model_names(
    ov_model: ov.Model,
    base_input_names: list[str],
    base_output_names: list[str],
    num_pkv: int,
    suffix_input_names: list[str] | None = None,
) -> None:
    input_names = list(base_input_names)
    output_names = list(base_output_names)
    for i in range(num_pkv):
        input_names.extend(
            [
                f"{PKV_INPUT_PREFIX}.{i}.key",
                f"{PKV_INPUT_PREFIX}.{i}.value",
            ]
        )
        output_names.extend(
            [
                f"{PKV_OUTPUT_PREFIX}.{i}.key",
                f"{PKV_OUTPUT_PREFIX}.{i}.value",
            ]
        )
    if suffix_input_names:
        input_names.extend(suffix_input_names)
    for inp, name in zip(ov_model.inputs, input_names):
        inp.get_tensor().set_names({name})
    for out, name in zip(ov_model.outputs, output_names):
        out.get_tensor().set_names({name})


def quantize_and_save_ov_model(
    ov_model: ov.Model,
    output_path: Path,
    quantization_config: dict[str, Any] | None = None,
) -> None:
    if quantization_config is not None:
        import nncf

        print(f"Weights compression with {quantization_config['mode']} mode started...")
        ov_model = nncf.compress_weights(ov_model, **quantization_config)
        print("Weights compression finished")
    ov.save_model(ov_model, output_path, compress_to_fp16=True)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
