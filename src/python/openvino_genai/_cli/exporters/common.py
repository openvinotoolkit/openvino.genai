# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Building blocks shared by the native genai-cli exporters.

The output layout and the post-processing (stateful KV cache, weight compression defaults, tokenizer
conversion) follow optimum-intel, so OpenVINO GenAI pipelines consume the models exactly as they
consume optimum-intel exports.
"""

import gc
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import openvino as ov

from openvino_genai._cli.args import no_compression_parameter_provided

logger = logging.getLogger(__name__)

OV_TOKENIZER_NAME = "openvino_tokenizer.xml"
OV_DETOKENIZER_NAME = "openvino_detokenizer.xml"

# Submodels with at least this number of weights are compressed to int8 when no weight format is given.
_MAX_UNCOMPRESSED_SIZE = 1e9

# Files of a Hugging Face checkpoint that are not needed next to the OpenVINO IRs.
_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".h5", ".gguf")
_WEIGHT_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def torch_export(model, args: tuple = (), kwargs: Optional[Dict[str, Any]] = None, dynamic_shapes=None):
    """Captures ``model`` with torch.export (no TorchScript involved)."""
    import torch

    _export_kwargs = {"args": args, "kwargs": kwargs or {}, "dynamic_shapes": dynamic_shapes, "strict": False}
    # torch.export.export_for_training was deprecated in torch 2.10 and removed in 2.11, where torch.export.export
    # produces the same training IR.
    export_for_training = getattr(torch.export, "export_for_training", None) or torch.export.export
    with torch.no_grad():
        ep = export_for_training(model, **_export_kwargs)
    return ep


def convert_exported_program(ep, input_names: Sequence[str], output_names: Sequence[str]) -> ov.Model:
    """Converts a torch.export ExportedProgram to ov.Model and names its inputs and outputs."""
    # The training IR keeps autocast / grad-mode regions as higher-order ops. Lowering with an empty decomposition
    # table inlines them while keeping operations such as scaled_dot_product_attention intact for OpenVINO fusions.
    ep = ep.run_decompositions(decomp_table={})
    ov_model = ov.convert_model(ep)
    if len(ov_model.inputs) != len(input_names) or len(ov_model.outputs) != len(output_names):
        raise RuntimeError(
            f"Converted model has {len(ov_model.inputs)} inputs and {len(ov_model.outputs)} outputs, "
            f"expected {len(input_names)} and {len(output_names)}"
        )
    # Dimensions that torch.export kept static (e.g. hidden size) stay static in the OpenVINO model.
    user_inputs = set(ep.graph_signature.user_inputs)
    example_inputs = [node.meta["val"] for node in ep.graph.nodes if node.op == "placeholder" and node.name in user_inputs]
    for parameter, example in zip(ov_model.get_parameters(), example_inputs):
        dims = [dim if isinstance(dim, int) else -1 for dim in example.shape]
        parameter.set_partial_shape(ov.PartialShape(dims))
    for port, name in zip(ov_model.inputs, input_names):
        port.get_tensor().set_names({name})
    for port, name in zip(ov_model.outputs, output_names):
        port.get_tensor().set_names({name})
    ov_model.validate_nodes_and_infer_types()
    return ov_model


def make_stateful(ov_model: ov.Model, batch_dim: int = 0) -> None:
    """Hides the ``past_key_values.*`` inputs / ``present.*`` outputs of a decoder inside the model as states.

    Adds the ``beam_idx`` input that reorders the cache between generation steps (as optimum-intel does).
    """
    from openvino import opset13
    from openvino._offline_transformations import apply_make_stateful_transformation

    kv_inputs = [port for port in ov_model.inputs if port.get_any_name().startswith("past_key_values")]
    kv_outputs = [port for port in ov_model.outputs if port.get_any_name().startswith("present")]
    main_input = ov_model.input("inputs_embeds" if "inputs_embeds" in ov_model.input(0).get_names() else "input_ids")

    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([main_input.get_partial_shape()[0]]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    for kv_input in kv_inputs:
        consumers = kv_input.get_target_inputs()
        gather = opset13.gather(kv_input, beam_idx, opset13.constant(batch_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(
        ov_model, {kv_in.get_any_name(): kv_out.get_any_name() for kv_in, kv_out in zip(kv_inputs, kv_outputs)}
    )

    # Initialize the states with zero-sized caches of the current batch size.
    batch = opset13.gather(opset13.shape_of(main_input, output_type="i64"), opset13.constant([0]), opset13.constant(0))
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in op.get_output_partial_shape(0)]
            dims[batch_dim] = batch
            dims = [opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim for dim in dims]
            shape = opset13.concat(dims, axis=0)
            op.set_arguments([opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)])
    ov_model.validate_nodes_and_infer_types()


@dataclass
class WeightCompressionConfig:
    """Weight-only compression parameters, a subset of nncf.compress_weights arguments."""

    mode: str
    ratio: Optional[float] = None
    group_size: Optional[int] = None
    all_layers: Optional[bool] = None
    awq: Optional[bool] = None
    sensitivity_metric: Optional[str] = None
    backup_precision: Optional[str] = None
    group_size_fallback: Optional[str] = None

    def compress(self, ov_model: ov.Model) -> ov.Model:
        import nncf
        from nncf.quantization.advanced_parameters import AdvancedCompressionParameters, GroupSizeFallbackMode

        kwargs = {
            "mode": nncf.CompressWeightsMode(self.mode),
            "ratio": self.ratio,
            "group_size": self.group_size,
            "all_layers": self.all_layers,
            "awq": self.awq,
            "sensitivity_metric": nncf.SensitivityMetric(self.sensitivity_metric) if self.sensitivity_metric else None,
            "backup_mode": nncf.BackupMode(self.backup_precision) if self.backup_precision else None,
        }
        if self.group_size_fallback is not None:
            kwargs["advanced_parameters"] = AdvancedCompressionParameters(
                group_size_fallback_mode=GroupSizeFallbackMode(self.group_size_fallback)
            )
        return nncf.compress_weights(ov_model, **kwargs)


INT8_SYM_CONFIG = WeightCompressionConfig(mode="int8_sym", ratio=1.0, group_size=-1)
INT8_ASYM_CONFIG = WeightCompressionConfig(mode="int8_asym", ratio=1.0, group_size=-1)
# optimum-intel's default 4-bit config, used when --weight-format int4 comes without other compression arguments.
_DEFAULT_4BIT_CONFIG = {"ratio": 1.0, "sym": False, "group_size": 128, "all_layers": None, "group_size_fallback": "ignore"}

# Options that need a calibration dataset, which native exporters do not collect yet.
_DATA_AWARE_OPTIONS = ("dataset", "num_samples", "scale_estimation", "gptq", "lora_correction", "quantization_statistics_path", "smooth_quant_alpha")


def validate_weight_compression_args(args) -> None:
    """Rejects arguments the native exporters do not implement (they need the optimum-intel path)."""
    if args.quant_mode is not None:
        raise ValueError("--quant-mode (full quantization) is not supported by the native genai-cli exporter yet.")
    data_aware = [f"--{name.replace('_', '-')}" for name in _DATA_AWARE_OPTIONS if getattr(args, name) is not None]
    if data_aware:
        raise ValueError(
            f"Data-aware compression ({', '.join(data_aware)}) is not supported by the native genai-cli exporter yet."
        )
    if args.sensitivity_metric not in (None, "weight_quantization_error"):
        raise ValueError(f"--sensitivity-metric {args.sensitivity_metric} requires a dataset, which is not supported yet.")
    if args.weight_format is None and not no_compression_parameter_provided(args):
        raise ValueError(
            "Some compression parameters are provided, but the weight format is not specified. "
            "Please provide it with --weight-format argument."
        )


def weight_compression_config_from_args(args) -> Optional[WeightCompressionConfig]:
    """The compression requested on the command line, None for floating-point (or unspecified) weight formats."""
    weight_format = args.weight_format
    if weight_format in (None, "fp32", "fp16"):
        return None
    if weight_format == "int8":
        return WeightCompressionConfig(mode="int8_sym" if args.sym else "int8_asym", ratio=1.0, group_size=-1)

    params = dict(_DEFAULT_4BIT_CONFIG)
    if not (weight_format == "int4" and no_compression_parameter_provided(args)):
        params.update(
            ratio=args.ratio if args.ratio is not None else 1.0,
            sym=args.sym or False,
            group_size=args.group_size,
            all_layers=args.all_layers,
            group_size_fallback=args.group_size_fallback or _DEFAULT_4BIT_CONFIG["group_size_fallback"],
        )
    if weight_format == "int4":
        mode = "int4_sym" if params["sym"] else "int4_asym"
    else:
        mode = weight_format  # mxfp4, nf4, cb4
    return WeightCompressionConfig(
        mode=mode,
        ratio=params["ratio"],
        group_size=params["group_size"],
        all_layers=params["all_layers"],
        awq=args.awq,
        sensitivity_metric=args.sensitivity_metric,
        backup_precision=args.backup_precision,
        group_size_fallback=params["group_size_fallback"],
    )


def count_weights(ov_model: ov.Model) -> int:
    num = 0
    for op in ov_model.get_ops():
        if op.get_type_name() == "Constant" and op.get_element_type() in (ov.Type.f16, ov.Type.f32, ov.Type.bf16):
            num += int(np.prod(op.get_output_shape(0)))
    return num


def save_submodel(
    ov_model: ov.Model,
    path: Path,
    weight_format: Optional[str],
    compression: Optional[WeightCompressionConfig],
) -> None:
    """Compresses (if requested or if the submodel is large) and saves a submodel.

    ``compression`` is the config this particular submodel gets for an explicit integer weight format. Without a
    weight format, submodels with more than 1B weights are compressed to int8_asym, like optimum-intel does.
    """
    if weight_format is None and count_weights(ov_model) >= _MAX_UNCOMPRESSED_SIZE:
        logger.info(f"{path.name}: the model weights will be quantized to int8_asym.")
        compression = INT8_ASYM_CONFIG
    if compression is not None:
        logger.info(f"{path.name}: applying {compression.mode} weight compression")
        ov_model = compression.compress(ov_model)
    ov.save_model(ov_model, path, compress_to_fp16=weight_format == "fp16")
    del ov_model
    gc.collect()


def copy_model_files(src_dir: Path, output: Path) -> None:
    """Copies configs, processor and tokenizer files of a checkpoint next to the IRs (everything but weights)."""
    for src in src_dir.iterdir():
        if not src.is_file() or src.suffix in _WEIGHT_FILE_SUFFIXES or src.name in _WEIGHT_INDEX_FILES:
            continue
        dst = output / src.name
        # Hub cache files are read-only: copy the content only and replace files of a previous export.
        dst.unlink(missing_ok=True)
        shutil.copyfile(src, dst)


def convert_tokenizer(model_dir: Path, output: Path, chat_template: Optional[str] = None) -> None:
    """Converts the Hugging Face tokenizer to openvino_tokenizer.xml / openvino_detokenizer.xml."""
    from openvino_tokenizers import convert_tokenizer as ov_convert_tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    ov_tokenizer, ov_detokenizer = ov_convert_tokenizer(tokenizer, with_detokenizer=True)
    if chat_template is not None:
        # The processor's chat template (that also lays out images) takes precedence, as in optimum-intel.
        ov_tokenizer.set_rt_info(chat_template, "chat_template")
    ov.save_model(ov_tokenizer, output / OV_TOKENIZER_NAME)
    ov.save_model(ov_detokenizer, output / OV_DETOKENIZER_NAME)


def resolve_model_dir(model: str, cache_dir: Optional[str], allow_patterns: List[str]) -> Path:
    """Local directory with the checkpoint, downloading only the files matching ``allow_patterns`` from the Hub."""
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, cache_dir=cache_dir, allow_patterns=allow_patterns))
