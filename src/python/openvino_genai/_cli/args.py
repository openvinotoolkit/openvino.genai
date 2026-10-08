# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Arguments of ``genai-cli export``.

The options mirror ``optimum-cli export openvino`` one to one, so that a command line written for
optimum-intel works with genai-cli unchanged (and is forwarded verbatim when genai-cli falls back
to optimum-intel).
"""

import json
from argparse import ArgumentParser
from pathlib import Path


def _default_cache_dir() -> str:
    try:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

        return HUGGINGFACE_HUB_CACHE
    except ImportError:
        return str(Path.home() / ".cache" / "huggingface" / "hub")


def add_export_openvino_args(parser: ArgumentParser) -> None:
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output", type=Path, help="Path indicating the directory where to store the generated OV model."
    )
    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred from the model's "
            "metadata or files. For tasks that generate text, add the `xxx-with-past` suffix to export the model "
            "using past key values caching."
        ),
    )
    optional_group.add_argument(
        "--framework",
        type=str,
        choices=["pt"],
        default="pt",
        help="The framework to use for the export. Defaults to 'pt' for PyTorch. ",
    )
    optional_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allows to use custom code for the modeling hosted in the model repository. This option should only be "
            "set for repositories you trust and in which you have read the code, as it will execute on your local "
            "machine arbitrary code present in the model repository."
        ),
    )
    optional_group.add_argument(
        "--weight-format",
        type=str,
        choices=["fp32", "fp16", "int8", "int4", "mxfp4", "nf4", "cb4"],
        default=None,
        help=(
            "The weight format of the exported model. Option 'cb4' represents a codebook with 16 fixed fp8 values in "
            "E4M3 format."
        ),
    )
    optional_group.add_argument(
        "--quant-mode",
        type=str,
        choices=["int8", "f8e4m3", "f8e5m2", "cb4_f8e4m3", "int4_f8e4m3", "int4_f8e5m2"],
        default=None,
        help="Quantization precision mode. This is used for applying full model quantization including activations. ",
    )
    optional_group.add_argument(
        "--library",
        type=str,
        choices=["transformers", "diffusers", "timm", "sentence_transformers", "open_clip", "kokoro"],
        default=None,
        help=(
            "The library used to load the model before export. If not provided, will attempt to infer the local "
            "checkpoint's library"
        ),
    )
    optional_group.add_argument(
        "--cache_dir",
        type=str,
        default=_default_cache_dir(),
        help=(
            "The path to a directory in which the downloaded model should be cached if the standard cache should not "
            "be used."
        ),
    )
    optional_group.add_argument(
        "--pad-token-id",
        type=int,
        default=None,
        help=(
            "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to "
            "guess it."
        ),
    )
    optional_group.add_argument(
        "--variant",
        type=str,
        default=None,
        help="If specified load weights from variant filename.",
    )
    optional_group.add_argument(
        "--ratio",
        type=float,
        default=None,
        help=(
            "A parameter used when applying 4-bit quantization to control the ratio between 4-bit and 8-bit "
            "quantization. If set to 0.8, 80%% of the layers will be quantized to int4 while 20%% will be quantized "
            "to int8. This helps to achieve better accuracy at the sacrifice of the model size and inference latency. "
            "Default value is 1.0. Note: If dataset is provided, and the ratio is less than 1.0, then data-aware "
            "mixed precision assignment will be applied."
        ),
    )
    optional_group.add_argument(
        "--sym",
        action="store_true",
        default=None,
        help=(
            "Whether to apply symmetric quantization. This argument is related to integer-typed --weight-format and "
            "--quant-mode options. For weight-only quantization (--weight-format) --sym argument does not affect "
            "backup precision."
        ),
    )
    optional_group.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.",
    )
    optional_group.add_argument(
        "--group-size-fallback",
        type=str,
        choices=["error", "ignore", "adjust"],
        default=None,
        help=(
            "Specifies how to handle operations that do not support the given group size. Possible values are: "
            "`error`: raise an error if the given group size is not supported by a node, this is the default "
            "behavior; `ignore`: skip nodes that cannot be compressed with the given group size; `adjust`: adjust "
            "the group size to the maximum supported value for each problematic node, if there is no valid value "
            "greater than or equal to 32, then the node is quantized to the backup precision which is int8_asym by "
            "default. "
        ),
    )
    optional_group.add_argument(
        "--backup-precision",
        type=str,
        choices=["none", "int8_sym", "int8_asym"],
        default=None,
        help=(
            "Defines a backup precision for mixed-precision weight compression. Only valid for 4-bit weight formats. "
            "If not provided, backup precision is int8_asym. 'none' stands for original floating-point precision of "
            "the model weights."
        ),
    )
    optional_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The dataset used for data-aware compression or quantization with NNCF.",
    )
    optional_group.add_argument(
        "--all-layers",
        action="store_true",
        default=None,
        help=(
            "Whether embeddings and last MatMul layers should be compressed to INT4. If not provided an weight "
            "compression is applied, they are compressed to INT8."
        ),
    )
    optional_group.add_argument(
        "--awq",
        action="store_true",
        default=None,
        help=(
            "Whether to apply AWQ algorithm. AWQ improves generation quality of INT4-compressed LLMs. If dataset is "
            "provided, a data-aware activation-based version of the algorithm will be executed, which requires "
            "additional time. Otherwise, data-free AWQ will be applied which relies on per-column magnitudes of "
            "weights instead of activations."
        ),
    )
    optional_group.add_argument(
        "--scale-estimation",
        action="store_true",
        default=None,
        help=(
            "Indicates whether to apply a scale estimation algorithm that minimizes the L2 error between the original "
            "and compressed layers. Providing a dataset is required to run scale estimation."
        ),
    )
    optional_group.add_argument(
        "--gptq",
        action="store_true",
        default=None,
        help=(
            "Indicates whether to apply GPTQ algorithm that optimizes compressed weights in a layer-wise fashion to "
            "minimize the difference between activations of a compressed and original layer."
        ),
    )
    optional_group.add_argument(
        "--lora-correction",
        action="store_true",
        default=None,
        help=(
            "Indicates whether to apply LoRA Correction algorithm. When enabled, this algorithm introduces low-rank "
            "adaptation layers in the model that can recover accuracy after weight compression at some cost of "
            "inference latency."
        ),
    )
    optional_group.add_argument(
        "--sensitivity-metric",
        type=str,
        default=None,
        help=(
            "The sensitivity metric for assigning quantization precision to layers. It can be one of the following: "
            "['weight_quantization_error', 'hessian_input_activation', 'mean_activation_variance', "
            "'max_activation_variance', 'mean_activation_magnitude']."
        ),
    )
    optional_group.add_argument(
        "--quantization-statistics-path",
        type=str,
        default=None,
        help="Directory path to dump/load data-aware weight-only quantization statistics.",
    )
    optional_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The maximum number of samples to take from the dataset for quantization.",
    )
    optional_group.add_argument(
        "--disable-stateful",
        action="store_true",
        help=(
            "Disable stateful converted models, stateless models will be generated instead. Stateful models are "
            "produced by default when this key is not used."
        ),
    )
    optional_group.add_argument(
        "--disable-convert-tokenizer",
        action="store_true",
        help="Do not add converted tokenizer and detokenizer OpenVINO models.",
    )
    optional_group.add_argument(
        "--smooth-quant-alpha",
        type=float,
        default=None,
        help=(
            "SmoothQuant alpha parameter that improves the distribution of activations before MatMul layers and "
            "reduces quantization error. Valid only when activations quantization is enabled."
        ),
    )
    optional_group.add_argument(
        "--model-kwargs",
        type=json.loads,
        help="Any kwargs passed to the model forward, or used to customize the export for a given model.",
    )


def no_compression_parameter_provided(args) -> bool:
    # Except statistics path
    return all(
        it is None
        for it in (
            args.ratio,
            args.group_size,
            args.sym,
            args.all_layers,
            args.dataset,
            args.num_samples,
            args.awq,
            args.scale_estimation,
            args.gptq,
            args.lora_correction,
            args.sensitivity_metric,
            args.backup_precision,
            args.group_size_fallback,
        )
    )
