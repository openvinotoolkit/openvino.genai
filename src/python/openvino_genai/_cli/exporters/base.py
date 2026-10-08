# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from typing import Tuple

from openvino_genai._cli.exporters.common import validate_weight_compression_args


class NativeExporter:
    """Converts one model type from a Hugging Face checkpoint to the OpenVINO IRs used by OpenVINO GenAI."""

    MODEL_TYPE: str
    # Tasks the exporter produces; the first one is used for `--task auto`.
    TASKS: Tuple[str, ...]
    # The transformers version the model is exported with, provided by a uv overlay when not installed.
    TRANSFORMERS_VERSION: str
    # Whether the modeling code is only available as remote code in the model repository.
    REQUIRES_REMOTE_CODE: bool = False

    def __init__(self, args: Namespace):
        self.args = args

    @classmethod
    def validate_args(cls, args: Namespace) -> None:
        """Raises ValueError for options this exporter does not support. Runs before any overlay is created."""
        if args.task not in ("auto", *cls.TASKS):
            raise ValueError(
                f"Task `{args.task}` is not supported for `{cls.MODEL_TYPE}` models, supported tasks: {list(cls.TASKS)}"
            )
        if args.framework != "pt":
            raise ValueError(f"Only the PyTorch framework is supported, got `{args.framework}`")
        if cls.REQUIRES_REMOTE_CODE and not args.trust_remote_code:
            raise ValueError(
                f"`{cls.MODEL_TYPE}` models are implemented by custom code in the model repository. Please read that "
                "code and pass --trust-remote-code to allow running it."
            )
        validate_weight_compression_args(args)

    def export(self) -> None:
        raise NotImplementedError
