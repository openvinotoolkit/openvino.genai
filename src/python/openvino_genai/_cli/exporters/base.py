# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from openvino_genai._cli.exporters.common import validate_weight_compression_args


class NativeExporter:
    """Converts one model type from a Hugging Face checkpoint to the OpenVINO IRs used by OpenVINO GenAI."""

    MODEL_TYPE: str
    # Tasks the exporter produces; the first one is used for `--task auto`.
    TASKS: tuple[str, ...]
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
        # OpenVINO GenAI pipelines need stateful language models (they always feed beam_idx), and the native exporters
        # build the graphs themselves, so optimum-intel's model customization options do not apply.
        unsupported = {
            "--disable-stateful": args.disable_stateful,
            "--model-kwargs": args.model_kwargs is not None,
            "--pad-token-id": args.pad_token_id is not None,
        }
        rejected = [option for option, used in unsupported.items() if used]
        if rejected:
            raise ValueError(
                f"{', '.join(rejected)} {'is' if len(rejected) == 1 else 'are'} not supported by the native "
                f"`{cls.MODEL_TYPE}` exporter."
            )
        validate_weight_compression_args(args)

    def export(self) -> None:
        raise NotImplementedError


# Native exporters by model type, filled by @register_exporter when their modules are imported.
_REGISTERED_EXPORTERS: dict[str, type[NativeExporter]] = {}


def register_exporter(exporter: type[NativeExporter]) -> type[NativeExporter]:
    """Class decorator that makes genai-cli convert ``exporter.MODEL_TYPE`` models with ``exporter``."""
    registered = _REGISTERED_EXPORTERS.get(exporter.MODEL_TYPE)
    if registered is not None and registered is not exporter:
        raise ValueError(
            f"Model type `{exporter.MODEL_TYPE}` already has the native exporter {registered.__qualname__}, "
            f"cannot register {exporter.__qualname__}"
        )
    _REGISTERED_EXPORTERS[exporter.MODEL_TYPE] = exporter
    return exporter


def registered_exporters() -> dict[str, type[NativeExporter]]:
    return dict(_REGISTERED_EXPORTERS)
