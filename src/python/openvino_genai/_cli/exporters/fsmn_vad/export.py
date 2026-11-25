# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native export of the FunASR FSMN-VAD model (funasr/fsmn-vad).

``openvino_model`` computes the frame posteriors of T consecutive feature frames of one recording and keeps no state
between calls:

* ``speech`` [1, T, 400]: the 80-bin fbank features stacked by 5 frames (LFR) and normalized with am.mvn.
* ``in_cache0`` ... ``in_cache3`` [1, 128, 19, 1]: the cache of each FSMN layer, zeros at the start of a recording.
* ``logits`` [1, T, 248]: the softmax posteriors of each frame, class 0 is silence.
* ``out_cache0`` ... ``out_cache3`` [1, 128, 19, 1]: the caches to pass as ``in_cache0`` ... with the next frames.

The output directory also holds am.mvn and the FunASR configuration files of the checkpoint, and a config.json that
identifies the model type and keeps the fields of the checkpoint config.json, if any.
"""

import json
import logging
from argparse import Namespace
from pathlib import Path

from openvino_genai._cli.exporters.base import NativeExporter, register_exporter
from openvino_genai._cli.exporters.common import (
    convert_exported_program,
    copy_model_files,
    resolve_model_dir,
    save_submodel,
    torch_export,
    weight_compression_config_from_args,
)

logger = logging.getLogger(__name__)

_CHECKPOINT_FILE = "model.pt"
_CMVN_FILE = "am.mvn"
_CONFIG_FILE = "config.yaml"
_ALLOW_PATTERNS = [_CHECKPOINT_FILE, _CMVN_FILE, _CONFIG_FILE, "configuration.json", "config.json"]

_FUNASR_MODEL = "FsmnVADStreaming"
_FUNASR_FRONTEND = "WavFrontendOnline"
_FUNASR_ENCODER = "FSMN"
# `speech` is defined as the features of this frontend, so checkpoints with another feature extraction are rejected.
_FRONTEND_CONF = {
    "fs": 16000,
    "window": "hamming",
    "n_mels": 80,
    "frame_length": 25,
    "frame_shift": 10,
    "dither": 0.0,
    "lfr_m": 5,
    "lfr_n": 1,
}

_INPUT_NAMES = ["speech", *(f"in_cache{i}" for i in range(4))]
_OUTPUT_NAMES = ["logits", *(f"out_cache{i}" for i in range(4))]


def _config_mismatches(name: str, actual, expected: dict) -> list[str]:
    if not isinstance(actual, dict):
        return [f"{name} is {actual!r}, expected a mapping"]
    missing = object()
    keys = sorted(expected.keys() | actual.keys())
    return [
        f"{name}.{key} is {actual.get(key, '<missing>')!r}, expected {expected.get(key, '<absent>')!r}"
        for key in keys
        if actual.get(key, missing) != expected.get(key, missing)
    ]


@register_exporter
class FSMNVADExporter(NativeExporter):
    MODEL_TYPE = "fsmn_vad"
    TASKS = ("voice-activity-detection",)
    TRANSFORMERS_VERSION = None

    @classmethod
    def validate_args(cls, args: Namespace) -> None:
        super().validate_args(args)
        if args.variant is not None:
            raise ValueError(f"--variant is not supported for `{cls.MODEL_TYPE}` models.")
        cls._validate_checkpoint(resolve_model_dir(args.model, args.cache_dir, _ALLOW_PATTERNS))

    @classmethod
    def _validate_checkpoint(cls, model_dir: Path) -> None:
        """Checks the checkpoint files and the FunASR settings the exported model depends on."""
        missing = [name for name in (_CHECKPOINT_FILE, _CMVN_FILE, _CONFIG_FILE) if not (model_dir / name).is_file()]
        if missing:
            raise ValueError(
                f"`{cls.MODEL_TYPE}` models are exported from the FunASR checkpoint files {_CHECKPOINT_FILE}, "
                f"{_CMVN_FILE} and {_CONFIG_FILE}; {model_dir} has no {', '.join(missing)}"
            )
        import yaml

        from openvino_genai._cli.exporters.fsmn_vad.model import ENCODER_CONF

        config_path = model_dir / _CONFIG_FILE
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as error:
            raise ValueError(f"Cannot parse {config_path}: {error}") from error
        if not isinstance(config, dict):
            raise ValueError(f"{config_path} is not a FunASR model configuration")
        mismatches = [
            f"{key} is {config.get(key)!r}, expected {expected!r}"
            for key, expected in (
                ("model", _FUNASR_MODEL),
                ("frontend", _FUNASR_FRONTEND),
                ("encoder", _FUNASR_ENCODER),
            )
            if config.get(key) != expected
        ]
        mismatches += _config_mismatches("frontend_conf", config.get("frontend_conf"), _FRONTEND_CONF)
        mismatches += _config_mismatches("encoder_conf", config.get("encoder_conf"), ENCODER_CONF)
        if mismatches:
            raise ValueError(f"{config_path} is not a supported FSMN-VAD configuration: {'; '.join(mismatches)}")

    def export(self) -> None:
        import torch
        from torch.export import Dim

        from openvino_genai._cli.exporters.fsmn_vad import model as fsmn_vad

        args = self.args
        model_dir = resolve_model_dir(args.model, args.cache_dir, _ALLOW_PATTERNS)
        self._validate_checkpoint(model_dir)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {model_dir / _CHECKPOINT_FILE}")
        encoder = fsmn_vad.load_encoder(model_dir / _CHECKPOINT_FILE)
        speech = torch.zeros(1, 30, fsmn_vad.ENCODER_CONF["input_dim"])
        caches = [torch.zeros(fsmn_vad.CACHE_SHAPE) for _ in range(4)]
        # torch 2.6 cannot solve the shape guards of an unbounded frame dimension. The bound only applies to tracing:
        # the frame dimension of the OpenVINO model is unbounded.
        frames = Dim("frames", max=1_000_000)
        ep = torch_export(encoder, args=(speech, *caches), dynamic_shapes=({1: frames}, None, None, None, None))
        ov_model = convert_exported_program(ep, _INPUT_NAMES, _OUTPUT_NAMES)
        save_submodel(
            ov_model, output / "openvino_model.xml", args.weight_format, weight_compression_config_from_args(args)
        )

        copy_model_files(model_dir, output)
        source_config = model_dir / "config.json"
        config = json.loads(source_config.read_text(encoding="utf-8")) if source_config.is_file() else {}
        config["model_type"] = self.MODEL_TYPE
        output_config = output / "config.json"
        output_config.unlink(missing_ok=True)
        output_config.write_text(json.dumps(config, indent=2) + "\n")
        logger.info(f"Model exported to {output}")
