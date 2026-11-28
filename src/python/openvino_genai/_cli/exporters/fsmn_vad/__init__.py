# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native genai-cli exporter of the FunASR FSMN-VAD model; the export is implemented in export.py."""

from openvino_genai._cli.exporters.fsmn_vad.export import FSMNVADExporter

__all__ = ["FSMNVADExporter"]
