# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Models that genai-cli converts natively (torch.export + OpenVINO, without optimum-intel)."""

from typing import Dict, Optional, Type

from openvino_genai._cli.exporters.base import NativeExporter
from openvino_genai._cli.exporters.molmo2 import Molmo2Exporter

NATIVE_EXPORTERS: Dict[str, Type[NativeExporter]] = {
    exporter.MODEL_TYPE: exporter
    for exporter in (Molmo2Exporter,)
}


def get_native_exporter(model_type: Optional[str], library: Optional[str]) -> Optional[Type[NativeExporter]]:
    if library not in (None, "transformers"):
        return None
    return NATIVE_EXPORTERS.get(model_type)
