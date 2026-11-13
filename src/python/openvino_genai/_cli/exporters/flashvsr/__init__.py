# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native genai-cli exporter of FlashVSR video super-resolution models; the export is implemented in export.py and the
PyTorch modules in modeling.py."""

from openvino_genai._cli.exporters.flashvsr.export import FlashVSRExporter

__all__ = ["FlashVSRExporter"]
