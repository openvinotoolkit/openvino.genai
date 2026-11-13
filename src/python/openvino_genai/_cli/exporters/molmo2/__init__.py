# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native genai-cli exporter of Molmo2 image-text-to-text models; the export is implemented in export.py."""

from openvino_genai._cli.exporters.molmo2.export import Molmo2Exporter

__all__ = ["Molmo2Exporter"]
