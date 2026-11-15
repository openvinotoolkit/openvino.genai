# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native genai-cli exporter of BAGEL models; the export is implemented in export.py."""

from openvino_genai._cli.exporters.bagel.export import BagelExporter

__all__ = ["BagelExporter"]
