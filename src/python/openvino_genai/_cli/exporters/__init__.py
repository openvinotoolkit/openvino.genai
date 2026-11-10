# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Models that genai-cli converts natively (torch.export + OpenVINO, without optimum-intel).

An exporter is a module of this package with a NativeExporter subclass decorated with @register_exporter; adding the
module is enough for genai-cli to use it.
"""

import importlib
import pkgutil

from openvino_genai._cli.exporters.base import NativeExporter, registered_exporters

# Modules of this package that hold the exporter infrastructure, not exporters.
_INFRASTRUCTURE_MODULES = {"base", "common"}


def _import_exporter_modules() -> None:
    """Imports every exporter module of the package, so that their @register_exporter decorators run."""
    for module in pkgutil.iter_modules(__path__):
        if module.name not in _INFRASTRUCTURE_MODULES:
            importlib.import_module(f"{__name__}.{module.name}")


def get_native_exporter(model_type: str | None, library: str | None) -> type[NativeExporter] | None:
    if library not in (None, "transformers"):
        return None
    _import_exporter_modules()
    return registered_exporters().get(model_type)
