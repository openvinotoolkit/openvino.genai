# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Models that genai-cli converts natively (torch.export + OpenVINO, without optimum-intel).

An exporter is a subpackage of this package named after the model type (e.g. molmo2/, with the export in export.py,
like the per-model directories of transformers), whose __init__.py imports a NativeExporter subclass decorated with
@register_exporter; adding the subpackage is enough for genai-cli to use it.
"""

import importlib
import pkgutil

from openvino_genai._cli.exporters.base import NativeExporter, registered_exporters

# Modules of this package that hold the exporter infrastructure, not exporters.
_INFRASTRUCTURE_MODULES = {"base", "common"}


def _import_exporter_modules() -> None:
    """Imports every exporter subpackage of the package, so that their @register_exporter decorators run."""
    for module in pkgutil.iter_modules(__path__):
        if module.name not in _INFRASTRUCTURE_MODULES:
            importlib.import_module(f"{__name__}.{module.name}")


def get_native_exporter(model_type: str | None, library: str | None) -> type[NativeExporter] | None:
    if library not in (None, "transformers"):
        return None
    _import_exporter_modules()
    return registered_exporters().get(model_type)
