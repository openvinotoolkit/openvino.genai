import torch
import openvino as ov
from typing import Any


def export_to_ov(
    model: torch.nn.Module,
    example_kwargs: dict[str, Any],
    dynamic_shapes: dict[str, Any] | None = None,
) -> ov.Model:
    model.float()
    exported = torch.export.export(
        model,
        args=(),
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    return ov.convert_model(exported)
