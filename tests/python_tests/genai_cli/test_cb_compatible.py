# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checks that the models exported by genai-cli can be served with continuous batching."""

from collections import Counter
from pathlib import Path

import openvino as ov
import pytest
from openvino._offline_transformations import paged_attention_transformation

from genai_cli.utils import EXPORTS, SUPPORTED_ARCHITECTURES, download, export

# The submodels of each model type whose stateful ScaledDotProductAttention blocks the SDPA -> PagedAttention
# transformation converts. Submodels without a key/value cache have no state to page, the transformation rejects them.
CB_COMPATIBLE_SUBMODELS = {
    "molmo2": ("openvino_language_model.xml",),
}

# The inputs through which a PagedAttention graph receives the paged cache and its layout.
_PAGED_ATTENTION_INPUTS = (
    "past_lens",
    "subsequence_begins",
    "block_indices",
    "block_indices_begins",
    "max_context_len",
)


def test_every_native_exporter_has_cb_compatible_submodels():
    assert set(CB_COMPATIBLE_SUBMODELS) == set(SUPPORTED_ARCHITECTURES)


@pytest.mark.parametrize("model_type, task", EXPORTS, ids=[f"{model_type}-{task}" for model_type, task in EXPORTS])
def test_cb_compatible(tmp_path: Path, model_type: str, task: str):
    export(download(model_type), tmp_path, model_type, ["--task", task])
    core = ov.Core()
    for submodel in CB_COMPATIBLE_SUBMODELS[model_type]:
        model = core.read_model(tmp_path / submodel)
        num_sdpa = Counter(op.get_type_name() for op in model.get_ops())["ScaledDotProductAttention"]
        assert num_sdpa > 0, f"{submodel}: no ScaledDotProductAttention to convert"

        paged_attention_transformation(model)

        ops = Counter(op.get_type_name() for op in model.get_ops())
        assert ops["ScaledDotProductAttention"] == 0, f"{submodel}: SDPA blocks left unconverted"
        assert ops["PagedAttentionExtension"] == num_sdpa, f"{submodel}: not every SDPA became a PagedAttention block"
        # The paged cache replaces the stateful one entirely. The transformation removes the state operations
        # but leaves their variables registered on the model, so the operations are checked.
        assert ops["ReadValue"] + ops["Assign"] == 0, f"{submodel}: key/value cache state left in the graph"

        input_names = {name for port in model.inputs for name in port.get_names()}
        expected_inputs = [
            *_PAGED_ATTENTION_INPUTS,
            *(f"{cache}.{layer}" for layer in range(num_sdpa) for cache in ("key_cache", "value_cache")),
        ]
        assert not (missing := [name for name in expected_inputs if name not in input_names]), (
            f"{submodel}: missing PagedAttention inputs {missing}"
        )

        # The converted graph has to be one a device can actually run.
        core.compile_model(model, "CPU")
