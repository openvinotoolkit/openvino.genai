# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datasets
from typing import Any
from .network import retry_request
from huggingface_hub import snapshot_download


def load_dataset_via_snapshot(
    repo_id: str, *args: Any, **kwargs: Any
) -> "datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset | datasets.IterableDatasetDict":
    local_path = retry_request(lambda: snapshot_download(repo_id, repo_type="dataset"))
    return datasets.load_dataset(local_path, *args, **kwargs)
