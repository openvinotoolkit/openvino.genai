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
    print("snapshot_download downloaded to:", local_path)
    print(f"datasets.config.HF_DATASETS_CACHE: {datasets.config.HF_DATASETS_CACHE}")
    result = datasets.load_dataset(local_path, *args, **kwargs)
    return result
