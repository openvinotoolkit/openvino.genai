# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datasets
from pathlib import Path
from typing import Any
from .network import retry_request
from huggingface_hub import snapshot_download


def load_dataset_via_snapshot(
    repo_id: str, *args: Any, **kwargs: Any
) -> datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset | datasets.IterableDatasetDict:
    local_path = retry_request(lambda: snapshot_download(repo_id, repo_type="dataset"))
    return datasets.load_dataset(local_path, *args, **kwargs)


def load_parquet_dataset_via_snapshot(
    repo_id: str, data_files: dict[str, str], revision: str | None = None, **kwargs: Any
) -> datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset | datasets.IterableDatasetDict:
    """Download only the parquet files matching `data_files` patterns (split -> repo-relative glob) and load them."""
    local_path = Path(
        retry_request(
            lambda: snapshot_download(
                repo_id, repo_type="dataset", revision=revision, allow_patterns=list(data_files.values())
            )
        )
    )
    local_data_files = {split: (local_path / pattern).as_posix() for split, pattern in data_files.items()}
    return datasets.load_dataset("parquet", data_files=local_data_files, **kwargs)
