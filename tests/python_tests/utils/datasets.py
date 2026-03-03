# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import datasets
from typing import Any
from .hugging_face import retry_request
from huggingface_hub import snapshot_download


def load_dataset_via_snapshot(
    repo_id: str, *args: Any, **kwargs: Any
) -> "datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset | datasets.IterableDatasetDict":
    """Download dataset with snapshot_download, then load from local path.

    In CI snapshot_download stores files under HF_HOME/hub/ on the shared NFS
    mount (proven reliable for model downloads). datasets.load_dataset
    then reads from that local snapshot with cache_dir on local disk,
    avoiding NFS lock issues with filelock on NFS mounts. Ticket: 181288.

    Set HF_DATASETS_LOCAL_CACHE_PATH env variable to redirect builder
    lock/Arrow cache files to local disk (e.g. /tmp/hf_datasets_cache).
    When unset, datasets uses its default cache location.
    """

    local_cache = os.environ.get("HF_DATASETS_LOCAL_CACHE_PATH")
    local_cache_kwargs = {}
    if local_cache is not None:
        local_cache_kwargs["cache_dir"] = local_cache
    local_path = retry_request(lambda: snapshot_download(repo_id, repo_type="dataset"))
    return datasets.load_dataset(local_path, *args, **{**kwargs, **local_cache_kwargs})
