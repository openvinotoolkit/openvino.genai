# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest

from common import run_test_pipeline, get_models_list

scheduler_params_list = [{"num_kv_blocks": 300, "block_size": 16, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256},
                         {"num_kv_blocks": 40, "block_size": 4, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, # test preemption for dynamic_split_fuse
                         {"num_kv_blocks": 40, "block_size": 4, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}] # test preemption for vllm
@pytest.mark.parametrize("scheduler_params", scheduler_params_list)
@pytest.mark.precommit
def test_preemption(tmp_path, scheduler_params):
    run_test_pipeline(tmp_path, "facebook/opt-125m", scheduler_params)