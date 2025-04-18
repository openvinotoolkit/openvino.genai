# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from conftest import SAMPLES_CPP_DIR
from test_utils import run_sample

class TestContinuousBatching:
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("TinyLlama-1.1B-Chat-v1.0", ["-n", "5"]),
        ],
        indirect=["convert_model"],
    )
    def test_cpp_tool_accuracy(self, convert_model, sample_args):
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'continuous_batching_accuracy')
        cpp_command =[cpp_sample, '-m', convert_model] + sample_args
        run_sample(cpp_command)
        
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-Chat-v1.0"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["ShareGPT_V3_unfiltered_cleaned_split.json"], indirect=True)
    @pytest.mark.parametrize("sample_args", [["-n", "10", "--cache_size", "1"], ["-n", "10", "--dynamic_split_fuse", "--max_batch_size", "256", "--max_input_len", "256", "--cache_size", "1"]])
    def test_cpp_tool_benchmark(self, convert_model, download_test_content, sample_args):
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'continuous_batching_benchmark')
        cpp_command =[cpp_sample, '-m', convert_model, '--dataset', download_test_content] + sample_args
        run_sample(cpp_command)
        
