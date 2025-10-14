# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR, SAMPLES_JS_DIR
from test_utils import run_sample

class TestBenchmarkGenAI:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, prompt, sample_args",
        [
            pytest.param("TinyLlama-1.1B-Chat-v1.0", "Why is the sun yellow?", ["-nw", "2", "-n", "3", "-mt", "50", "-d", "CPU"]),
        ],
        indirect=["convert_model"],
    )
    def test_py_sample_benchmark_genai(self, convert_model, prompt, sample_args):
        if sys.platform == 'darwin':
            pytest.xfail("Ticket 173586")
        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/benchmark_genai.py")
        py_command = [sys.executable, py_script, '-m', convert_model, '-p', f'"{prompt}"'] + sample_args
        run_sample(py_command)

    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, prompt, sample_args",
        [
            pytest.param("TinyLlama-1.1B-Chat-v1.0", "Why is the sun yellow?", ["--nw", "2", "-n", "3", "--mt", "50", "-d", "CPU"]),
        ],
        indirect=["convert_model"],
    )
    def test_cpp_sample_benchmark_genai(self, convert_model, prompt, sample_args):
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'benchmark_genai')
        cpp_command =[cpp_sample, '-m', convert_model, '-p', f'"{prompt}"'] + sample_args
        run_sample(cpp_command)

    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, prompt, sample_args",
        [
            pytest.param("TinyLlama-1.1B-Chat-v1.0", "Why is the sun yellow?", ["--nw", "2", "-n", "3", "--mt", "50", "-d", "CPU"]),
        ],
        indirect=["convert_model"],
    )
    def test_cpp_sample_benchmark_genai(self, convert_model, prompt, sample_args):
        # Test C sample
        c_sample = os.path.join(SAMPLES_C_DIR, 'benchmark_genai_c')
        c_command =[c_sample, '-m', convert_model, '-p', f'"{prompt}"'] + sample_args
        run_sample(c_command)

    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, prompt, sample_args",
        [
            pytest.param("TinyLlama-1.1B-Chat-v1.0", "Why is the sun yellow?", ["--nw", "2", "-n", "3", "--mt", "50", "-d", "CPU"]),
        ],
        indirect=["convert_model"],
    )
    def test_js_sample_benchmark_genai(self, convert_model, prompt, sample_args):
        if sys.platform == 'darwin':
            pytest.xfail("Ticket 173586")
        # Test JS sample
        js_sample = os.path.join(SAMPLES_JS_DIR, "text_generation/benchmark_genai.js")
        js_command =['node', js_sample, '-m', convert_model, '-p', f'"{prompt}"'] + sample_args
        run_sample(js_command)
