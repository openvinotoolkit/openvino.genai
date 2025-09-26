# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import logger, MODELS, SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR, SAMPLES_JS_DIR
from test_utils import run_sample

class TestGreedyCausalLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("SmolLM-135M", "return 0"),
            pytest.param("SmolLM2-135M-GGUF", "return 0", marks=pytest.mark.skipif(sys.platform == "win32", reason="CVS-173467")),
            pytest.param("Qwen2-0.5B-Instruct", "69"),
            pytest.param("Qwen2-0.5B-Instruct-GGUF", "69", marks=pytest.mark.skipif(sys.platform == "win32", reason="CVS-173467")),
            pytest.param("phi-1_5", "Alan Turing was a"),
            pytest.param("TinyLlama-1.1B-Chat-v1.0", "Alan Turing was a"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_greedy_causal_lm(self, request, convert_model, sample_args):
        if sys.platform == 'darwin':
            pytest.xfail("Ticket 173586")
        prompt = sample_args
        
        # C++ test
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
        cpp_command = [cpp_sample, convert_model, prompt]
        cpp_result = run_sample(cpp_command)
        cpp_predictions = cpp_result.stdout

        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/greedy_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model, prompt]
        py_result = run_sample(py_command)
        py_predictions = py_result.stdout

        # Test C sample
        c_sample = os.path.join(SAMPLES_C_DIR, "greedy_causal_lm_c")
        c_command =[c_sample, convert_model, sample_args]
        c_result = run_sample(c_command)

        # Test JS sample
        js_sample = os.path.join(SAMPLES_JS_DIR, "text_generation/greedy_causal_lm.js")
        js_command =['node', js_sample, convert_model, sample_args]
        js_result = run_sample(js_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
        assert cpp_result.stdout == c_result.stdout, f"Results should match"
        assert c_result.stdout == js_result.stdout, f"Results should match"
                
        model_name = request.node.callspec.params['convert_model']
        model = MODELS[model_name]

        # some GGUF models return different result than transformers
        if model.get("gguf_filename", None):
            return
        
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(model['name'], local_files_only=True)
                
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    
        for output in transformers.AutoModelForCausalLM.from_pretrained(model['name'], local_files_only=True).generate(**tokenized, max_length=100, do_sample=False):
            ref = tokenizer.decode(output[tokenized['input_ids'].numel():], skip_special_tokens=True)
            logger.info(f'Checking for "{ref=}"')

            idx = cpp_predictions.find(ref)
            assert -1 != idx, f'Missing "{ref=}" from predictions'
            cpp_predictions = cpp_predictions[:idx] + cpp_predictions[idx + len(ref):]
