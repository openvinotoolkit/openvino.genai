# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import logger, MODELS, SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestGreedyCausalLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("LaMini-GPT-124M", "test"),
            pytest.param("SmolLM-135M", "return 0"),
            pytest.param("Qwen2.5-0.5B-Instruct", "69"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_greedy_causal_lm(self, convert_model, sample_args):
        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/greedy_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model, sample_args]
        py_result = run_sample(py_command)

        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
        cpp_command =[cpp_sample, convert_model, sample_args]
        cpp_result = run_sample(cpp_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
    
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["phi-1_5", "redpajama-3b-chat"], indirect=True)
    @pytest.mark.parametrize("sample_args", [["Alan Turing was a"]])
    def test_sample_greedy_causal_lm_refs(self, request, convert_model, sample_args):
        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/beam_search_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model] + [f'"{arg}"' for arg in sample_args]
        py_result = run_sample(py_command)
        py_predictions = py_result.stdout

        # C++ test
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'beam_search_causal_lm')
        cpp_command = [cpp_sample, convert_model] + [f'"{arg}"' for arg in sample_args]
        cpp_result = run_sample(cpp_command)
        cpp_predictions = cpp_result.stdout
        
        model_name = request.node.callspec.params['convert_model']
        model = MODELS[model_name]
        
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(model['name'])
        for prompt in sample_args:
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': f'"{prompt}"'}], tokenize=False, add_generation_prompt=True)
            tokenized = tokenizer(f'"{prompt}"', return_tensors='pt', add_special_tokens=False)
        
            for beam in transformers.LlamaForCausalLM.from_pretrained(model['name']).generate(**tokenized, max_length=100, do_sample=False):
                ref = ': ' + tokenizer.decode(beam[tokenized['input_ids'].numel():], skip_special_tokens=True)
                logger.info(f'Checking for "{ref=}"')
                
                idx = py_predictions.find(ref)
                assert -1 != idx, f'Missing "{ref=}" from Python predictions'
                py_predictions = py_predictions[:idx] + py_predictions[idx + len(ref):]

                idx = cpp_predictions.find(ref)
                assert -1 != idx, f'Missing "{ref=}" from C++ predictions'
                cpp_predictions = cpp_predictions[:idx] + cpp_predictions[idx + len(ref):]