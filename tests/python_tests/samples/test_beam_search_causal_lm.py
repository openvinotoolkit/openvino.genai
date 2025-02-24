# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import logger, SAMPLES_PY_DIR, SAMPLES_CPP_DIR, MODELS
from test_utils import run_sample
    
class TestBeamSearchCausalLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("Qwen2-0.5B-Instruct", "你好！"),
            pytest.param("phi-1_5", "69"),
            pytest.param("SmolLM2-135M", "69"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_beam_search_causal_lm(self, convert_model, sample_args):
        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/beam_search_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model, sample_args]
        py_result = run_sample(py_command)

        # C++ test
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'beam_search_causal_lm')
        cpp_command = [cpp_sample, convert_model, sample_args]
        cpp_result = run_sample(cpp_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        

    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["SmolLM2-135M"], indirect=True)
    @pytest.mark.parametrize("sample_args",
        [
            ["Why is the Sun yellow?"],
            ["69"],
            ["Hi"],
            ["return 0"],
            ["你好！ 你好嗎？"],
            ["Why is the Sun yellow?", "return 0", "你好！ 你好嗎？"],
        ],
    )
    def test_sample_beam_search_causal_lm_refs(self, request, convert_model, sample_args):
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
        
            for beam in transformers.LlamaForCausalLM.from_pretrained(model['name']).generate(**tokenized, num_beam_groups=3, num_beams=15, num_return_sequences=15, diversity_penalty=1.0, max_new_tokens=20, early_stopping=False, length_penalty=1.0, no_repeat_ngram_size=9**9, do_sample=False):
                ref = ': ' + tokenizer.decode(beam[tokenized['input_ids'].numel():], skip_special_tokens=True)
                logger.info(f'Checking for "{ref=}"')
                
                idx = py_predictions.find(ref)
                assert -1 != idx, f'Missing "{ref=}" from Python predictions'
                py_predictions = py_predictions[:idx] + py_predictions[idx + len(ref):]

                idx = cpp_predictions.find(ref)
                assert -1 != idx, f'Missing "{ref=}" from C++ predictions'
                cpp_predictions = cpp_predictions[:idx] + cpp_predictions[idx + len(ref):]
