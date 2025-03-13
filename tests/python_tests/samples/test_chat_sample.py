# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
 
import os
import pytest
import sys

from conftest import logger, SAMPLES_PY_DIR, SAMPLES_CPP_DIR, MODELS
from test_utils import run_sample
    
class TestChatSample:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-Chat-v1.0"], indirect=True)
    @pytest.mark.parametrize("prompts",
        [
            ['What is 2 + 2?', 'What is the previous answer?', 'Add 1 to it.', 'Subtract 5 from it.', 'Why is the sun yellow?', 'What was my first question?'],
        ],
    )
    def test_chat_sample_refs(self, request, convert_model, prompts):
        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/chat_sample.py")
        py_command = [sys.executable, py_script, convert_model]
        py_result = run_sample(py_command, '\n'.join(prompts))
        py_predictions = py_result.stdout

        # C++ test
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'chat_sample')
        cpp_command = [cpp_sample, convert_model]
        cpp_result = run_sample(cpp_command, '\n'.join(prompts))
        cpp_predictions = cpp_result.stdout
        
        # Compare results
        assert py_predictions == cpp_predictions, "Python and C++ results should match"
        
        model_name = request.node.callspec.params['convert_model']
        model = MODELS[model_name]
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model['name'])
        model = AutoModelForCausalLM.from_pretrained(model['name'])
        
        for prompt in prompts:
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': f'"{prompt}"'}], tokenize=False, add_generation_prompt=True)
            tokenized = tokenizer(f'"{prompt}"', return_tensors='pt', add_special_tokens=False)
        
            for answer in model.generate(**tokenized, max_length=1000, do_sample=False):
                ref = ': ' + tokenizer.decode(answer[tokenized['input_ids'].numel():], skip_special_tokens=True)
                logger.info(f'Checking for "{ref=}"')
                
                idx = py_predictions.find(ref)
                assert -1 != idx, f'Missing "{ref=}" from predictions'
                py_predictions = py_predictions[:idx] + py_predictions[idx + len(ref):]