# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
 
import os
import pytest
import sys

from conftest import logger, SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_JS_DIR, MODELS
from test_utils import run_sample
    
class TestBeamSearchCausalLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("Qwen2-0.5B-Instruct", "你好！", marks=pytest.mark.skipif(sys.platform == "win32", reason="Chinese input failed on Windows")),
            pytest.param("Qwen2-0.5B-Instruct-GGUF", "你好！", marks=pytest.mark.skipif(sys.platform == "win32", reason="Chinese input failed on Windows")),
            pytest.param("phi-1_5", "69", marks=pytest.mark.skipif(sys.platform == "win32", reason="Subprocess returned non-zero exit status 3221225477 on Windows")),
        ],
        indirect=["convert_model"],
    )
    def test_sample_beam_search_causal_lm(self, convert_model, sample_args):
        if sys.platform == 'darwin':
            pytest.xfail("Ticket 173586")
        # C++ test
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'beam_search_causal_lm')
        cpp_command = [cpp_sample, convert_model, f'"{sample_args}"']
        cpp_result = run_sample(cpp_command)

        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/beam_search_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model, f'"{sample_args}"']
        py_result = run_sample(py_command)

        # Test JS sample
        js_sample = os.path.join(SAMPLES_JS_DIR, "text_generation/beam_search_causal_lm.js")
        js_command =['node', js_sample, convert_model, f'"{sample_args}"']
        js_result = run_sample(js_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        assert py_result.stdout == js_result.stdout, "Python and JS results should match"
        

    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model",
        [
            "SmolLM2-135M",
            pytest.param("SmolLM2-135M-GGUF", marks=pytest.mark.skip(reason="Linux and mac failed with chinese input due to CVS-173471, Windows due to CVS-173467")),
        ], indirect=True)
    @pytest.mark.parametrize("sample_args",
        [
            ["Why is the Sun yellow?"],
            ["69"],
            ["Hi"],
            ["return 0"],
            pytest.param(["你好！ 你好嗎？"], marks=pytest.mark.skipif(sys.platform == "win32", reason="Chinese input failed on Windows")),
            pytest.param(["Why is the Sun yellow?", "return 0", "你好！ 你好嗎？"], marks=pytest.mark.skipif(sys.platform == "win32", reason="Chinese input failed on Windows")),
        ],
    )
    def test_sample_beam_search_causal_lm_refs(self, request, convert_model, sample_args):
        if sys.platform == 'darwin':
            pytest.xfail("Ticket 173586")
        # C++ test
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'beam_search_causal_lm')
        cpp_command = [cpp_sample, convert_model] + [f'"{arg}"' for arg in sample_args]
        cpp_result = run_sample(cpp_command)
        cpp_predictions = cpp_result.stdout

        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/beam_search_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model] + [f'"{arg}"' for arg in sample_args]
        py_result = run_sample(py_command)
        py_predictions = py_result.stdout

        # Test JS sample
        js_sample = os.path.join(SAMPLES_JS_DIR, "text_generation/beam_search_causal_lm.js")
        js_command =['node', js_sample, convert_model] + [f'"{arg}"' for arg in sample_args]
        js_result = run_sample(js_command)
        js_predictions = js_result.stdout
        
        # Compare results
        assert py_predictions == cpp_predictions, "Python and C++ results should match"
        assert py_predictions == js_predictions, "Python and JS results should match"
        
        model_name = request.node.callspec.params['convert_model']
        model = MODELS[model_name]

        # some GGUF models return different result than transformers
        if model.get("gguf_filename", None):
            return
        
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(model['name'], local_files_only=True)
        for prompt in sample_args:
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': f'"{prompt}"'}], tokenize=False, add_generation_prompt=True)
            tokenized = tokenizer(f'"{prompt}"', return_tensors='pt', add_special_tokens=False)
        
            for beam in transformers.LlamaForCausalLM.from_pretrained(model['name'], local_files_only=True).generate(**tokenized, num_beam_groups=3, num_beams=15, num_return_sequences=15, diversity_penalty=1.0, max_new_tokens=20, early_stopping=False, length_penalty=1.0, no_repeat_ngram_size=9**9, do_sample=False):
                ref = ': ' + tokenizer.decode(beam[tokenized['input_ids'].numel():], skip_special_tokens=True)
                logger.info(f'Checking for "{ref=}"')
                
                idx = py_predictions.find(ref)
                assert -1 != idx, f'Missing "{ref=}" from predictions'
                py_predictions = py_predictions[:idx] + py_predictions[idx + len(ref):]