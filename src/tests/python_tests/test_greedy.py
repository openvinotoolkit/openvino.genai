# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def test_tiny_llama():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    max_new_tokens = 32
    prompt = 'table is made of'

    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    hf_encoded_output = model.generate(encoded_prompt, max_new_tokens=max_new_tokens, do_sample=False)
    hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])
    print(f'hf_output: {hf_output}')

    import sys
    sys.path.append('build-Debug/src/python-bindings')
    import py_generate_pipeline as genai
    
    pipe = genai.LLMPipeline('text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/')
    ov_output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    print(f'ov_output: {ov_output}')

    assert hf_output == ov_output

if __name__ == '__main__':
    test_tiny_llama()
