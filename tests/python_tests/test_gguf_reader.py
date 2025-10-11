# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import gc
import sys
from pathlib import Path

import openvino as ov
import openvino_genai as ov_genai

from utils.hugging_face import generation_config_to_hf, download_gguf_model, load_hf_model_from_gguf, load_hf_tokenizer_from_gguf
from utils.ov_genai_pipelines import create_ov_pipeline, get_gguf_pipeline_types
from data.models import get_gguf_model_list


@pytest.mark.parametrize("pipeline_type", get_gguf_pipeline_types())
@pytest.mark.parametrize("model_ids", get_gguf_model_list())
@pytest.mark.precommit
@pytest.mark.skipif(sys.platform == "win32", reason="CVS-174065")
def test_pipelines_with_gguf_generate(pipeline_type, model_ids):
    if sys.platform == 'darwin':
        pytest.skip(reason="168882: Sporadic segmentation fault failure on MacOS.")
    gguf_model_id = model_ids["gguf_model_id"]
    gguf_filename = model_ids["gguf_filename"]
    dynamic_quantization_group_size = model_ids["dynamic_quantization_group_size"]
    prompt = 'Why is the Sun yellow?'

    opt_model = load_hf_model_from_gguf(gguf_model_id, gguf_filename)
    hf_tokenizer = load_hf_tokenizer_from_gguf(gguf_model_id, gguf_filename)
    gc.collect()

    ov_generation_config = ov_genai.GenerationConfig()
    ov_generation_config.max_new_tokens = 30
    ov_generation_config.apply_chat_template = False
    ov_generation_config.set_eos_token_id(hf_tokenizer.eos_token_id)

    inputs = hf_tokenizer(prompt, return_tensors="pt")
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)
    generate_outputs = None
    with torch.no_grad():
        generate_outputs = opt_model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=hf_generation_config, tokenizer=hf_tokenizer)
    del opt_model
    gc.collect()
    prompt_len = 0 if ov_generation_config.echo else input_ids.numel()
    all_text_batch = hf_tokenizer.batch_decode([generated_ids[prompt_len:] for generated_ids in generate_outputs.sequences], skip_special_tokens=True)
    res_string_input_1 = all_text_batch[0]

    gguf_full_path = download_gguf_model(gguf_model_id, gguf_filename)
    ov_pipe_gguf = create_ov_pipeline(gguf_full_path, pipeline_type=pipeline_type, dynamic_quantization_group_size=dynamic_quantization_group_size)
    encoded_result  = ov_pipe_gguf.generate(ov.Tensor(input_ids.numpy()), generation_config=ov_generation_config)
    del ov_pipe_gguf
    gc.collect()
    res_string_input_2 = hf_tokenizer.batch_decode([encoded_result.tokens[0]], skip_special_tokens=True)[0]

    assert res_string_input_1 == res_string_input_2


@pytest.mark.parametrize("pipeline_type", get_gguf_pipeline_types())
@pytest.mark.parametrize("model_ids", get_gguf_model_list())
@pytest.mark.parametrize("enable_save_ov_model", [False, True])
@pytest.mark.parametrize("prompt", [
    'Why is the Sun yellow?', 
    # To check that special tokens are handled correctly.
    '<|endoftext|> <|im_end|>', 
    '<|endoftext|><|endoftext|><|im_end|>', 
    '<|endoftext|> Why the Sky is Blue? <|im_end|>',
])
@pytest.mark.precommit
@pytest.mark.skipif(sys.platform == "win32", reason="CVS-174065")
def test_full_gguf_pipeline(pipeline_type, model_ids, enable_save_ov_model, prompt):
    if sys.platform == 'darwin':
        pytest.skip(reason="168882: Sporadic segmentation fault failure on MacOS.")
    gguf_model_id = model_ids["gguf_model_id"]
    gguf_filename = model_ids["gguf_filename"]
    dynamic_quantization_group_size = model_ids["dynamic_quantization_group_size"]

    if gguf_model_id == "sammysun0711/tiny-random-deepseek-distill-qwen-gguf" and "<|endoftext|>" in prompt:
        pytest.skip(reason="Prompts to test special tokens for this model fail on HF side")
    
    opt_model = load_hf_model_from_gguf(gguf_model_id, gguf_filename)
    hf_tokenizer = load_hf_tokenizer_from_gguf(gguf_model_id, gguf_filename)
    gc.collect()

    # TODO: remove explicit switch-off of bos token
    hf_tokenizer.add_bos_token = False

    ov_generation_config = ov_genai.GenerationConfig()
    ov_generation_config.max_new_tokens = 30
    ov_generation_config.apply_chat_template = False
    ov_generation_config.set_eos_token_id(hf_tokenizer.eos_token_id)

    inputs = hf_tokenizer(prompt, return_tensors="pt")
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    hf_generation_config = generation_config_to_hf(opt_model.generation_config, ov_generation_config)
    generate_outputs = None
    with torch.no_grad():
        generate_outputs = opt_model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=hf_generation_config, tokenizer=hf_tokenizer)
    del opt_model
    gc.collect()
    prompt_len = 0 if ov_generation_config.echo else input_ids.numel()
    all_text_batch = hf_tokenizer.batch_decode([generated_ids[prompt_len:] for generated_ids in generate_outputs.sequences], skip_special_tokens=True)
    res_string_input_1 = all_text_batch[0]

    gguf_full_path = download_gguf_model(gguf_model_id, gguf_filename)
    ov_pipe_gguf = create_ov_pipeline(gguf_full_path, pipeline_type=pipeline_type, enable_save_ov_model=enable_save_ov_model, dynamic_quantization_group_size=dynamic_quantization_group_size)
    res_string_input_2 = ov_pipe_gguf.generate(prompt, generation_config=ov_generation_config)
    
    # Check that eos_token, bos_token string representations are loaded correctly from gguf file
    assert ov_pipe_gguf.get_tokenizer().get_eos_token() == hf_tokenizer.decode([ov_pipe_gguf.get_tokenizer().get_eos_token_id()])
    assert ov_pipe_gguf.get_tokenizer().get_bos_token() == hf_tokenizer.decode([ov_pipe_gguf.get_tokenizer().get_bos_token_id()])

    del ov_pipe_gguf
    gc.collect()

    if enable_save_ov_model:
        gguf_full_path = Path(gguf_full_path)
        ov_pipe_native = create_ov_pipeline(gguf_full_path.parent, pipeline_type=pipeline_type, dynamic_quantization_group_size=dynamic_quantization_group_size)
        res_string_input_3  = ov_pipe_native.generate(prompt, generation_config=ov_generation_config)
        del ov_pipe_native
        gc.collect()
        assert res_string_input_2 == res_string_input_3

    assert res_string_input_1 == res_string_input_2

@pytest.mark.parametrize("pipeline_type", get_gguf_pipeline_types())
@pytest.mark.parametrize("model_ids", [{"gguf_model_id": "Qwen/Qwen3-0.6B-GGUF", "gguf_filename": "Qwen3-0.6B-Q8_0.gguf"}])
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 172335")
@pytest.mark.precommit
@pytest.mark.skipif(sys.platform == "win32", reason="CVS-174065")
def test_full_gguf_qwen3_pipeline(pipeline_type, model_ids):
    # Temporal testing solution until transformers starts to support qwen3 in GGUF format
    # Please refer details in issue: https://github.com/huggingface/transformers/issues/38063
    gguf_model_id = model_ids["gguf_model_id"]
    gguf_filename = model_ids["gguf_filename"]
    prompt = 'Why is the Sun yellow?'

    ov_generation_config = ov_genai.GenerationConfig()
    ov_generation_config.max_new_tokens = 30
    ov_generation_config.apply_chat_template = True
    ov_generation_config.set_eos_token_id(151645)

    # Original GGUF model output (run with transformers >= 4.54.0):
    # <think>\nOkay, the user is asking why the Sun is yellow. Let me start by recalling what I know about the Sun's color.
    # Prompt after applying chat template is identical between HF and GenAI, so the issue is not in chat template.
    # TODO: Investigate output difference for GGUF models. Ticket: TBD
    res_string_input_1 = "\nOkay, the user is asking why the Sun is yellow. Let me start by recalling what I know about the Sun's color. I remember"

    gguf_full_path = download_gguf_model(gguf_model_id, gguf_filename)
    ov_pipe_gguf = create_ov_pipeline(gguf_full_path, pipeline_type=pipeline_type)
    res_string_input_2 = ov_pipe_gguf.generate(prompt, generation_config=ov_generation_config)

    assert res_string_input_1 == res_string_input_2
