import pytest
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, Tuple, List
from openvino_genai import Tokenizer
from common import delete_rt_info, convert_and_save_tokenizer

from pywrapper_for_tests import TextCallbackStreamer

tokenizer_model_ids = [
    "katuni4ka/tiny-random-phi3",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    # ("black-forest-labs/FLUX.1-dev", dict(subfolder="tokenizer")),  # FLUX.1-dev has tokenizer in subfolder
]

# This test need wrapper
prompts = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?',
    "如果您有任何疑问，请联系我们，我们将予以解答。",
    "מחרוזת בדיקה",
    "Multiline\nstring!\nWow!",
]
@pytest.mark.parametrize("model_id", tokenizer_model_ids)
@pytest.mark.precommit
@pytest.mark.parametrize("prompt", prompts)
def test_text_prompts(tmp_path, prompt, model_id):
    model_id, hf_tok_load_params = (model_id[0], model_id[1]) if isinstance(model_id, tuple) else (model_id, {})

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_tok_load_params)
    convert_and_save_tokenizer(hf_tokenizer, tmp_path)
    ov_tokenizer = Tokenizer(tmp_path)

    tokens = ov_tokenizer.encode(prompt=prompt).input_ids.data[0].tolist()
    streamer = TextCallbackStreamer(ov_tokenizer, lambda x: accumulated.append(x))
    accumulated = []
    for token in tokens:
        res = streamer.write(token)
    streamer.end()

    assert ''.join(accumulated) == ov_tokenizer.decode(tokens)

encoded_prompts = [
    [    2,  3479,   990,   122,   254,     9,    70,   498,   655]  # This tokens caused error in Meta-Llama-3-8B-Instruct
]
@pytest.mark.parametrize("model_id", tokenizer_model_ids)
@pytest.mark.precommit
@pytest.mark.parametrize("encoded_prompt", encoded_prompts)
def test_encoded_prompts(tmp_path, encoded_prompt, model_id):
    model_id, hf_tok_load_params = (model_id[0], model_id[1]) if isinstance(model_id, tuple) else (model_id, {})

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_tok_load_params)
    convert_and_save_tokenizer(hf_tokenizer, tmp_path)
    ov_tokenizer = Tokenizer(tmp_path)

    streamer = TextCallbackStreamer(ov_tokenizer, lambda x: accumulated.append(x))
    accumulated = []
    for token in encoded_prompt:
        res = streamer.write(token)
    streamer.end()

    assert ''.join(accumulated) == ov_tokenizer.decode(encoded_prompt)
