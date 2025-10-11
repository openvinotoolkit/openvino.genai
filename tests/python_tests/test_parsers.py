# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
from typing import Optional

import numpy as np
import openvino
import pytest
from openvino_genai import Tokenizer, IncrementalParserBase, ParserBase, TextParserStreamer
from transformers import AutoTokenizer
from utils.hugging_face import convert_and_save_tokenizer, download_and_convert_model


@pytest.fixture(scope="module")
def hf_ov_genai_models(request, tmp_path_factory):
    model_id, args = request.param
    tok_load_properties = {"add_second_input": args.pop("add_second_input")} if "add_second_input" in args else {}
    
    hf_args = args.copy()  # to overcome mutable default argument side effects
    if "padding_side" in hf_args and hf_args["padding_side"] is None:
        # HF does not accept None.
        # Need to remove padding_side and let HF to choose default value,
        hf_args.pop("padding_side")
    else:
        hf_args["truncation_side"] = hf_args["padding_side"]
    model_dir = tmp_path_factory.getbasetemp() / model_id.replace("/", "_")
    model_dir.mkdir(exist_ok=True, parents=True)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_args)
    convert_args = {"number_of_inputs": hf_args.pop("number_of_inputs")} if "number_of_inputs" in hf_args else {}
    convert_and_save_tokenizer(hf_tokenizer, model_dir, **convert_args)

    genai_tokenizer = Tokenizer(model_dir, tok_load_properties)
    return hf_tokenizer, genai_tokenizer


@pytest.mark.precommit
@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    [("katuni4ka/tiny-random-phi3", {"padding_side": "right"})],
    indirect=True
)
def test_parsers_1(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models
    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            if "content" in message:
                print(message["content"])
            return True
    
    streamer = TextParserStreamer(genai_tokenizer, parsers=["DeepSeekR1ReasoningParser"])
    
    msg = {}
    stream_string = [
        "<｜begin▁of▁sentence｜>", "First", ",", " I", " recognize", " that", " the", " question", " is", " asking", 
        " for", " the", " sum", " of", " ", "2", " and", " ", "1", ".\n\n", "I", " know", " that", " addition", 
        " involves", " combining", " two", " numbers", " to", " find", " their", " total", ".\n\n", "Starting", 
        " with", " ", "2", ",", " I", " add", " ", "1", " to", " it", ".\n\n", "2", " plus", " ", "1", " equals", 
        " ", "3", ".\n", "</think>", "\n\n", "**", "Solution", ":", "**\n\n", "To", " find", " the", " sum", 
        " of", " ", "2", " and", " ", "1", " follow", " these", " simple", " steps", ":\n\n", "1", ".", " **", 
        "Start", " with", " the", " number", " ", "2", ".", "**\n", "2", ".", " **", "Add", " ", "1", " to", 
        " it", ".", "**\n", "   \n", "  ", " \\", "[\n", "  "
    ]

    full_str = ''.join(stream_string)
    think_content = full_str.split("</think>")[0]
    content = full_str.split("</think>")[1]

    parsers = streamer.get_parsers()
    
    extended = stream_string[:]
    extended.append("")

    for parser in parsers:
        for (prev_subword, subword) in zip(extended, stream_string):
            msg = parser.parse(msg, prev_subword, subword)
    
    assert msg['reasoning_content'] == think_content
    assert msg['content'] == content

def test_parsers_2(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models
    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            if "content" in message:
                print(message["content"])
            return True
    
    streamer = TextParserStreamer(genai_tokenizer, parsers=["DeepSeekR1ReasoningParser"])
    
    msg = {}
    stream_string = [
        "<｜begin▁of▁sentence｜>", "First", ",", " I", " recognize", " that", " the", " question", " is", " asking", 
        " for", " the", " sum", " of", " ", "2", " and", " ", "1", ".\n\n", "I", " know", " that", " addition", 
        " involves", " combining", " two", " numbers", " to", " find", " their", " total", ".\n\n", "Starting", 
        " with", " ", "2", ",", " I", " add", " ", "1", " to", " it", ".\n\n", "2", " plus", " ", "1", " equals", 
        " ", "3", ".\n", "</think>", "\n\n", "**", "Solution", ":", "**\n\n", "To", " find", " the", " sum", 
        " of", " ", "2", " and", " ", "1", " follow", " these", " simple", " steps", ":\n\n", "1", ".", " **", 
        "Start", " with", " the", " number", " ", "2", ".", "**\n", "2", ".", " **", "Add", " ", "1", " to", 
        " it", ".", "**\n", "   \n", "  ", " \\", "[\n", "  "
    ]

    full_str = ''.join(stream_string)
    think_content = full_str.split("</think>")[0]
    content = full_str.split("</think>")[1]

    parsers = streamer.get_parsers()
    
    extended = stream_string[:]
    extended.append("")

    for parser in parsers:
        for (prev_subword, subword) in zip(extended, stream_string):
            msg = parser.parse(msg, prev_subword, subword)
    
    assert msg['reasoning_content'] == think_content
    assert msg['content'] == content

# TODO: add tests when streamer is called directly instead of manual subsequent calling of parsers.
