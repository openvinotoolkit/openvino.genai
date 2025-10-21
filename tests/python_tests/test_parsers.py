# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
from typing import Optional
from utils.hugging_face import convert_and_save_tokenizer, download_and_convert_model
from utils.ov_genai_pipelines import create_ov_pipeline
from utils.network import retry_request
import numpy as np
import openvino
import pytest
from openvino_genai import Tokenizer, IncrementalParserBase, ParserBase, TextParserStreamer, StreamingStatus, Llama32JsonToolParser, Phi4ReasoningParser, DeepSeekR1ReasoningParser, GenerationConfig
from transformers import AutoTokenizer
import re


@pytest.fixture(scope="module")
def hf_ov_genai_models(request, tmp_path_factory):
    model_id = request.param

    model_dir = tmp_path_factory.getbasetemp() / model_id.replace("/", "_")
    model_dir.mkdir(exist_ok=True, parents=True)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    convert_and_save_tokenizer(hf_tokenizer, model_dir)

    genai_tokenizer = Tokenizer(model_dir)
    return hf_tokenizer, genai_tokenizer


@pytest.mark.precommit
@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    ["katuni4ka/tiny-random-phi3"],  # this tokenizer is used as a stub only
    indirect=True
)
@pytest.mark.parametrize("answer", [
    "<think>\nOkay, the user is asking for the answer to 2 + 1.</think>\n\nThe answer to 2 + 1 is \boxed{3}.",

    (
        "<think>\nOkay, the user is asking for the answer to 2 + 1. Let me make sure I understand "
        "the question correctly. They want a short answer, so I shouldn't overcomplicate things. "
        "Basic addition here. Two plus one equals three. Yeah, that's straightforward. I need to "
        "respond with the answer inside a box using the specified format. Let me double-check the "
        "arithmetic to avoid any mistakes. Yep, 2 + 1 is definitely 3. Alright, time to put it in "
        "the box.\n</think>\n\nThe answer to 2 + 1 is \boxed{3}."
    ),
])
def test_phi4_reason_parser_1(hf_ov_genai_models, answer):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models
    
    stream_string = re.split(r"(\s+)", answer)
    
    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            msg.update(message)
            return StreamingStatus.RUNNING
    streamer = CustomStreamer(genai_tokenizer, parsers=[Phi4ReasoningParser()])
    
    msg = {}
    for subword in stream_string:
        streamer._write(subword)

    think_content = answer.split("</think>")[0].replace("<think>", "")
    content = answer

    assert msg['reasoning_content'] == think_content
    assert msg['content'] == content


@pytest.mark.precommit
@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    ["katuni4ka/tiny-random-phi3"],
    indirect=True
)
@pytest.mark.parametrize("split_answer", [
    ["<th", "ink>", "\nOkay, ", "the user is asking", " for the ", "answer ", "to 2 + 1.", "</think>", "\n\nThe answer ", "to", "2 ", "+ ", "1 ", "is ", "\boxed{3}."],
    ["<think>", "\nOkay, ", "the user is asking", " for the ", "answer ", "to 2 + 1.", "</th", "ink>", "\n\nThe answer ", "to", "2 ", "+ ", "1 ", "is ", "\boxed{3}."],
    ["<t", "h", "ink>", "\nOkay, ", "the user is asking", " for the ", "answer ", "to 2 + 1.", "</th", "ink>", "\n\nThe answer ", "to", "2 ", "+ ", "1 ", "is ", "\boxed{3}."],
    
    # check that if thinking opening and closing tags are passed in a single subword, it is still parsed correctly
    ["<think>\nOkay, the user is asking for the answer to 2 + 1.</think>\n\nThe answer to 2 + 1 is \boxed{3}."]
])
def test_phi4_reason_parser_2(hf_ov_genai_models, split_answer):
    # check that if thinking opening and closing tags are in the middle of the subword, it is still parsed correctly
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models
    
    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            msg.update(message)
            return StreamingStatus.RUNNING
    streamer = CustomStreamer(genai_tokenizer, parsers=[Phi4ReasoningParser()])
    
    msg = {}
    for subword in split_answer:
        streamer._write(subword)

    think_content = (''.join(split_answer)).split("</think>")[0].replace("<think>", "")
    content = ''.join(split_answer)

    assert msg['reasoning_content'] == think_content
    assert msg['content'] == content


@pytest.mark.precommit
@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    ["katuni4ka/tiny-random-phi3"],
    indirect=True
)
def test_final_parser_llama_32_json(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    json_str = '{"type": "function", "function": {"name": "get_weather", "parameters": {"location": "New York, NY", "unit": "celsius"}}}'
    content_json = {
        "content": f"Calling weather API: {json_str}"
    }

    parser = Llama32JsonToolParser()
    parser.parse(content_json)
    assert content_json['tool_calls'][0] == json.loads(json_str)


@pytest.mark.precommit
@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    ["katuni4ka/tiny-random-phi3"],
    indirect=True
)
def test_custom_streamer_parser(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    class CustomParser(IncrementalParserBase):
        main_part_started: bool = False

        def parse(self, msg: dict, previous_text: str, delta_text: str, prev_tokens = None, delta_tokens = None) -> str:
            if 'content' not in msg:
                msg['content'] = ''
            if 'main_text' not in msg:
                msg['main_text'] = ''

            if not self.main_part_started and delta_text == '<start>':
                self.main_part_started = True
            elif self.main_part_started and delta_text == '</stop>':
                self.main_part_started = False
            else:
                if self.main_part_started:
                    msg['main_text'] += delta_text
                
            return delta_text

    msg = {}
    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            msg.update(message)
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(genai_tokenizer, parsers=[CustomParser()])

    stream_string = ["Hello", "<start>", " ", "world", " ", "</stop>", "!"]

    for subword in stream_string:
        streamer._write(subword)

    assert msg['main_text'] == ''.join(" world ")

# @pytest.mark.precommit
# @pytest.mark.parametrize(
#     "hf_ov_genai_models", 
#     ["microsoft/Phi-4-mini-reasoning"],
#     # ["katuni4ka/tiny-random-phi3"],
#     indirect=True
# )
# def test_custom_parser_(hf_ov_genai_models):


#     msg = {
#         "content": "<think>This is reasoning.</think> And this is the answer"
#     }
#     parser.parse(msg)

#     assert msg['reasoning_content'] == "This is reasoning."

@pytest.mark.parametrize("model_id", ["microsoft/Phi-4-mini-reasoning"])
@pytest.mark.nightly
def test_custom_parser(tmp_path, model_id):
    _, _, models_path = download_and_convert_model(model_id, padding_side="left")
    pipe = create_ov_pipeline(models_path)
    tok = pipe.get_tokenizer()
    
    class CustomParser(ParserBase):
        def parse(self, msg: dict):
            content = None
            if 'content' in msg:
                content = msg['content']
            if not content:
                return

            # find text between <think> and </think>
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_start != -1 and think_end != -1 and think_end > think_start:
                think_text = content[think_start + len("<think>"):think_end].strip()
                msg['reasoning_content'] = think_text
    
    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            # make whatever you want with message, but it will be accumulated and parsed by parser afterwards
            # accumulated message can be found by get_parsed_message()
            return StreamingStatus.RUNNING
        
    parser = CustomParser()
    config = GenerationConfig()
    config.max_new_tokens = 600
    config.parsers = [parser]

    res = pipe.generate(["Please say \"hello\""], generation_config=config)
    
    # extract manually reasoning content from the parsed result
    content = res.texts[0]
    think_start = content.find("<think>")
    think_end = content.find("</think>")
    if think_start != -1 and think_end != -1 and think_end > think_start:
        think_text = content[think_start + len("<think>"):think_end].strip()
    
    assert 'reasoning_content' in res.parsed[0]
    assert res.parsed[0]['reasoning_content'] != ""
    assert res.parsed[0]['reasoning_content'] == think_text

def test_parsers_2(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models
    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            if "content" in message:
                print(message["content"])
            return StreamingStatus.RUNNING

    streamer = TextParserStreamer(genai_tokenizer, parsers=[DeepSeekR1ReasoningParser()])

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
    extended.insert(0, "")

    for parser in parsers:
        for (prev_subword, subword) in zip(extended, stream_string):
            msg = parser.parse(msg, prev_subword, subword)
    
    assert msg['reasoning_content'] == think_content
    assert msg['content'] == content

# TODO: add when streamer accepts integer tokens

