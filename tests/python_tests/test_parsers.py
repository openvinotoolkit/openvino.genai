# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import json
from utils.hugging_face import convert_and_save_tokenizer, download_and_convert_model
from utils.ov_genai_pipelines import create_ov_pipeline
import pytest
from openvino_genai import (
    Tokenizer,
    IncrementalParser,
    Parser,
    TextParserStreamer,
    StreamingStatus,
    Llama3JsonToolParser,
    Phi4ReasoningParser,
    Phi4ReasoningIncrementalParser,
    DeepSeekR1ReasoningIncrementalParser,
    GenerationConfig,
    ReasoningIncrementalParser,
)
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import re
from io import StringIO


def concatenate_dicts(dst_dict, src_dict):
    # keys that exist in both dictionaries
    keys = set(dst_dict.keys()).intersection(set(src_dict.keys()))
    for key in keys:
        dst_dict[key] += src_dict[key]

    # keys that exist in src_dict but missing in dst_dict
    missing_keys = set(src_dict.keys()) - set(dst_dict.keys())
    for key in missing_keys:
        dst_dict[key] = src_dict[key]


@pytest.fixture(scope="module")
def hf_ov_genai_models(request, tmp_path_factory):
    model_id = request.param

    model_dir = tmp_path_factory.getbasetemp() / model_id.replace("/", "_")
    model_dir.mkdir(exist_ok=True, parents=True)

    model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
    hf_tokenizer = AutoTokenizer.from_pretrained(model_cached)
    convert_and_save_tokenizer(hf_tokenizer, model_dir)

    genai_tokenizer = Tokenizer(model_dir)
    return hf_tokenizer, genai_tokenizer


@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"],  # this tokenizer is used as a stub only
    indirect=True
)
def test_several_incremental_parsers(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    class CustomReasonParser(IncrementalParser):
        thinking_started: bool = False
        deactivated: bool = False

        def parse(self, message: dict, delta_text: str, delta_tokens=None) -> dict:
            if self.deactivated:
                return delta_text

            if not self.thinking_started and delta_text == "<start>":
                self.thinking_started = True
            elif self.thinking_started and delta_text != "</stop>":
                message["reasoning_content"] = delta_text
            elif self.thinking_started and delta_text == "</stop>":
                self.deactivated = True

            return delta_text

    class IncrementalToolParser(IncrementalParser):
        started_took_call: bool = False
        accumulated_tool_call: StringIO = StringIO()
        deactivated: bool = False

        def parse(self, delta_msg: dict, delta_text: str, delta_tokens=None) -> str:
            if self.deactivated:
                return delta_text

            if delta_text == "{" and not self.started_took_call:
                self.started_took_call = True
                self.accumulated_tool_call.write(delta_text)

                # If not keep took call in resulting string
                # delta_text = ''
            elif self.started_took_call and delta_text == "}":
                self.started_took_call = False
                self.accumulated_tool_call.write(delta_text)
                self.deactivated = True
                delta_msg["tool_calls"] = [json.loads(self.accumulated_tool_call.getvalue())]
                # If not keep took call in resulting string
                # delta_text = ''
            elif self.started_took_call:
                self.accumulated_tool_call.write(delta_text)
            return delta_text

    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            print(message)
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(genai_tokenizer, parsers=[IncrementalToolParser(), CustomReasonParser()])

    stream_string = [
        "Hello",
        "<start>",
        " ",
        "world",
        " ",
        "</stop>",
        "!",
        "{",
        '"func_name": ',
        '"weather", "location": "New York"',
        "}",
    ]
    think_content = " world "
    content = "".join(stream_string)
    tool_call = {"func_name": "weather", "location": "New York"}

    for subword in stream_string:
        streamer._write(subword)

    final_msg = streamer.get_parsed_message()
    assert final_msg["reasoning_content"] == think_content
    assert final_msg["content"] == content
    assert final_msg["tool_calls"][0] == tool_call


@pytest.mark.parametrize(
    "hf_ov_genai_models",
    ["katuni4ka/tiny-random-phi3"],  # this tokenizer is used as a stub only
    indirect=True,
)
@pytest.mark.parametrize(
    "answer",
    [
        "<think>\nOkay, the user is asking for the answer to 2 + 1.</think>\n\nThe answer to 2 + 1 is \boxed{3}.",
        (
            "<think>\nOkay, the user is asking for the answer to 2 + 1. Let me make sure I understand "
            "the question correctly. They want a short answer, so I shouldn't overcomplicate things. "
            "Basic addition here. Two plus one equals three. Yeah, that's straightforward. I need to "
            "respond with the answer inside a box using the specified format. Let me double-check the "
            "arithmetic to avoid any mistakes. Yep, 2 + 1 is definitely 3. Alright, time to put it in "
            "the box.\n</think>\n\nThe answer to 2 + 1 is \boxed{3}."
        ),
    ],
)
def test_incremental_phi4_reason_parser_1(hf_ov_genai_models, answer):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    stream_string = re.split(r"(\s+)", answer)

    # manually accumulate content from streamer
    content = StringIO()

    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            nonlocal content
            content.write(message["content"])
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(genai_tokenizer, parsers=[Phi4ReasoningIncrementalParser()])

    for subword in stream_string:
        streamer._write(subword)

    think_content = answer.split("</think>")[0].replace("<think>", "")

    msg = streamer.get_parsed_message()
    assert msg["reasoning_content"] == think_content
    assert msg["content"] == answer
    assert msg["content"].endswith(content.getvalue())


@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"],  # this tokenizer is used as a stub only
    indirect=True
)
def test_incremental_phi4_reason_integer_token_ids(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    accumulated_message = {}

    class CustomStreamer(TextParserStreamer):
        def write(self, delta_message):
            concatenate_dicts(accumulated_message, delta_message)
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(genai_tokenizer, parsers=[Phi4ReasoningIncrementalParser()])

    answer = "<think>\nOkay, the user is asking for the answer to 2 + 1.</think>\n\nThe answer to 2 + 1 is \boxed{3}."
    encoded_tokens = genai_tokenizer.encode(answer).input_ids.data.tolist()[0]
    for token in encoded_tokens:
        streamer._write([token])
    streamer.end()

    think_content = answer.split("</think>")[0].replace("<think>", "")

    msg = streamer.get_parsed_message()
    assert msg["reasoning_content"] == think_content
    assert msg["content"] == answer
    assert accumulated_message["reasoning_content"] == think_content
    assert answer.endswith(accumulated_message["content"])


@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"],  # this tokenizer is used as a stub only
    indirect=True
)
def test_incremental_integer_token_ids(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    class CustomIncrementalParser(IncrementalParser):
        started_reasoning: bool = False

        def parse(self, delta_message: dict, delta_text: str, delta_tokens=None) -> str:
            if 1 in delta_tokens and not self.started_reasoning:
                self.started_reasoning = True
                delta_message["reasoning_content"] = delta_text
                delta_text = ""
            elif 1 in delta_tokens and self.started_reasoning:
                self.started_reasoning = False
                delta_text = ""
            elif self.started_reasoning:
                delta_message["reasoning_content"] = delta_text
                delta_text = ""

            # Here we are only collecting ordinary text, therefore leave delta_text unchanged.
            delta_message["content"] = delta_text  # will happen under the hood
            return delta_text

    accumulated_message = {}

    class CustomStreamer(TextParserStreamer):
        def write(self, delta_message):
            concatenate_dicts(accumulated_message, delta_message)
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(genai_tokenizer, parsers=[CustomIncrementalParser()])

    # All closing tags </s>, <|/inst|>, <|endoftext|>, etc. in tiny-random-phi3 add strange \x0c\x0c characters
    # so we avoid them in this test.
    answer = "<s>\nOkay, the user is asking for the answer to 2 + 1.<s>The answer to 2 + 1 is 3."
    encoded_tokens = genai_tokenizer.encode(answer, add_special_tokens=False).input_ids.data.tolist()[0]

    for token in encoded_tokens:
        streamer._write([token])
    streamer.end()

    assert accumulated_message["reasoning_content"] == "\nOkay, the user is asking for the answer to 2 + 1"
    assert accumulated_message["content"] == " The answer to 2 + 1 is 3."


@pytest.mark.parametrize(
    "hf_ov_genai_models",
    ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"],
    indirect=True,
)
@pytest.mark.parametrize(
    "split_answer",
    [
        [
            "<th",
            "ink>",
            "\nOkay, ",
            "the user is asking",
            " for the ",
            "answer ",
            "to 2 + 1.",
            "</think>",
            "\n\nThe answer ",
            "to",
            "2 ",
            "+ ",
            "1 ",
            "is ",
            "\\boxed{3}.",
        ],
        [
            "<think>",
            "\nOkay, ",
            "the user is asking",
            " for the ",
            "answer ",
            "to 2 + 1.",
            "</th",
            "ink>",
            "\n\nThe answer ",
            "to",
            "2 ",
            "+ ",
            "1 ",
            "is ",
            "\\boxed{3}.",
        ],
        [
            "<t",
            "h",
            "ink>",
            "\nOkay, ",
            "the user is asking",
            " for the ",
            "answer ",
            "to 2 + 1.",
            "</th",
            "ink>",
            "\n\nThe answer ",
            "to",
            "2 ",
            "+ ",
            "1 ",
            "is ",
            "\\boxed{3}.",
        ],
        # # check that if thinking opening and closing tags are passed in a single subword, it is still parsed correctly
        ["<think>\nOkay, the user is asking for the answer to 2 + 1.</think>\n\nThe answer to 2 + 1 is \\boxed{3}."],
        # check that if there is a false chunk for open/close thinking tag that accumulating reasoning content will not start/stop
        [
            "<think>",
            "\nOkay, ",
            "the user is asking",
            " for the ",
            "answer ",
            "to 2 + 1.",
            "</thi",
            "\n\nThe answer ",
            "nk>",
            "to",
            "2 ",
            "+ ",
            "1 ",
            "is ",
            "\\boxed{3}.",
            "</think>",
        ],
        [
            "<thi",
            "\nOkay, ",
            "nk>",
            "<think>",
            "the user is asking",
            " for the ",
            "answer ",
            "to 2 + 1.",
            "</think>",
            "\n\nThe answer ",
            "nk>",
            "to",
            "2 ",
            "+ ",
            "1 ",
            "is ",
            "\\boxed{3}.",
        ],
    ],
)
def test_incremental_phi4_reason_parser_2(hf_ov_genai_models, split_answer):
    # check that if thinking opening and closing tags are in the middle of the subword, it is still parsed correctly
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    msg_manual = {}

    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            # will be accumulated automatically inside streamer
            concatenate_dicts(msg_manual, message)
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(genai_tokenizer, parsers=[Phi4ReasoningIncrementalParser()])

    for subword in split_answer:
        streamer._write(subword)

    think_content = ("".join(split_answer)).split("</think>")[0].split("<think>")[1]
    content = "".join(split_answer)

    msg = streamer.get_parsed_message()

    assert msg["reasoning_content"] == think_content
    assert msg["content"].endswith(content)  # since msg contains all accumulated content
    assert msg_manual["reasoning_content"] == think_content
    assert msg_manual["content"] == content


@pytest.mark.parametrize("hf_ov_genai_models", ["katuni4ka/tiny-random-phi3"], indirect=True)
@pytest.mark.parametrize(
    "split_answer",
    [
        [
            "<th",
            "\nOkay, ",
            "the user is asking",
            " for the ",
            "answer ",
            "to 2 + 1.",
            "\n\nThe answer ",
            "to",
            "2 ",
            "+ ",
            "1 ",
            "is ",
            "\\boxed{3}.",
        ],
        [
            "<th",
            "\nOkay, ",
            "the user is asking",
            "ink>",
            " for the ",
            "answer ",
            "to 2 + 1.",
            "</think>",
            "\n\nThe answer ",
            "to",
            "2 ",
            "+ ",
            "1 ",
            "is ",
            "\\boxed{3}.",
        ],
    ],
)
def test_incremental_phi4_reason_parser_3(hf_ov_genai_models, split_answer):
    # check that if there is a false chunk of opening/close thinking tag that it will not start/stop the thinking.
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    msg_manual = {}

    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            # will be accumulated automatically inside streamer
            concatenate_dicts(msg_manual, message)
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(genai_tokenizer, parsers=[Phi4ReasoningIncrementalParser()])

    for subword in split_answer:
        streamer._write(subword)

    msg = streamer.get_parsed_message()
    # since no valid thinking tags were provided
    assert "reasoning_content" not in msg
    assert "reasoning_content" not in msg_manual


@pytest.mark.parametrize(
    "answer",
    [
        "<think>\nOkay, the user is asking for the answer to 2 + 1.</think>\n\nThe answer to 2 + 1 is \boxed{3}.",
    ],
)
def test_incremental_phi4_reason_parser_nostreamer(answer):
    # In this test we are calling parser directly without streamer
    parser = Phi4ReasoningIncrementalParser()

    stream_string = re.split(r"(\s+)", answer)

    accumulated_message = {}
    for subword in stream_string:
        delta_message = {}  # msg when the first parser is called should be empty
        parser.parse(delta_message, subword)
        concatenate_dicts(accumulated_message, delta_message)

    think_content = answer.split("</think>")[0].replace("<think>", "")

    assert accumulated_message["reasoning_content"] == think_content


@pytest.mark.parametrize("keep_original_content", [True, False])
@pytest.mark.parametrize("do_reset", [False])
@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"],  # this tokenizer is used as a stub only
    indirect=True
)
@pytest.mark.parametrize(
    "answer",
    [
        "<think>\nOkay, the user is asking for the answer to 2 + 1.</think>\n\nThe answer to 2 + 1 is \boxed{3}.",
    ],
)
def test_reasoning_parser_cut_content(hf_ov_genai_models, answer, keep_original_content, do_reset):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    stream_string = re.split(r"(\s+)", answer)

    msg = {}

    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            concatenate_dicts(msg, message)
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(
        genai_tokenizer,
        parsers=[ReasoningIncrementalParser(expect_open_tag=True, keep_original_content=keep_original_content)],
    )

    num_runs = 2
    for i in range(num_runs):
        if do_reset:
            streamer.reset()

        for subword in stream_string:
            streamer._write(subword)

        think_content = answer.split("</think>")[0].replace("<think>", "")

    if do_reset:
        # If has been reset, check that content is parsed correctly
        assert msg["reasoning_content"] == think_content
        assert msg["content"] == (answer if keep_original_content else "\n\nThe answer to 2 + 1 is \boxed{3}.")
    else:
        # If has not been reset(), then content msg["content"] will continue to accumulate thinking parts from the next runs
        assert msg["content"].find("<think>") >= 0


def test_incremental_deepseek_parser():
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

    full_str = "".join(stream_string)
    think_content = full_str.split("</think>")[0]

    delta_message = {}
    accumulated_message = {}
    parser = DeepSeekR1ReasoningIncrementalParser()
    for subword in stream_string:
        parser.parse(delta_message, subword)
        concatenate_dicts(accumulated_message, delta_message)

    assert accumulated_message["reasoning_content"] == think_content


@pytest.mark.parametrize(
    "hf_ov_genai_models", ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"], indirect=True
)
def test_custom_incremental_parser(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    class CustomParser(IncrementalParser):
        main_part_started: bool = False

        def parse(self, delta_message: dict, delta_text: str, delta_tokens=None) -> str:
            if not self.main_part_started and delta_text == "<start>":
                self.main_part_started = True
            elif self.main_part_started and delta_text == "</stop>":
                self.main_part_started = False
            else:
                if self.main_part_started:
                    delta_message["main_text"] = delta_text
            delta_message["content"] = delta_text
            return delta_text

    accumulated_message = {}

    class CustomStreamer(TextParserStreamer):
        def write(self, delta_message):
            concatenate_dicts(accumulated_message, delta_message)
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(genai_tokenizer, parsers=[CustomParser()])

    stream_string = ["Hello", "<start>", " ", "world", " ", "</stop>", "!"]

    for subword in stream_string:
        streamer._write(subword)
    assert accumulated_message["main_text"] == " world "


@pytest.mark.parametrize(
    "hf_ov_genai_models", ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"], indirect=True
)
def test_final_parser_llama_32_json(hf_ov_genai_models):
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models

    json_str = '{"type": "function", "function": {"name": "get_weather", "parameters": {"location": "New York, NY", "unit": "celsius"}}}'
    content_json = {"content": f"Calling weather API: {json_str}"}

    parser = Llama3JsonToolParser()
    parser.parse(content_json)
    assert content_json["tool_calls"][0] == json.loads(json_str)


@pytest.mark.parametrize("model_id", ["microsoft/Phi-4-mini-reasoning"])
@pytest.mark.nightly
def test_custom_parser(tmp_path, model_id):
    models_path = download_and_convert_model(model_id, padding_side="left").models_path
    pipe = create_ov_pipeline(models_path)

    class CustomParser(Parser):
        def parse(self, msg: dict):
            content = None
            if "content" in msg:
                content = msg["content"]
            if not content:
                return

            # find text between <think> and </think>
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_start != -1 and think_end != -1 and think_end > think_start:
                think_text = content[think_start + len("<think>") : think_end].strip()
                msg["reasoning_content"] = think_text

    parser = CustomParser()
    config = GenerationConfig()
    config.max_new_tokens = 600
    config.parsers = [parser]

    res = pipe.generate(['Please say "hello"'], generation_config=config)

    # extract manually reasoning content from the parsed result
    content = res.texts[0]
    think_start = content.find("<think>")
    think_end = content.find("</think>")
    if think_start != -1 and think_end != -1 and think_end > think_start:
        think_text = content[think_start + len("<think>") : think_end].strip()

    assert "reasoning_content" in res.parsed[0]
    assert res.parsed[0]["reasoning_content"] != ""
    assert res.parsed[0]["reasoning_content"] == think_text


@pytest.mark.parametrize("model_id", ["microsoft/Phi-4-mini-reasoning"])
@pytest.mark.nightly
def test_reset_incremental_parser(tmp_path, model_id):
    models_path = download_and_convert_model(model_id, padding_side="left").models_path
    pipe = create_ov_pipeline(models_path)
    tok = pipe.get_tokenizer()

    class CustomStreamer(TextParserStreamer):
        def write(self, message):
            return StreamingStatus.RUNNING

    streamer = CustomStreamer(tok, parsers=[Phi4ReasoningIncrementalParser()])

    prompt = 'Please say "hello"'
    res = pipe.generate([prompt], max_new_tokens=600, parsers=[Phi4ReasoningParser()])

    # extract manually reasoning content from the parsed result
    content = res.texts[0]
    think_start = content.find("<think>")
    think_end = content.find("</think>")
    if think_start != -1 and think_end != -1 and think_end > think_start:
        think_text = content[think_start + len("<think>") : think_end]

    assert "reasoning_content" in res.parsed[0]
    assert res.parsed[0]["reasoning_content"] != ""
    assert res.parsed[0]["reasoning_content"] == think_text

    res_streamer_1 = pipe.generate([prompt], max_new_tokens=600, streamer=streamer)
    res_streamer_2 = pipe.generate([prompt], max_new_tokens=600, streamer=streamer)
    # Check that results from streamer generation are the same as from non-streamer generation.
    assert res_streamer_1.parsed == res.parsed

    # Also asserts that resetting streamer between generations works correctly.
    assert res_streamer_2.parsed == res.parsed
