#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from typing import Any

from openvino_genai import (
    GenerationConfig,
    LLMPipeline,
    StreamingStatus,
    Parser,
    IncrementalParser,
    TextParserStreamer,
    DecodedResults
)

from openvino_genai import (
    StructuredOutputConfig as SOC,
)
from pydantic import BaseModel, Field


def streamer(subword):
    print(subword, end="", flush=True)
    return StreamingStatus.RUNNING


class book_flight_ticket(BaseModel):
    """booking flights"""

    origin_airport_code: str = Field(description="The name of Departure airport code")
    destination_airport_code: str = Field(description="The name of Destination airport code")
    departure_date: str = Field(description="The date of outbound flight")
    return_date: str = Field(description="The date of return flight")


class book_hotel(BaseModel):
    """booking hotel"""

    destination: str = Field(description="The name of the city")
    check_in_date: str = Field(description="The date of check in")
    checkout_date: str = Field(description="The date of check out")


def _recursive_purge_dict_key(d: dict[str, Any], k: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == k and "type" in d.keys():
                del d[key]
            else:
                _recursive_purge_dict_key(d[key], k)


def tool_to_dict(tool: BaseModel, with_description: bool = True) -> dict[str, Any]:
    schema = tool.model_json_schema()
    _recursive_purge_dict_key(schema, "title")
    if not with_description:
        _recursive_purge_dict_key(schema, "description")
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": [tool.__name__]},
            "arguments": schema,
        },
        "required": ["name", "arguments"],
    }


def tools_to_array_schema(*tools: BaseModel) -> str:
    return json.dumps(
        {
            "type": "array",
            "items": {"anyOf": [tool_to_dict(tool, with_description=False) for tool in tools]},
        }
    )


class IncrementalToolCallParser(IncrementalParser):
    """Incremental parser to extract tool calls from the model output.

    Custom parser should be inherited from IncrementalParser and implement 'parse' and 'reset' methods.
    """
    def parse(self, msg: dict, delta_text: str, delta_tokens: list) -> str:
        if 'content' not in msg:
            msg['content'] = ''
        msg['delta_text'] = delta_text

        # Join the previous content with the new delta_text for searching.
        # Do not modify msg['content'] yet, as it will be updated automatically by the streamer.
        content = msg['content'] + delta_text
        
        start_tag = "functools"
        start_index = content.find(start_tag)
        if start_index == -1:
            return delta_text

        msg['generates_tool_call'] = True
        if not content.endswith("}]"):
            return delta_text
        json_part = content[start_index + len(start_tag):]
        try:
            tool_calls = json.loads(json_part)
            msg['tool_calls'] = tool_calls
            msg['generates_tool_call'] = False
            return delta_text
        except json.JSONDecodeError as e:
            return delta_text
    
    def reset(self):
        # This method should be implemented in inherited classes.
        # e.g. self.text_cache = ""
        # But since in this implementation no states were used so do nothing.
        print('Parser state has been reset.')

class CurrentTextParserStreamer(TextParserStreamer):
    """
    A TextParserStreamer that receives parsed dictionary every time new text is generated.

    In order to get get parsed dictionary from the model output, a custom implementation of TextParserStreamer 
    with defined 'write' methods is needed.
    """
    def write(self, msg: dict):
        # If the tool call is not yet complete, continue streaming
        print(msg['delta_text'], end="", flush=True)
        return StreamingStatus.RUNNING


def print_tool_call(answer: DecodedResults):
    for tool_call in answer.parsed[0]['tool_calls']:
        print(f"{tool_call['name']}({', '.join(f'{key}=\"{value}\"' for key, value in tool_call['arguments'].items())})")


# modified system message from:
# https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_phi4_mini.jinja
sys_message = """You are a helpful AI assistant.
You can answer yes or no to questions, or you can chose to call one or more of the provided functions.

Use the following rule to decide when to call a function:
    * if the response can be generated from your internal knowledge, do so, but use only yes or no as the response
    * if you need external information that can be obtained by calling one or more of the provided functions, generate function calls

If you decide to call functions:
    * prefix function calls with functools marker (no closing marker required)
    * all function calls should be generated in a single JSON list formatted as functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
    * follow the provided JSON schema. Do not hallucinate arguments or values. Do not blindly copy values from the provided samples
    * respect the argument type formatting. E.g., if the type is number and format is float, write value 7 as 7.0
    * make sure you pick the right functions that match the user intent
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        help="Path to the model directory. It should contain the OpenVINO model files.",
    )
    args = parser.parse_args()

    pipe = LLMPipeline(args.model_dir, "CPU")
    tokenizer = pipe.get_tokenizer()
    chat_history = [{"role": "system", "content": sys_message}]
    tools = [tool_to_dict(tool) for tool in [book_flight_ticket, book_hotel]]

    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 300
    generation_config.do_sample = True

    user_text_1 = "Do dolphins have fingers?"
    print("User: ", user_text_1)
    chat_history.append({"role": "user", "content": user_text_1})
    model_input = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, tools=tools)
    # same as SOC.Union(SOC.ConstString("yes"), SOC.ConstString("no"))
    yes_or_no_grammar = SOC.ConstString("yes") | SOC.ConstString("no")
    generation_config.structured_output_config = SOC(structural_tags_config=yes_or_no_grammar)
    print("Assistant: ", end="")
    answer = pipe.generate(model_input, generation_config, streamer=streamer)
    chat_history.append({"role": "assistant", "content": answer})
    print()

    user_text_2 = (
        "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , "
        "then book hotel from 2025-12-04 to 2025-12-10 in Paris"
    )
    print("User: ", user_text_2)
    chat_history.append({"role": "user", "content": user_text_2})
    model_input = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, tools=tools)

    start_tool_call_tag = SOC.ConstString(r"functools")
    tools_json = SOC.JSONSchema(tools_to_array_schema(book_flight_ticket, book_hotel))
    tool_call_grammar = start_tool_call_tag + tools_json  # SOC.Concat(start_tool_call_tag, tools_json)
    generation_config.structured_output_config.structural_tags_config = tool_call_grammar

    print("Assistant: ", end="")
    custom_streamer = CurrentTextParserStreamer(pipe.get_tokenizer(), [IncrementalToolCallParser()])
    answer = pipe.generate([model_input], generation_config, streamer=custom_streamer)
    
    print("\n\nThe following tool calls were generated:")
    print_tool_call(answer)

    print()


if __name__ == "__main__":
    main()
