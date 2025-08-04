#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from typing import Any

from openvino_genai import LLMPipeline, GenerationConfig, StructuredOutputConfig as SOC
from pydantic import BaseModel, Field


# Define the schema for available tools

class booking_flight_tickets(BaseModel):
    """booking flights"""
    origin_airport_code: str = Field(description="The name of Departure airport code")
    destination_airport_code: str = Field(description="The name of Destination airport code")
    departure_date: str = Field(description="The date of outbound flight")
    return_date: str = Field(description="The date of return flight")


class booking_hotels(BaseModel):
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
    schema = {"name": {"type": "string", "enum": [tool.__name__]}}
    schema |= tool.model_json_schema()
    schema["arguments"] = schema.pop("properties", {})
    _recursive_purge_dict_key(schema, "title")  # Remove 'title' key if present
    if not with_description:
        _recursive_purge_dict_key(schema, "description")
    return schema


def generate_system_prompt_tools(*tools: BaseModel) -> str:
    """Generate system prompt with available tools"""
    return f"<|tool|>{json.dumps([tool_to_dict(tool) for tool in tools])}</|tool|>"


def tools_to_array_schema(*tools: BaseModel) -> str:
    return json.dumps(
        {
            "type": "array",
            "items": {
                "anyOf": [
                    tool_to_dict(tool, with_description=False)
                    for tool in tools
                ]
            },
        }
    )


# sys_message = "You are a helpful assistant with these tools."
sys_message = """You are a helpful AI assistant.
In addition to plain text responses, you can chose to call one or more of the provided functions.

Use the following rule to decide when to call a function:
    * if the response can be generated from your internal knowledge (e.g., as in the case of queries like "What is the capital of Poland?"), do so
    * if you need external information that can be obtained by calling one or more of the provided functions, generate a function calls
    
If you decide to call functions:
    * prefix function calls with functools marker (no closing marker required)
    * all function calls should be generated in a single JSON list formatted as functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
    * follow the provided JSON schema. Do not hallucinate arguments or values. Do to blindly copy values from the provided samples
    * respect the argument type formatting. E.g., if the type if number and format is float, write value 7 as 7.0
    * make sure you pick the right functions that match the user intent
"""
sys_message += generate_system_prompt_tools(booking_flight_tickets, booking_hotels)

def main():
    pipe = LLMPipeline("/home/apaniuko/cpp/openvino.genai/samples/python/text_generation/phi4_mini_instruct", "CPU")
    tokenizer = pipe.get_tokenizer()
    chat_history = [{"role": "system", "content": sys_message}]
    user_text = (
        "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , "
        "then book hotel from 2025-12-04 to 2025-12-10 in Paris"
    )
    chat_history.append({"role": "user", "content": user_text})
    model_input = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True)
    print(model_input)

    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 300
    generation_config.do_sample = True

    results_without_compound_grammar = pipe.generate(model_input, generation_config)
    print(results_without_compound_grammar)

    any_text = SOC.Regex(r"[^{](.|\n)*/ ")
    start_tool_call_tag = SOC.Regex(r"functools")
    tools_json = SOC.JSONSchema(tools_to_array_schema(booking_flight_tickets, booking_hotels))
    generation_config.structured_output_config = SOC(
        compound_grammar=(any_text | (start_tool_call_tag + tools_json)),
    )

    results_with_compound_grammar = pipe.generate(model_input, generation_config)
    print(results_with_compound_grammar)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('model_dir', help="Path to the model directory. It should contain the OpenVINO model files.")
    # args = parser.parse_args()
    #
    # device = 'CPU'  # GPU can be used as well
    # pipe = LLMPipeline(args.model_dir, device)
    #
    # config = GenerationConfig()
    # config.max_new_tokens = 300
    #
    # print("This is a smart assistant that generates structured output in JSON format. "
    #       "You can ask to generate information about a person, car, or bank transaction. "
    #       'For example, you can ask: "Please generate jsons for 3 persons and 1 transaction."')
    #
    # while True:
    #     try:
    #         prompt = input('> ')
    #     except EOFError:
    #         break
    #     pipe.start_chat(sys_message)
    #     config.structured_output_config = StructuredOutputConfig(
    #         json_schema=json.dumps(ItemQuantities.model_json_schema()))
    #     config.do_sample = False
    #     res = json.loads(pipe.generate(prompt, config))
    #     pipe.finish_chat()
    #     print(f"Generated JSON with item quantities: {res}")
    #
    #     config.do_sample = True
    #     config.temperature = 0.8
    #
    #     pipe.start_chat(sys_message_for_items)
    #     generate_has_run = False
    #     for item, quantity in res.items():
    #         config.structured_output_config = StructuredOutputConfig(
    #             json_schema=json.dumps(items_map[item].model_json_schema()))
    #         for _ in range(quantity):
    #             generate_has_run = True
    #             json_strs = pipe.generate(prompt, config)
    #             print(json.loads(json_strs))
    #     pipe.finish_chat()
    #     if not generate_has_run:
    #         print("No items generated. Please try again with a different request.")


if '__main__' == __name__:
    main()
