#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
from datetime import datetime
from pprint import pprint
from typing import ClassVar

from openvino_genai import (
    GenerationConfig,
    LLMPipeline,
    StreamingStatus,
)
from openvino_genai import (
    StructuredOutputConfig as SOC,
)
from pydantic import BaseModel, Field


class ToolRequest(BaseModel):
    @classmethod
    def string_representation(cls) -> str:
        return f'<function_name="{cls.get_name()}">, arguments={list(cls.model_fields)}'

    @classmethod
    def get_name(cls) -> str:
        return cls._name


class WeatherRequest(ToolRequest):
    _name: ClassVar[str] = "get_weather"

    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    date: str = Field(pattern=r"2\d\d\d-[0-1]\d-[0-3]\d", description="Date in YYYY-MM-DD format")


class CurrencyExchangeRequest(ToolRequest):
    _name: ClassVar[str] = "get_currency_exchange"

    from_currency: str = Field(description="Currency to convert from")
    to_currency: str = Field(description="Currency to convert to")
    amount: float = Field(description="Amount to convert")


tools = {tool.get_name(): tool for tool in [WeatherRequest, CurrencyExchangeRequest]}

new_line = "\n"  # to use inside f-string
sys_message = (
    "You are a helpful assistant that can provide weather information and currency exchange rates. "
    f"Today is {datetime.today().strftime('%Y-%m-%d')}. "  # Use the current date in the system message in YYYY-MM-DD format
    "You can respond in natural language, always start your answer with appropriate greeting, "
    "If you need additional information to respond you can request it by calling particular tool with structured JSON. "
    "You can use the following tools:\n"
    f"{new_line.join([tool.string_representation() for tool in tools.values()])}\n"
    "Please, only use the following format for tool calling in your responses:\n"
    '<function="function_name">'
    '{"argument1": "value1", ...}'
    "</function>\n"
    "Use the tool name and arguments as defined in the tool schema.\n"
    "If you don't know the answer, just say that you don't know, but try to call the tool if it helps to answer the question.\n"
)

function_pattern = r'<function="([^"]+)">({.*?})</function>'
function_pattern = re.compile(function_pattern, re.DOTALL)


def parse_tools_from_response(response: str) -> list[ToolRequest]:
    """
    Parse the tool response from the model output.
    The response should be in the format:
    <function="function_name">{"argument1": "value1", ...}</function>
    """
    matches = re.finditer(function_pattern, response)
    return [tools.get(match.group(1)).model_validate_json(match.group(2)) for match in matches]


def streamer(subword):
    print(subword, end="", flush=True)
    return StreamingStatus.RUNNING


def main():
    default_prompt = (
        "What is the weather in London today and in Paris yesterday, and how many pounds can I get for 100 euros?"
    )

    description = (
        "This script demonstrates how to use OpenVINO GenAI with structured tags to generate responses "
        "that include tool calls. It uses a simple LLM pipeline to generate a response based on the provided prompt, "
        "and it parses the tool calls from the response. Available tools are weather and currency exchange."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "model_dir",
        help="Path to the model directory. It should contain the OpenVINO model files.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="Prompt to generate the response.",
    )
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well
    pipe = LLMPipeline(args.model_dir, device)

    print(f"User prompt: {args.prompt}")

    for use_structural_tags in [False, True]:
        print("=" * 80)
        print(f"{'Using structural tags' if use_structural_tags else 'Using no structural tags':^80}")
        print("=" * 80)
        config = GenerationConfig()
        config.max_new_tokens = 300

        pipe.start_chat(sys_message)
        if use_structural_tags:
            config.structured_output_config = SOC(
                structural_tags_config=SOC.TriggeredTags(
                    triggers=["<function="],
                    tags=[
                        SOC.Tag(
                            begin=f'<function="{name}">',
                            content=SOC.JSONSchema(json.dumps(tool.model_json_schema())),
                            end="</function>",
                        )
                        for name, tool in tools.items()
                    ],
                )
            )
            config.do_sample = True
        response = pipe.generate(args.prompt, config, streamer=streamer)
        pipe.finish_chat()
        print("\n" + "-" * 80)

        print("Correct tool calls by the model:")
        pprint(parse_tools_from_response(response))


if "__main__" == __name__:
    main()
