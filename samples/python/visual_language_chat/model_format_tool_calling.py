#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from datetime import date

import numpy as np
from openvino import Tensor
from openvino_genai import (
    ChatHistory,
    GenerationConfig,
    StreamingStatus,
    StructuredOutputConfig,
    VLMPipeline,
)
from PIL import Image
from pydantic import BaseModel, Field


class GetWeatherArgs(BaseModel):
    city: str = Field(description="City name inferred from the image or prompt.")
    date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Forecast date in YYYY-MM-DD format.",
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city on a specific date.",
            "parameters": GetWeatherArgs.model_json_schema(),
        },
    }
]

TOOL_CHOICE = {"type": "function", "function": {"name": "get_weather"}}


def read_image(path: str) -> Tensor:
    image = Image.open(path).convert("RGB")
    return Tensor(np.array(image))


def streamer(subword):
    print(subword, end="", flush=True)
    return StreamingStatus.RUNNING


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Qwen 3.5 VLM model-format tool call using an image, "
            "OpenAI-style tool definitions, and StructuredOutputConfig.from_model_format."
        )
    )
    parser.add_argument("model_dir", help="Path to the VLM model directory.")
    parser.add_argument("image", help="Path to the image used by the VLM prompt.")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on.")
    parser.add_argument(
        "--prompt",
        default=(
            "Look at the image, identify the city or location if possible, "
            "and call get_weather for tomorrow. If the image has no clear city, use Dublin."
        ),
        help="User prompt to generate a tool call for.",
    )
    args = parser.parse_args()

    pipe = VLMPipeline(args.model_dir, args.device)
    image = read_image(args.image)

    chat_history = ChatHistory()
    chat_history.set_tools(TOOLS)
    chat_history.append(
        {
            "role": "system",
            "content": (
                "You are a helpful visual-language assistant. Use the provided tools when they are needed. "
                f"Today's date is {date.today().isoformat()}."
            ),
        }
    )
    chat_history.append({"role": "user", "content": args.prompt})

    config = GenerationConfig()
    config.max_new_tokens = 200
    config.do_sample = False

    structured_output_config = StructuredOutputConfig.from_model_format(
        "qwen_3_5",
        TOOLS,
        TOOL_CHOICE,
        reasoning=False,
    )

    print(f"Prompt: {args.prompt}\n")

    for use_structured_output in [False, True]:
        if use_structured_output:
            print("Model output with StructuredOutputConfig:")
            config.structured_output_config = structured_output_config
        else:
            print("Model output without StructuredOutputConfig:")
            config.structured_output_config = None

        pipe.generate(chat_history, images=[image], generation_config=config, streamer=streamer)
        print("\n", "=" * 80, "\n")


if "__main__" == __name__:
    main()
