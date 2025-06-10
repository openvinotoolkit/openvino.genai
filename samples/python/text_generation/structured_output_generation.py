#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from typing import Literal
import openvino_genai
from pydantic import BaseModel

class Person(BaseModel):
    given_name: str
    surname: str
    age: int
    city: Literal["Dublin", "Dubai", "Munich"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help="Path to the model directory. It should contain the OpenVINO model files.")
    parser.add_argument('prompt', nargs='?', help="Prompt to generate structured output. If not provided, a default prompt will be used.")
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    structured_output_config = openvino_genai.StructuredOutputConfig()
    structured_output_config.json_schema = json.dumps(Person.model_json_schema())
    
    prompt = args.prompt if args.prompt else "Generate a json about a person."
    config.max_new_tokens = 100
    config.repetition_penalty = 2
    config.do_sample = True
    config.structured_output_config = structured_output_config
    print(pipe.generate(prompt, config))

if '__main__' == __name__:
    main()
