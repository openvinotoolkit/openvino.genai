#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from typing import Literal
import openvino_genai
from typing import Literal
from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
    surname: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
    age: int
    city: Literal["Dublin", "Dubai", "Munich"]
    

class Car(BaseModel):
    model: str = Field(pattern=r"^[A-Z][a-z]{1,20} ?[A-Z][a-z]{0,20} ?.?$")
    year: int
    engine_type: Literal["diesel", "petrol", "electric", "hybrid"]


class Transaction(BaseModel):
    id: int = Field(ge=1000, le=10_000_000)
    amount: float
    currency: Literal["EUR", "PLN", "RUB", "AED", "CHF", "GBP", "USD"]


class ItemQuantities(BaseModel):
    person: int = Field(ge=0, le=100)
    car: int = Field(ge=0, le=100)
    transaction: int = Field(ge=0, le=100)


items_map = {"person": Person, "car": Car, "transaction": Transaction}

sys_message = (
    "You generate jsons based on the user's request. You can generate jsons with different types of objects: person, car, transaction. "
    "If user requested different type json fields should remain zero. "
    "Please not that words 'individual', 'person', 'man', 'human', 'woman', 'people', 'inhabitant', 'citizen' are synonyms and can be used interchangeably. "
    "E.g. if user wants 5 houses, then json must be {\"person\": 0, \"car\": 0, \"transactions\": 0}, "
    "if user wand 3 person and 1 house then json must be {\"person\": 3, \"car\": 0, \"transaction\": 0}. "
    "Make sure that json contans numbers that user requested. If user asks specifi attributes, like 'surname', 'model', etc. "
    "ignore this information and generate jsons with the same fields as in the schema. "
)

sys_message_for_items = "Please try to avoid generating the same jsons multiple times."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help="Path to the model directory. It should contain the OpenVINO model files.")
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    structured_output_config = openvino_genai.StructuredOutputConfig()
    config.max_new_tokens = 300

    print("I'm a smart assistant that generates structured output in JSON format."
          "You can ask me to generate information about a person, car, or bank transaction."
          'For example, you can ask: "Please generate jsons for 3 persons and 1 transaction."')

    while True:
        try:
            prompt = input('> ')
        except EOFError:
            break
        pipe.start_chat(sys_message)
        structured_output_config.json_schema = json.dumps(ItemQuantities.model_json_schema())
        config.structured_output_config = structured_output_config
        config.do_sample = False
        res = pipe.generate(prompt, config)
        pipe.finish_chat()
        print(f"Generated JSON with item quantities: {res}")

        config.do_sample = True
        config.temperature = 0.8

        pipe.start_chat(sys_message_for_items)
        generate_has_run = False
        for item, quantity in json.loads(res).items():
            config.structured_output_config.json_schema = json.dumps(items_map[item].model_json_schema())
            for _ in range(quantity):
                generate_has_run = True
                json_strs = pipe.generate(prompt, config)
                print(json.loads(json_strs))
        pipe.finish_chat()
        if not generate_has_run:
            print("No items generated. Please try again with a different request.")


if '__main__' == __name__:
    main()
