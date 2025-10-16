# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys
import json

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample
from pydantic import BaseModel, Field
from typing import Literal


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

@pytest.mark.llm
@pytest.mark.samples
@pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-Chat-v1.0"], indirect=True)
@pytest.mark.parametrize("prompt,expected_quantities", [
    ("Give me a json of 2 persons and 1 car.", {"person": 2, "car": 1, "transaction": 0}),
    ("Give me a json for 3 persons and 4 cars.", {"person": 3, "car": 4, "transaction": 0}),
    ("Generate json of one car and 1 transaction.", {"person": 0, "car": 1, "transaction": 1}),
    ("Generate 10000 horses.",  {"person": 0, "car": 0, "transaction": 0}),
])
def test_python_structured_output_sample(convert_model, prompt, expected_quantities):
    py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/structured_output_generation.py")
    py_command = [sys.executable, py_script, convert_model]

    user_input = prompt + "\n"
    result = run_sample(py_command, user_input)
    output = result.stdout

    items_generated = False
    items = []
    # Find the line with "Generated JSON with item quantities:"
    for line in output.splitlines():
        if not items_generated and line.startswith("> Generated JSON with item quantities:"):
            item_quantities = line.split(":", 1)[1].strip()
            items_generated = True
        elif items_generated and line.startswith('{'):
            items.append(line.strip())

    data = json.loads(item_quantities.replace("'", '"'))
    
    # Validate the first stage of the sample where it extracts item quantities from the input prompt
    assert data == expected_quantities, (
        f"Expected item quantities {expected_quantities}, but got {data}"
    )

    # Validate the second stage of the sample where items itself are generated
    items_new_count = {"person": 0, "car": 0, "transaction": 0}
    for item in items:
        item_data = json.loads(item.replace("'", '"'))

        if "name" in item_data:
            Person.model_validate(item_data)
            items_new_count["person"] += 1
        if "model" in item_data:
            Car.model_validate(item_data)
            items_new_count["car"] += 1
        if "currency" in item_data:
            Transaction.model_validate(item_data)
            items_new_count["transaction"] += 1
    
    # In case if the generated correct items, but quantities differ from the expected
    assert items_new_count == expected_quantities, (
        f"Expected item counts {expected_quantities}, but got {items_new_count}"
    )

# TinyLlama-1.1B-Chat-v1.0 is a bit dummy in math,
# It generate wrong answers even for simplest equations like 8x + 7 = -23. 
# But here we just check that the structured output gives output in expected format.
@pytest.mark.llm
@pytest.mark.samples
@pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-Chat-v1.0"], indirect=True)
@pytest.mark.parametrize("prompt,final_answer", [
    ("Solve the equation 8x + 7 = -23 step by step.", "x = 7 or x = 8"),
    ("Solve the equation 18x + 7 - 8 = 0 step by step.", "x = 0 or x = -3"),
])
def test_cpp_structured_output_sample(convert_model, prompt, final_answer):
    if sys.platform == 'darwin':
            pytest.xfail("Ticket 173586")
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, "structured_output_generation")
    cpp_command = [cpp_sample, convert_model]

    user_input = prompt + "\n"
    cpp_result = run_sample(cpp_command, user_input)
    output = cpp_result.stdout
    
    res_json = json.loads(output.split('> ')[1].replace('\'', '"').replace('\n----------\n',''))
    assert 'steps' in res_json and len(res_json['steps']) > 0
    assert 'explanation' in res_json['steps'][0]
    assert 'output' in res_json['steps'][0]
    assert res_json['final_answer'] == final_answer
