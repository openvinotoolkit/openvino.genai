import pytest
import json
import openvino_genai as ov_genai

from pydantic import BaseModel
from typing import Literal
from data.models import get_models_list
from utils.hugging_face import download_and_convert_model
from utils.ov_genai_pipelines import create_ov_pipeline

class Person(BaseModel):
    name: str
    age: int
    city: Literal["Dublin", "Dubai", "Munich"]

class Transaction(BaseModel):
    id: int
    amount: float
    currency: Literal["USD", "EUR", "GBP"]

class RESTAPIResponse(BaseModel):
    status: Literal["success", "error"]
    data: str

@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("model_id", [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    "meta-llama/Llama-3.1-8B-Instruct", # TODO: check why ends with ##########
])
@pytest.mark.parametrize("prompt_and_scheme", [
    ("Generate a json about a person.", Person), 
    ("Generate a json about a transaction.", Transaction),
    ("Generate a json about a REST API response.", RESTAPIResponse)
])
def test_structured_output_generation(model_id, prompt_and_scheme):
    prompt, SchemeType = prompt_and_scheme
    opt_model, hf_tokenizer, models_path  = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(models_path)


    structured_output_config = ov_genai.StructuredOutputConfig()
    structured_output_config.json_schema = json.dumps(SchemeType.model_json_schema())

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 100
    gen_config.apply_chat_template = False
    gen_config.structured_output_config = structured_output_config

    res_str = ov_pipe.generate(prompt, generation_config=gen_config)

    # If it's invalid it will raise an error.
    SchemeType.model_validate_json(res_str)

    # If generation is not constrained by json schema, 
    # assert that output is not valid.
    gen_config.structured_output_config = None
    res_str = ov_pipe.generate(prompt, generation_config=gen_config)
    with pytest.raises(ValueError):
        SchemeType.model_validate_json(res_str)