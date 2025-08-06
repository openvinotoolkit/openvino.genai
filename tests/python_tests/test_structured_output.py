import pytest
import json
import openvino_genai as ov_genai

from pydantic import BaseModel, Field
from typing import Literal
from utils.hugging_face import download_and_convert_model
from utils.ov_genai_pipelines import create_ov_pipeline
import re

@pytest.fixture(scope="module")
def ov_pipe(request):
    _, _, models_path = download_and_convert_model(request.param)
    return create_ov_pipeline(models_path)

class Person(BaseModel):
    name: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
    age: int = Field(ge=0, le=128)
    city: Literal["Dublin", "Dubai", "Munich"]

class Transaction(BaseModel):
    id: int = Field(ge=0, le=2**14)
    amount: float = Field(ge=0.0, le=1e6)
    currency: Literal["USD", "EUR", "GBP"]

class RESTAPIResponse(BaseModel):
    status: Literal["success", "error"]
    data: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")

structured_id_models = [
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'katuni4ka/tiny-random-phi3',
]

@pytest.mark.precommit
@pytest.mark.parametrize("ov_pipe", structured_id_models, indirect=True)
@pytest.mark.parametrize("prompt_and_scheme", [
    ("Generate a json about a person.", Person), 
    ("Generate a json about a transaction.", Transaction),
    ("Generate a json about a REST API response.", RESTAPIResponse)
])
def test_structured_output_generation(ov_pipe, prompt_and_scheme):
    prompt, SchemeType = prompt_and_scheme

    structured_output_config = ov_genai.StructuredOutputConfig()
    structured_output_config.json_schema = json.dumps(SchemeType.model_json_schema())

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 100
    gen_config.structured_output_config = structured_output_config

    res_str = ov_pipe.generate(prompt, generation_config=gen_config)

    # If it's invalid it will raise an error.
    SchemeType.model_validate_json(res_str)


@pytest.mark.precommit
@pytest.mark.parametrize("ov_pipe", structured_id_models, indirect=True)
@pytest.mark.parametrize("prompt_and_regex", [
    ("Generate a json about a person.", r'^\{"city":"(Dublin|Dubai|Munich)"\}$'),
    # without regex constraint it generates email letter content, but with the regex it generates an email address string
    ("Generate an email.", r'^[a-zA-Z0-9._%+-]{1,64}@[a-z]{1,64}\.[a-z]{1,10}$'),
    ("Generate a json about a REST API response.", r'^\{"status":"(success|error)"\}$'),
])
def test_structured_regex(ov_pipe, prompt_and_regex):
    prompt, regex_str = prompt_and_regex
    structured_output_config = ov_genai.StructuredOutputConfig()
    structured_output_config.regex = regex_str

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 100
    gen_config.structured_output_config = structured_output_config
    res_str = ov_pipe.generate(prompt, generation_config=gen_config)
    
    assert re.match(regex_str, res_str), f"Output {res_str} does not match regex {regex_str}"
   
@pytest.mark.precommit
@pytest.mark.parametrize("ov_pipe", structured_id_models, indirect=True)
@pytest.mark.parametrize("prompt_and_ebnf", [
    # EBNF grammar for generating a date in the format YYYY-MM-DD
    (
        "Generate a date",
        """
        root ::= date
        date ::= year "-" month "-" day
        year ::= digit digit digit digit
        month ::= digit digit
        day ::= digit digit
        digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        """
    ),
])
def test_structured_ebnf(ov_pipe, prompt_and_ebnf):
    prompt, ebnf_grammar = prompt_and_ebnf
    structured_output_config = ov_genai.StructuredOutputConfig()
    structured_output_config.grammar = ebnf_grammar

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 100
    gen_config.structured_output_config = structured_output_config

    res_str = ov_pipe.generate(prompt, generation_config=gen_config)

    # Basic checks for the generated format
    # Currently there is not general way to validate EBNF output,
    # so we will just check if it matches the expected date format.
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", res_str), f"Output {res_str} does not match date format"


@pytest.mark.precommit
@pytest.mark.parametrize(
    "ov_pipe", [model_id for model_id in structured_id_models if "random" not in model_id], indirect=True
)
@pytest.mark.parametrize("prompt_and_structural_tag", [
    (
        "Repeat the word 'function'",
        ov_genai.StructuralTagItem(
            begin="function",
            schema=json.dumps(RESTAPIResponse.model_json_schema()),
            end="</function>"
        )
    ),
])
def test_structural_tags(ov_pipe, prompt_and_structural_tag):
    prompt, structural_tag = prompt_and_structural_tag
    structured_output_config = ov_genai.StructuredOutputConfig(
        structural_tags_config=ov_genai.StructuralTagsConfig(
            structural_tags=[structural_tag],
            triggers=["function"],
        )
    )
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 100
    gen_config.do_sample = False
    gen_config.structured_output_config = structured_output_config

    res_str = ov_pipe.generate(prompt, generation_config=gen_config)

    match = re.search(rf"{structural_tag.begin}(.*?){structural_tag.end}", res_str)
    assert match, f"Output `{res_str}` does not contain structural tag {structural_tag.begin}...{structural_tag.end}"
    RESTAPIResponse.model_validate_json(match.group(1))
