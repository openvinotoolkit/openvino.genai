import json
import re
from typing import Literal

import openvino_genai as ov_genai
import pytest
from openvino_genai import StructuredOutputConfig as SOC
from pydantic import BaseModel, Field
from utils.hugging_face import download_and_convert_model
from utils.ov_genai_pipelines import create_ov_pipeline


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
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "katuni4ka/tiny-random-phi3",
]


@pytest.mark.precommit
@pytest.mark.parametrize("ov_pipe", structured_id_models, indirect=True)
@pytest.mark.parametrize(
    "prompt_and_scheme",
    [
        ("Generate a json about a person.", Person),
        ("Generate a json about a transaction.", Transaction),
        ("Generate a json about a REST API response.", RESTAPIResponse),
    ],
)
@pytest.mark.parametrize("use_compound_grammar", [True, False])
def test_structured_json(ov_pipe, prompt_and_scheme, use_compound_grammar, capfd):
    prompt, SchemeType = prompt_and_scheme

    structured_output_config = ov_genai.StructuredOutputConfig()
    if use_compound_grammar:
        structured_output_config.compound_grammar = SOC.JSONSchema(json.dumps(SchemeType.model_json_schema()))
    else:
        structured_output_config.json_schema = json.dumps(SchemeType.model_json_schema())

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 100
    gen_config.structured_output_config = structured_output_config

    res_str = ov_pipe.generate(prompt, generation_config=gen_config)
    try:
        SchemeType.model_validate_json(res_str)
    except Exception as e:
        pytest.fail(f"Output {res_str} is not valid json schema {SchemeType.model_json_schema()}: {e}")


@pytest.mark.precommit
@pytest.mark.parametrize("ov_pipe", structured_id_models, indirect=True)
@pytest.mark.parametrize(
    "prompt_and_regex",
    [
        ("Generate a json about a person.", r'^\{"city":"(Dublin|Dubai|Munich)"\}$'),
        # without regex constraint it generates email letter content, but with the regex it generates an email address string
        ("Generate an email.", r"^[a-zA-Z0-9._%+-]{1,64}@[a-z]{1,64}\.[a-z]{1,10}$"),
        ("Generate a json about a REST API response.", r'^\{"status":"(success|error)"\}$'),
    ],
)
@pytest.mark.parametrize("use_compound_grammar", [True, False])
def test_structured_regex(ov_pipe, prompt_and_regex, use_compound_grammar):
    prompt, regex_str = prompt_and_regex
    structured_output_config = ov_genai.StructuredOutputConfig()
    if use_compound_grammar:
        structured_output_config.compound_grammar = structured_output_config.Regex(regex_str)
    else:
        structured_output_config.regex = regex_str

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 100
    gen_config.structured_output_config = structured_output_config
    res_str = ov_pipe.generate(prompt, generation_config=gen_config)

    assert re.match(regex_str, res_str), f"Output {res_str} does not match regex {regex_str}"


@pytest.mark.precommit
@pytest.mark.parametrize("ov_pipe", structured_id_models, indirect=True)
@pytest.mark.parametrize(
    "prompt_and_ebnf",
    [
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
        """,
        ),
    ],
)
@pytest.mark.parametrize("use_compound_grammar", [True, False])
def test_structured_ebnf(ov_pipe, prompt_and_ebnf, use_compound_grammar):
    prompt, ebnf_grammar = prompt_and_ebnf
    structured_output_config = ov_genai.StructuredOutputConfig()
    if use_compound_grammar:
        structured_output_config.compound_grammar = structured_output_config.EBNF(ebnf_grammar)
    else:
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
@pytest.mark.parametrize(
    "prompt_and_structural_tag",
    [
        (
            "Repeat the word 'function'",
            ov_genai.StructuralTagItem(
                begin="function", schema=json.dumps(RESTAPIResponse.model_json_schema()), end="</function>"
            ),
        ),
    ],
)
def test_structural_tags_old(ov_pipe, prompt_and_structural_tag):
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


@pytest.mark.precommit
# use only non-random model for stable output in TriggeredTags test
@pytest.mark.parametrize("ov_pipe", ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"], indirect=True)
@pytest.mark.parametrize(
    "prompt,tag,validate",
    [
        pytest.param(
            "",
            """
            {
                "type": "structural_tag",
                "format": {
                    "type": "const_string",
                    "value": "abc"
                }
            }""",
            lambda x: x == "abc",
            id="Raw string structural tag",
        ),
        pytest.param("", SOC.Regex("a*"), lambda x: re.match(r"^a*$", x) is not None, id="Regex"),
        pytest.param(
            "",
            SOC.JSONSchema(json.dumps(RESTAPIResponse.model_json_schema())),
            RESTAPIResponse.model_validate_json,
            id="JSONSchema",
        ),
        pytest.param(
            "",
            SOC.EBNF(
                """
                root ::= date
                date ::= year "-" month "-" day
                year ::= digit digit digit digit
                month ::= digit digit
                day ::= digit digit
                digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
                """
            ),
            lambda x: re.match(r"^\d{4}-\d{2}-\d{2}$", x) is not None,
            id="EBNF",
        ),
        pytest.param("", SOC.ConstString("constant_string"), lambda x: x == "constant_string", id="ConstantString"),
        pytest.param("", SOC.AnyText(), lambda x: len(x) > 0, id="AnyText"),
        pytest.param(
            "",
            SOC.Tag(begin="function", content=SOC.ConstString("..."), end="</function>"),
            lambda x: x == "function...</function>",
            id="Tag",
        ),
        pytest.param(
            "", SOC.ConstString("a") + SOC.ConstString("b") + SOC.ConstString("c"), lambda x: x == "abc", id="Concat"
        ),
        pytest.param(
            "",
            SOC.ConstString("a") | SOC.ConstString("b") | SOC.ConstString("c"),
            lambda x: x in ["a", "b", "c"],
            id="Union",
        ),
        pytest.param(
            "",
            SOC.QwenXMLParametersFormat(json.dumps(RESTAPIResponse.model_json_schema())),
            lambda x: (
                # enum values are placed in double quotes for some reason
                re.search(r"<parameter=status>\"(success|error)\"</parameter>", x) is not None
                and re.search(r"<parameter=data>[A-Z][a-z]{1,20}</parameter>", x) is not None
            ),
            id="QwenXMLParametersFormat",
        ),
        pytest.param(
            "TriggeredTags. Repeat word 'function'",
            SOC.TriggeredTags(
                triggers=["function"],
                tags=[
                    SOC.Tag(begin="function", content=SOC.ConstString("A"), end="</function>"),
                    SOC.Tag(begin="function", content=SOC.ConstString("B"), end="</function>"),
                ],
                at_least_one=True,
                stop_after_first=True,
            ),
            lambda x: re.match(r"(function(A|B)</function>)", x) is not None,
            id="TriggeredTags",
        ),
        pytest.param(
            "",
            SOC.TagsWithSeparator(
                tags=[
                    SOC.Tag(begin="<f>", content=SOC.ConstString("A"), end="</f>"),
                    SOC.Tag(begin="<f>", content=SOC.ConstString("B"), end="</f>"),
                ],
                separator=";",
                at_least_one=True,
                stop_after_first=False,
            ),
            lambda x: re.match(r"(<f>(A|B)</f>(;<f>(A|B)</f>))*", x) is not None,
            id="TagsWithSeparator",
        ),
    ],
)
def test_structural_tags(ov_pipe, prompt, tag, validate):
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = 3 if isinstance(tag, SOC.AnyText) else 100
    gen_config.do_sample = False
    gen_config.structured_output_config = SOC(structural_tags_config=tag)
    res_str = ov_pipe.generate(prompt, generation_config=gen_config)
    assert validate(res_str), f"Output `{res_str}` does not match structural tag {tag}"
