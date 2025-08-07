# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
from typing import Optional

import numpy as np
import openvino
import openvino.properties.hint as hints
import pytest
from data.models import get_models_list
from openvino_genai import Tokenizer
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer

from utils.constants import get_disabled_mmap_ov_config
from utils.hugging_face import convert_and_save_tokenizer, download_and_convert_model
from utils.network import retry_request
from utils.tokenizers import delete_rt_info, model_tmp_path


def load_genai_tokenizer_with_configs(configs: list[tuple], temp_path):
    delete_rt_info(configs, temp_path)

    for json_file in temp_path.glob("*.json"):
        json_file.unlink()

    for config_json, config_name in configs:
        with (temp_path / config_name).open("w", encoding="utf-8") as f:
            json.dump(config_json, f)

    ov_tokenizer = Tokenizer(temp_path)
    return ov_tokenizer


def get_chat_templates():
    # Returns chat templates saved in tokenizer_configs.py,
    # but skips some models that currently are not processed correctly.

    skipped_models = {
        # TODO: openchat/openchat_3.5 and berkeley-nest/Starling-LM-7B-alpha have the same template.
        # Need to enable and unskip, since it's preset in continuous batching and has >100 000 downloads.
        "openchat/openchat-3.5-0106",
        # These models fail even on HF so no need to check if applying chat matches.
        "vibhorag101/llama-2-13b-chat-hf-phr_mental_therapy",
        "codellama/CodeLlama-34b-Instruct-hf",
        "deepseek-ai/deepseek-math-7b-rl",
        "allenai/tulu-2-7b",
        "alexsobolev/IcaroLM",
        "tokyotech-llm/Swallow-7b-instruct-v0.1",
        "bofenghuang/vigogne-2-7b-chat",
        "OpenBuddy/openbuddy-mistral2-7b-v20.3-32k",
        "AliAbdelrasheed/maqa_llama_4bit",
        "stephenlzc/Mistral-7B-v0.3-Chinese-Chat-uncensored",
        # TODO: Need to support chat templates in more models: CVS-145963
        # Either ov_genai is unable to parse chat_template or results do not match with HF.
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "databricks/dbrx-instruct",  # Chat template is not supported by Jinja2Cpp
        "mosaicml/mpt-30b-chat",
        "deepseek-ai/deepseek-coder-6.7b-instruct",  # Chat template is not supported by Jinja2Cpp
        "maldv/winter-garden-7b-alpha",  # Chat template is not supported by Jinja2Cpp
        "ishorn5/RTLCoder-Deepseek-v1.1",  # Chat template is not supported by Jinja2Cpp
        "openchat/openchat-3.5-0106",
        "casperhansen/llama-3-70b-instruct-awq",
        "TheBloke/deepseek-coder-33B-instruct-GPTQ",
        "AI-Sweden-Models/gpt-sw3-356m-instruct",
        "google/gemma-7b-it",
        "THUDM/cogvlm2-llama3-chat-19B",
        "KnutJaegersberg/internlm-20b-llama",
        "maywell/Synatra-Mixtral-8x7B",
        "MediaTek-Research/Breeze-7B-Instruct-v1_0",
        "bofenghuang/vigostral-7b-chat",
        "meetkai/functionary-small-v2.5",  # Chat template is not supported by Jinja2Cpp
        "openchat/openchat-3.6-8b-20240522",
        "tenyx/TenyxChat-7B-v1",
        "LoneStriker/TinyLlama-1.1B-32k-Instruct-3.0bpw-h6-exl2",
        "yam-peleg/Hebrew-Gemma-11B-V2",
        "shenzhi-wang/Llama3-8B-Chinese-Chat",  # AssertionError
        "nlpai-lab/KULLM3",
        "HuggingFaceH4/zephyr-7b-gemma-sft-v0.1",
        "MediaTek-Research/Breeze-7B-Instruct-v0_1",
        "shanchen/llama3-8B-slerp-biomed-chat-chinese",  # AssertionError
        "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        "aloobun/CosmicBun-8B",  # Chat template is not supported by Jinja2Cpp
        "codellama/CodeLlama-70b-Instruct-hf",
        "gorilla-llm/gorilla-openfunctions-v2",  # Chat template is not supported by Jinja2Cpp
        "BramVanroy/Llama-2-13b-chat-dutch",
    }

    from data.tokenizer_configs import get_tokenizer_configs

    return [(k, v) for k, v in get_tokenizer_configs().items() if k not in skipped_models]


prompts = [
    "table is made of",
    "你好！ 你好嗎？",
    "Alan Turing was a",
    "The Sun is yellow because",
    ["The Sun is yellow because", "Alan Turing was a", "Alan Turing was a"],
]


@pytest.fixture(scope="module")
def ov_hf_tokenizers(request):
    _, hf_tokenizer, models_path = download_and_convert_model(request.param)
    ov_tokenizer = Tokenizer(models_path)
    return ov_tokenizer, hf_tokenizer


@pytest.mark.parametrize("ov_hf_tokenizers", get_models_list(), indirect=True)
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.precommit
def test_encode(ov_hf_tokenizers, prompt):
    ov_tokenizer, hf_tokenizer = ov_hf_tokenizers

    encoded_ov = ov_tokenizer.encode(prompt).input_ids.data
    if isinstance(prompt, list):
        encoded_hf = hf_tokenizer.batch_encode_plus(prompt)["input_ids"]
        for tokens_ov, tokens_hf in zip(encoded_ov, encoded_hf):
            assert np.all(tokens_ov == tokens_hf)
    else:
        encoded_hf = hf_tokenizer.encode(prompt)
        assert np.all(encoded_hf == encoded_ov[0])


encoded_prompts = [
    [1, 1591, 338, 1754, 310],
    [1, 17102, 323, 3864, 471, 263],
    # chineze characters
    [1, 29871, 30919, 31076, 30584, 29871, 30919, 31076, 232, 154, 145, 30882],
    # On meta-llama/Meta-Llama-3-8B-Instruct this becomes longer  after removing the last token
    [3113, 264, 364, 267],
    # batched tokens
    [
        [1, 1591, 338, 1754, 310],
        [1, 1591, 338, 1754, 310],
        [1, 17102, 323, 3864, 471, 263],
    ],
]


@pytest.mark.parametrize("ov_hf_tokenizers", get_models_list(), indirect=True)
@pytest.mark.parametrize("encoded_prompt", encoded_prompts)
@pytest.mark.precommit
def test_decode(ov_hf_tokenizers, encoded_prompt):
    ov_tokenizer, hf_tokenizer = ov_hf_tokenizers
    decoded_ov = ov_tokenizer.decode(encoded_prompt)

    if isinstance(encoded_prompt[0], list):
        decoded_hf = hf_tokenizer.batch_decode(encoded_prompt, skip_special_tokens=True)
        for tokens_ov, tokens_hf in zip(decoded_ov, decoded_hf):
            assert np.all(tokens_ov == tokens_hf)
    else:
        decoded_hf = hf_tokenizer.decode(encoded_prompt, skip_special_tokens=True)
        assert decoded_hf == decoded_ov


conversation = [
    {"role": "user", "content": "1+1="},
    {"role": "assistant", "content": "1 + 1 = 2"},
    {"role": "user", "content": "What is the previous answer?"},
    {
        "role": "assistant",
        "content": "The previous answer was: 1 + 1 = 2. Please ask me your next question.",
    },
    {"role": "user", "content": "Why is the sun yellow?"},
    {"role": "assistant", "content": "Because it emits yeloow light."},
    {"role": "user", "content": "What was my first question?"},
]


@pytest.mark.precommit
@pytest.mark.parametrize("chat_config", get_chat_templates())
@pytest.mark.parametrize("ov_hf_tokenizers", get_models_list(), indirect=True)
def test_apply_chat_template(model_tmp_path, chat_config: tuple[str, dict], ov_hf_tokenizers):
    tokenizer_config = chat_config[1]
    # load hf_tokenizer only to apply chat template to ov_tokenizer later
    _, hf_tokenizer = ov_hf_tokenizers

    hf_full_history_str = hf_tokenizer.apply_chat_template(
        conversation, add_generation_prompt=False, tokenize=False, **tokenizer_config
    )

    ov_tokenizer = load_genai_tokenizer_with_configs([(tokenizer_config, "tokenizer_config.json")], model_tmp_path[1])
    ov_tokenizer.set_chat_template(tokenizer_config["chat_template"])
    ov_full_history_str = ov_tokenizer.apply_chat_template(conversation, add_generation_prompt=False)

    if ov_full_history_str != hf_full_history_str:
        print(f"hf reference: {hf_full_history_str}")
        print(f"ov_genai out: {ov_full_history_str}")
    assert ov_full_history_str == hf_full_history_str

    # Test throwing exception for empty rendered chat template
    # Example: Qwen2-VL chat template
    chat_template_for_empty_output = "{% if messages is string %}{{ messages }}{% else %}{% for content in messages %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}"
    with pytest.raises(Exception):
        ov_tokenizer.apply_chat_template(conversation, chat_template=chat_template_for_empty_output)


@pytest.mark.precommit
@pytest.mark.parametrize(
    "hf_ov_genai_models", 
    [("Xenova/c4ai-command-r-v01-tokenizer", { "padding_side": None })],
    indirect=True
)
def test_non_string_chat_template(hf_ov_genai_models):
    hf_tokenizer, genai_tokenzier = hf_ov_genai_models
    
    hf_full_history_str = hf_tokenizer.apply_chat_template(
        conversation, add_generation_prompt=False, tokenize=False,
    )

    ov_full_history_str = genai_tokenzier.apply_chat_template(conversation, add_generation_prompt=False)

    if ov_full_history_str != hf_full_history_str:
        print(f"hf reference: {hf_full_history_str}")
        print(f"ov_genai out: {ov_full_history_str}")
    assert ov_full_history_str == hf_full_history_str


@pytest.mark.precommit
@pytest.mark.parametrize("ov_hf_tokenizers", get_models_list(), indirect=True)
def test_set_chat_template(ov_hf_tokenizers):
    ov_tokenizer, hf_tokenizer = ov_hf_tokenizers

    prompt = "how are you?"
    dummy_conversation = [
        {"role": "user", "content": prompt},
    ]
    identity_chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    templated_prompt_inline = ov_tokenizer.apply_chat_template(
        dummy_conversation,
        add_generation_prompt=False,
        chat_template=identity_chat_template,
    )

    ov_tokenizer.set_chat_template(identity_chat_template)
    templated_prompt = ov_tokenizer.apply_chat_template(dummy_conversation, add_generation_prompt=False)

    assert templated_prompt_inline == templated_prompt
    assert prompt == templated_prompt


eng_prompts = [
    "1+1=",
    "What is the previous answer?",
    "Why is the Sun yellow?",
    "What was my first question?",
    ["Why is the Sun yellow?"],
    "Multiline\nstring\nWow!",
]
unicode_prompts = [
    *(str.encode(x, "unicode_escape") for x in [
            "如果您有任何疑问，请联系我们，我们将予以解答。",
            "מחרוזת בדיקה",
        ])
]


@pytest.mark.parametrize(
    "ov_hf_tokenizers",
    [
        "katuni4ka/tiny-random-phi3",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # ("black-forest-labs/FLUX.1-dev", dict(subfolder="tokenizer")),  # FLUX.1-dev has tokenizer in subfolder
    ],
    indirect=True,
)
@pytest.mark.precommit
@pytest.mark.parametrize("prompt", [*eng_prompts, *unicode_prompts])
def test_special_tokens(prompt, ov_hf_tokenizers):
    prompt = prompt.decode("unicode_escape") if isinstance(prompt, bytes) else prompt

    ov_tokenizer, hf_tokenizer = ov_hf_tokenizers

    # Calling encode with 'add_special_tokens' will set state flag.
    ov_res_add_spec = ov_tokenizer.encode(prompt, add_special_tokens=True).input_ids.data
    ov_res_no_spec = ov_tokenizer.encode(prompt, add_special_tokens=False).input_ids.data
    hf_res_add_spec = hf_tokenizer(prompt, return_tensors="np", add_special_tokens=True)["input_ids"]
    hf_res_no_spec = hf_tokenizer(prompt, return_tensors="np", add_special_tokens=False)["input_ids"]
    assert np.all(ov_res_add_spec == hf_res_add_spec)
    assert np.all(ov_res_no_spec == hf_res_no_spec)

    # Check that add_special_tokens flag indeed made any difference
    assert ov_res_add_spec.size != ov_res_no_spec.size
    assert hf_res_add_spec.size != hf_res_no_spec.size

    # Decode with 'skip_special_tokens'
    decoded_genai_skip_spec = ov_tokenizer.decode(hf_res_add_spec, skip_special_tokens=True)[0]
    decoded_genai_no_skip = ov_tokenizer.decode(hf_res_add_spec, skip_special_tokens=False)[0]
    decoded_hf_skip_spec = hf_tokenizer.decode(hf_res_add_spec[0], skip_special_tokens=True)
    decoded_hf_no_skip = hf_tokenizer.decode(hf_res_add_spec[0], skip_special_tokens=False)
    assert decoded_genai_skip_spec == decoded_hf_skip_spec
    assert decoded_genai_no_skip == decoded_hf_no_skip

    # Check that skip_special_tokens indeed made any difference
    assert decoded_genai_skip_spec != decoded_genai_no_skip
    assert decoded_hf_skip_spec != decoded_hf_no_skip


@pytest.mark.precommit
def test_multiple_infer_request_state(tmp_path):
    hf_tokenizer = retry_request(lambda: AutoTokenizer.from_pretrained("llamafactory/tiny-random-Llama-3"))
    ov_tokenizer = convert_tokenizer(hf_tokenizer)
    openvino.save_model(ov_tokenizer, tmp_path / "openvino_tokenizer.xml")
    del ov_tokenizer, hf_tokenizer

    ov_tokenizer = Tokenizer(
        tmp_path,
        properties={hints.performance_mode: hints.PerformanceMode.THROUGHPUT} | get_disabled_mmap_ov_config(),
        )
    text = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

    What is OpenVINO?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    result_tensors = [ov_tokenizer.encode(text, add_special_tokens=False).input_ids.data for _ in range(10)]
    gt_tensor = result_tensors.pop()
    assert all(((tensor.shape == gt_tensor.shape) and (tensor == gt_tensor).all()) for tensor in result_tensors)


@pytest.fixture(scope="module")
def hf_ov_genai_models(request, tmp_path_factory):
    model_id, args = request.param
    tok_load_properties = {"add_second_input": args.pop("add_second_input", False)}
    
    hf_args = args.copy()  # to overcome mutable default argument side effects
    if "padding_side" in hf_args and hf_args["padding_side"] is None:
        # HF does not accept None.
        # Need to remove padding_side and let HF to choose default value,
        hf_args.pop("padding_side")
    else:
        hf_args["truncation_side"] = hf_args["padding_side"]
    model_dir = tmp_path_factory.getbasetemp() / model_id.replace("/", "_")
    model_dir.mkdir(exist_ok=True, parents=True)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_args)
    convert_args = {"number_of_inputs": hf_args.pop("number_of_inputs")} if "number_of_inputs" in hf_args else {}
    convert_and_save_tokenizer(hf_tokenizer, model_dir, **convert_args)

    genai_tokenizer = Tokenizer(model_dir, tok_load_properties)
    return hf_tokenizer, genai_tokenizer


prompts = [
    ["1+1=", "What is the previous answer?"],
    # long sentence exceeding max_length, check that is truncated
    "What is the previous answers? " * 1000,
    # check that short sentence is padded to long
    "what",
    # check that large batch with multilangual data is correctly padded
    [
        "1+1=",
        "What is the previous answer?",
        "Why is the Sun yellow?",
        "What was my first question?",
        "若我有一亿美元，在人工智能盛行的今天，我怎样投资才能收益最大化？",
        "מחרוזת בדיקה",
        "Multiline\nstring!\nWow!",
    ],
]


@pytest.mark.precommit
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize("max_length", [None, 16, 103, 512, 1024])
@pytest.mark.parametrize("pad_to_max_length", [None, True, False])
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.parametrize(
    "hf_ov_genai_models",
    [
        ("katuni4ka/tiny-random-phi3", {"padding_side": None}),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", {"padding_side": None}),
        ("katuni4ka/tiny-random-llava-next", {"padding_side": "right"}),
        ("katuni4ka/tiny-random-llava-next", {"padding_side": "left"}),
        (
            "BAAI/bge-small-en-v1.5",
            {"padding_side": None},
        ),  # model with 2 RaggedToDense ops
        # ("black-forest-labs/FLUX.1-dev", dict(subfolder="tokenizer")),  # FLUX.1-dev has tokenizer in subfolder
    ],
    indirect=True,
)
def test_padding(
    hf_ov_genai_models,
    add_special_tokens,
    max_length,
    pad_to_max_length,
    prompt,
):
    hf_tokenizer, genai_tokenzier = hf_ov_genai_models

    # In openvino_tokenizers if sequences are of different length by default padding is applied
    # to the longest sequence in the batch since resulting tokenization is stored as a signe ov::Tensor
    # which cannot store irregular/ragged array.
    # Therefore, for default mode truncation=True.
    # For the same reason runcation is always applied.
    # Truncate only if max_length is set.
    is_max_len_set = max_length is not None
    hf_pad_params_map = {
        None: {"padding": "longest", "truncation": is_max_len_set},
        False: {"padding": "longest", "truncation": is_max_len_set},
        True: {"padding": "max_length", "truncation": is_max_len_set},
    }
    hf_params = dict(
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        **hf_pad_params_map[pad_to_max_length],
    )
    ov_params = {
        "add_special_tokens": add_special_tokens,
        "max_length": max_length,
        "pad_to_max_length": pad_to_max_length,
    }
    if pad_to_max_length is None:
        ov_params.pop("pad_to_max_length")
    if max_length is None:
        hf_params.pop("max_length")
        ov_params.pop("max_length")

    ov_res = genai_tokenzier.encode(prompt, **ov_params)
    hf_res = hf_tokenizer(prompt, return_tensors="np", **hf_params)
    assert np.all(ov_res.input_ids.data == hf_res["input_ids"])
    assert np.all(ov_res.attention_mask.data == hf_res["attention_mask"])


# Define model base configs
base_models_for_paired_input_test = [
    ("answerdotai/ModernBERT-base", {"padding_side": None}),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", {"padding_side": None}),
    ("katuni4ka/tiny-random-llava-next", {"padding_side": "right"}),
    ("katuni4ka/tiny-random-llava-next", {"padding_side": "left"}),
]

def make_model_params():
    # Parametrize over add_second_input and number_of_inputs
    
    params = []
    for model_id_and_params in base_models_for_paired_input_test:
        model_id, params_dict = model_id_and_params
        params.append((model_id, {**params_dict, "add_second_input": True}))
        params.append((model_id, {**params_dict, "number_of_inputs": 2}))

        # in this case even user requested add_second_input, since during conversion model will be ready for paired input
        # without calling the AddSecondInputPass.
        params.append((model_id, {**params_dict, "add_second_input": True, "number_of_inputs": 2}))
    return params

models_with_pair_input = make_model_params()

@pytest.mark.parametrize("hf_ov_genai_models", models_with_pair_input, indirect=True)
@pytest.mark.precommit
@pytest.mark.parametrize("input_pair", [[
    ["hi", "sun in yellow"],
    ["Eng... test, string?!" * 100, "Multiline\nstring!\nWow!"],
    ["Eng... test, string?!", "Multiline\nstring!\nWow!" * 100],
    ["Eng... test, string?!" * 100, "Multiline\nstring!\nWow!" * 100],
    ["hi" * 20, "buy" * 90],
]])
def test_two_inputs_string_list_of_lists_batched(hf_ov_genai_models, input_pair):
    # Check with batched inputs: list of [str, str] pairs, consistent with HF format.
    hf_tokenizer, genai_tokenizer = hf_ov_genai_models
    ov_encoded = genai_tokenizer.encode(input_pair).input_ids.data
    hf_encoded = hf_tokenizer(input_pair, return_tensors="np", padding=True)["input_ids"]
    assert np.all(ov_encoded == hf_encoded)

@pytest.mark.parametrize("hf_ov_genai_models", models_with_pair_input, indirect=True)
@pytest.mark.precommit
@pytest.mark.parametrize("input_pair", [
    [["hi", "sun in yellow"]],
    [["Eng... test, string?!" * 100, "Multiline\nstring!\nWow!"]],
    [["Eng... test, string?!", "Multiline\nstring!\nWow!" * 100]],
    [["Eng... test, string?!" * 100, "Multiline\nstring!\nWow!" * 100]],
    [["hi" * 20, "buy" * 90]],
])
def test_two_inputs_string_list_of_lists(hf_ov_genai_models, input_pair):
    # Check with inputs consisted of lists of lists consistent with HF format.
    hf_tokenizer, genai_tokenzier = hf_ov_genai_models
    ov_encoded = genai_tokenzier.encode(input_pair).input_ids.data
    hf_encoded = hf_tokenizer(input_pair, return_tensors="np")["input_ids"]
    assert np.all(ov_encoded == hf_encoded)


@pytest.mark.parametrize("hf_ov_genai_models", models_with_pair_input, indirect=True)
@pytest.mark.precommit
@pytest.mark.parametrize("input_pair", [
    [["Eng... test, string?!" * 100], ["Multiline\nstring!\nWow!"]],
    [["hi" * 20], ["buy" * 90]],
    [["What is the capital of Great Britain"] * 4, ["London is capital of Great Britain"]],
    [["What is the capital of Great Britain"], ["London is capital of Great Britain"] * 4],
])
def test_two_inputs_string(hf_ov_genai_models, input_pair):
    # Test when inputs are separate and they are broadcasted to the same length.
    # For HF we broadcast manually, but in GenAI this happens automatically.
    hf_tokenizer, genai_tokenzier = hf_ov_genai_models

    # broadcast ([N], [1]) and ([1], [N]) to ([N], [N]) for HF
    if len(input_pair[0]) > len(input_pair[1]):
        input_pair_hf = [[input_pair[0][i], input_pair[1][0]] for i in range(len(input_pair[0]))]
    else:
        input_pair_hf = [[input_pair[0][0], input_pair[1][i]] for i in range(len(input_pair[1]))]
    
    ov_encoded = genai_tokenzier.encode(*input_pair).input_ids.data
    hf_encoded = hf_tokenizer(input_pair_hf, return_tensors="np")["input_ids"]
    assert np.all(ov_encoded == hf_encoded)


@pytest.mark.precommit
def test_load_special_tokens_from_config_json(model_tmp_path):
    # test when there is an available config.json
    config_json = {
        "pad_token_id": 422,
        "bos_token_id": 42,
        "eos_token_id": 37,
    }
    tok = load_genai_tokenizer_with_configs([(config_json, "config.json")], model_tmp_path[1])
    assert tok.get_pad_token_id() == config_json["pad_token_id"]
    assert tok.get_bos_token_id() == config_json["bos_token_id"]
    assert tok.get_eos_token_id() == config_json["eos_token_id"]


@pytest.mark.precommit
def test_load_special_tokens_from_special_tokens_map_json(model_tmp_path):
    # test with special_tokens_map
    special_tokens_map_json = {
        "pad_token": {"content": "<custom_pad>"},
        "bos_token": {"content": "<custom_bos>"},
        "eos_token": {"content": "<custom_eos>"},
    }
    tok = load_genai_tokenizer_with_configs([(special_tokens_map_json, "special_tokens_map.json")], model_tmp_path[1])
    assert tok.get_pad_token() == special_tokens_map_json["pad_token"]["content"]
    assert tok.get_bos_token() == special_tokens_map_json["bos_token"]["content"]
    assert tok.get_eos_token() == special_tokens_map_json["eos_token"]["content"]


@pytest.mark.precommit
def test_load_special_tokens_from_tokenizer_config_json(model_tmp_path):
    # special_tokens_map is not available
    # but tokenize_config.json exists
    # will load both string and integer representations
    tok_config_json = {
        "added_tokens_decoder": {
            "422": {"content": "<pad>"},
            "37": {"content": "<s>"},
            "42": {"content": "</s>"},
        },
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }

    tok = load_genai_tokenizer_with_configs([(tok_config_json, "tokenizer_config.json")], model_tmp_path[1])
    assert tok.get_pad_token() == tok_config_json["pad_token"]
    assert tok.get_bos_token() == tok_config_json["bos_token"]
    assert tok.get_eos_token() == tok_config_json["eos_token"]

    assert tok.get_pad_token_id() == 422
    assert tok.get_bos_token_id() == 37
    assert tok.get_eos_token_id() == 42


@pytest.mark.precommit
def test_load_special_tokens_from_tokenizer_config_and_config_json(model_tmp_path):
    # both config.json is available and tokenizer_config.json available
    # check that it does not read int values from tokenizer_config.json if they are in config.json
    tok_config_json = {
        "added_tokens_decoder": {
            # integers differ from config.json to check they don't override config.json
            "777": {"content": "<pad>"},
            "888": {"content": "<s>"},
            "656": {"content": "</s>"},
        },
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    config_json = {
        "pad_token_id": 422,
        "bos_token_id": 42,
        "eos_token_id": 37,
    }
    configs = [(tok_config_json, "tokenizer_config.json"), (config_json, "config.json")]
    tok = load_genai_tokenizer_with_configs(configs, model_tmp_path[1])
    assert tok.get_pad_token_id() == config_json["pad_token_id"]
    assert tok.get_bos_token_id() == config_json["bos_token_id"]
    assert tok.get_eos_token_id() == config_json["eos_token_id"]

    assert tok.get_pad_token() == tok_config_json["pad_token"]
    assert tok.get_bos_token() == tok_config_json["bos_token"]
    assert tok.get_eos_token() == tok_config_json["eos_token"]


@pytest.mark.precommit
@pytest.mark.xfail(
    raises=AssertionError,
    reason="CVS-143410 ov tokenizer should be aligned with hf",
    strict=False,
)
def test_load_special_tokens_from_special_tokens_map_json_with_string_repr(
    model_tmp_path,
):
    # only string representation is provided, find token integers by inference
    model_id, temp_path = model_tmp_path
    tokenizer = retry_request(lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))

    special_tokens_map_json = {}
    token_str_int_map = {}
    special_token_names = ["pad_token", "bos_token", "eos_token"]
    for token_str in special_token_names:
        if hasattr(tokenizer, token_str):
            token_val = getattr(tokenizer, token_str)
            special_tokens_map_json.update({token_str: {"content": token_val}})
            token_id = tokenizer(token_val, add_special_tokens=False)["input_ids"][0]
            token_str_int_map.update({token_str: token_id})

    # since only string representations are present in the json will try to get by inference
    tok = load_genai_tokenizer_with_configs([(special_tokens_map_json, "special_tokens_map.json")], temp_path)

    # check ids inferred correctly for special tokens existing if HF tokenizer
    if "pad_token" in token_str_int_map:
        assert tok.get_pad_token_id() == token_str_int_map["pad_token"]
    if "bos_token" in token_str_int_map:
        assert tok.get_bos_token_id() == token_str_int_map["bos_token"]
    if "eos_token" in token_str_int_map:
        assert tok.get_eos_token_id() == token_str_int_map["eos_token"]


@dataclasses.dataclass(frozen=True)
class ChatTemplates:
    reference: Optional[str]
    rt_simplified: Optional[str]
    rt_template: Optional[str]
    chat_template_json: Optional[str]
    processor_config_json: Optional[str]
    tokenizer_config_json: Optional[str]


def generate_tokenizer(tmp_path, chat_templates):
    input_ids = openvino.op.Constant(openvino.Type.i64, openvino.Shape([0, 0]), []).output(0)
    input_ids.get_tensor().set_names({"input_ids"})
    attention_mask = openvino.op.Constant(openvino.Type.i64, openvino.Shape([0, 0]), []).output(0)
    attention_mask.get_tensor().set_names({"attention_mask"})
    model = openvino.Model(
        [openvino.op.Result(input_ids), openvino.op.Result(attention_mask)],
        [openvino.op.Parameter(openvino.Type.string, openvino.Shape([1]))],
    )
    if chat_templates.rt_simplified is not None:
        model.set_rt_info(chat_templates.rt_simplified, "simplified_chat_template")
    if chat_templates.rt_template is not None:
        model.set_rt_info(chat_templates.rt_template, "chat_template")
    if chat_templates.chat_template_json is not None:
        with open(tmp_path / "chat_template.json", "w", encoding="utf-8") as file:
            json.dump({"chat_template": chat_templates.chat_template_json}, file)
    if chat_templates.processor_config_json is not None:
        with open(tmp_path / "processor_config.json", "w", encoding="utf-8") as file:
            json.dump({"chat_template": chat_templates.processor_config_json}, file)
    if chat_templates.tokenizer_config_json is not None:
        with open(tmp_path / "tokenizer_config.json", "w", encoding="utf-8") as file:
            json.dump({"chat_template": chat_templates.tokenizer_config_json}, file)
    openvino.save_model(model, tmp_path / "openvino_tokenizer.xml")
    return Tokenizer(tmp_path)


QWEN2_VL_2B = "{% if messages is string %}{{ messages }}{% else %}{% for content in messages %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}"


SIMPLIFIED_QWEN2_VL_2B = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


SIMPLIFIED_QWEN3 = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


PATCHED_SIMPLIFIED_QWEN3 = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


@pytest.mark.precommit
def test_set_special_runtime_template(tmp_path):
    tokenizer = generate_tokenizer(tmp_path, ChatTemplates(None, None, None, None, None, None))
    tokenizer.chat_template = QWEN2_VL_2B
    assert tokenizer.chat_template == SIMPLIFIED_QWEN2_VL_2B


@pytest.mark.precommit
@pytest.mark.parametrize(
    "chat_templates",
    [
        # If the template was not in the list of not supported GenAI templates from (5), it's replaced with simplified_chat_template entry from rt_info section of ov::Model:
        ChatTemplates("correct template", "correct template", "", "", "", ""),
        ChatTemplates("correct template", None, "correct template", "", "", ""),
        ChatTemplates("correct template", None, None, "correct template", "", ""),
        ChatTemplates("correct template", None, None, None, "correct template", ""),
        ChatTemplates("correct template", None, None, None, None, "correct template"),
        # If the template is known to be not supported by GenAI, it's replaced with a simplified supported version:
        ChatTemplates(SIMPLIFIED_QWEN2_VL_2B, "", QWEN2_VL_2B, "", "", ""),
        # Replace not supported instructions with equivalents in chat_template rt_info:
        ChatTemplates(PATCHED_SIMPLIFIED_QWEN3, None, SIMPLIFIED_QWEN3, "", "", ""),
        # Replace not supported instructions with equivalents in simplified_chat_template rt_info:
        ChatTemplates(PATCHED_SIMPLIFIED_QWEN3, SIMPLIFIED_QWEN3, "", "", "", ""),
    ],
)
def test_template_priorities(tmp_path, chat_templates):
    tokenizer = generate_tokenizer(tmp_path, chat_templates)
    assert tokenizer.chat_template == chat_templates.reference

