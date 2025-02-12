import pytest
import numpy as np
from transformers import AutoTokenizer
from openvino_genai import Tokenizer, TextStreamer
from utils.hugging_face import convert_and_save_tokenizer


tokenizer_model_ids = [
    "microsoft/phi-1_5",
    "openbmb/MiniCPM-o-2_6",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "NousResearch/Meta-Llama-3-8B-Instruct", # Open analog for gated "meta-llama/Meta-Llama-3-8B-Instruct",
    # ("black-forest-labs/FLUX.1-dev", dict(subfolder="tokenizer")),  # FLUX.1-dev has tokenizer in subfolder
]

# Chekc that fix for CVS-157216 works.
# String with apostrophe to check "meta-llama/Meta-Llama-3-8B-Instruct".
# If we decode without delay then we would get  "' " and then "'Set", note that space is removed.
# To fix this we introduced delay, this test checks that it works fine
str_with_apostrophe = """
Sub PrintFiles()
    Dim folder As Object
    Dim file As Object

'Set the folder to print from
    Set folder = Application.GetNamespace("Microsoft Office").PackagedInstance.GetFolder("Folder Name")
'Get all files in the folder
    folder.Files.Clear
"""

prompts = [
    # '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?',
    "如果您有任何疑问，请联系我们，我们将予以解答。",
    "מחרוזת בדיקה",
    "Multiline\nstring!\nWow!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Тестовая строка!",
    "Tester, la chaîne...",
    "سلسلة الاختبار",
    "Сынақ жолы á",
    str_with_apostrophe,
]
@pytest.mark.parametrize("model_id", tokenizer_model_ids)
@pytest.mark.precommit
@pytest.mark.parametrize("prompt", prompts)
def test_text_prompts(tmp_path, prompt, model_id):
    model_id, hf_tok_load_params = (model_id[0], model_id[1]) if isinstance(model_id, tuple) else (model_id, {})

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_tok_load_params, trust_remote_code=True)
    convert_and_save_tokenizer(hf_tokenizer, tmp_path)
    ov_tokenizer = Tokenizer(tmp_path)
    if prompt == str_with_apostrophe and model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        pytest.skip(reason="This test is skipped because of the specific behaviour of TinyLlama CVS-162362. It's not a bug HF behaves the same.")
    tokens = ov_tokenizer.encode(prompt=prompt).input_ids.data[0].tolist()
    streamer = TextStreamer(ov_tokenizer, lambda x: accumulated.append(x))
    accumulated = []
    for token in tokens:
        streamer.write(token)
    streamer.end()

    assert ''.join(accumulated) == ov_tokenizer.decode(tokens)

encoded_prompts = [
    # This tokens caused error in Meta-Llama-3-8B-Instruct
    [2, 3479, 990, 122, 254, 9, 70, 498, 655],

    # '\n\n# 利用re.sub()方法，�' with UTF8 invalid for "microsoft/phi-1_5"
    [198, 198, 2, 10263, 230, 102, 18796, 101, 260, 13],

    # '룅튜룅튜�' causes error on "openbmb/MiniCPM-o-2_6" / "katuni4ka/tiny-random-minicpmv-2_6"
    [167, 96, 227, 169, 232, 250, 167, 96, 227, 169, 232, 250, 167]
]
@pytest.mark.parametrize("model_id", tokenizer_model_ids)
@pytest.mark.precommit
@pytest.mark.parametrize("encoded_prompt", encoded_prompts)
def test_encoded_prompts(tmp_path, encoded_prompt, model_id):
    model_id, hf_tok_load_params = (model_id[0], model_id[1]) if isinstance(model_id, tuple) else (model_id, {})

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_tok_load_params, trust_remote_code=True)
    convert_and_save_tokenizer(hf_tokenizer, tmp_path)
    ov_tokenizer = Tokenizer(tmp_path)

    streamer = TextStreamer(ov_tokenizer, lambda x: accumulated.append(x))
    accumulated = []
    for token in encoded_prompt:
        streamer.write(token)
    streamer.end()

    assert ''.join(accumulated) == ov_tokenizer.decode(encoded_prompt)
