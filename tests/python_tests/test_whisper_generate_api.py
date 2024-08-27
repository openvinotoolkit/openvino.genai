# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import functools
import pytest
import numpy as np
import openvino_tokenizers
import openvino
from ov_genai_test_utils import get_whisper_models_list
import pathlib
import os
import requests
from urllib.parse import urlparse
from scipy.io import wavfile


@functools.lru_cache(1)
def read_whisper_model(params, **tokenizer_kwargs):
    model_id, path = params

    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if (path / "openvino_encoder_model.xml").exists():
        opt_model = OVModelForSpeechSeq2Seq.from_pretrained(
            path, trust_remote_code=True, compile=False, device="CPU"
        )
    else:
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
            tokenizer, with_detokenizer=True, **tokenizer_kwargs
        )
        openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")

        # to store tokenizer config jsons with special tokens
        tokenizer.save_pretrained(path)

        opt_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            trust_remote_code=True,
            compile=False,
            device="CPU",
            load_in_8bit=False,
        )
        opt_model.generation_config.save_pretrained(path)
        opt_model.config.save_pretrained(path)
        opt_model.save_pretrained(path)

    return (
        model_id,
        path,
        tokenizer,
        opt_model,
        ov_genai.WhisperSpeechRecognitionPipeline(
            str(path), device="CPU", config={"ENABLE_MMAP": False}
        ),
    )


def download_file(wav_url):
    response = requests.get(wav_url, stream=True)

    prefix = pathlib.Path(os.getenv("GENAI_ASSETS_PATH_PREFIX", "assets"))

    pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
    file_name = os.path.basename(urlparse(wav_url).path)
    file_path = prefix / file_name
    with open(file_path, "wb") as wav_file:
        for data in response.iter_content():
            wav_file.write(data)
    return file_path


def read_wav(wav_file_path):
    samplerate, data = wavfile.read(wav_file_path)

    # normalize to [-1 1] range
    return data / np.iinfo(data.dtype).max


# todo: replace with hf dataset
test_cases = [
    (
        "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav",
        " How are you doing today?",
    ),
]


@pytest.mark.parametrize("wav_url,expected", test_cases)
@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.precommit
def test_whisper(model_descr, wav_url, expected):
    model_id, path, tokenizer, opt_model, pipe = read_whisper_model(model_descr)

    file_path = download_file(wav_url)
    raw_speech = read_wav(file_path)

    result = pipe.generate(raw_speech)
    assert result.texts[0] == expected
