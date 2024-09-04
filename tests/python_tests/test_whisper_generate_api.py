# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import functools
import pytest
import openvino_tokenizers
import openvino
from ov_genai_test_utils import get_whisper_models_list
import pathlib
import os
import requests
from urllib.parse import urlparse
import datasets
from transformers import WhisperProcessor, pipeline
from optimum.intel.openvino import OVModelForSpeechSeq2Seq


@functools.lru_cache(1)
def read_whisper_model(params, **tokenizer_kwargs):
    model_id, path = params

    processor = WhisperProcessor.from_pretrained(model_id, trust_remote_code=True)

    if (path / "openvino_encoder_model.xml").exists():
        opt_model = OVModelForSpeechSeq2Seq.from_pretrained(
            path,
            trust_remote_code=True,
            compile=False,
            device="CPU",
            load_in_8bit=False,
        )
    else:
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
            processor.tokenizer, with_detokenizer=True, **tokenizer_kwargs
        )

        openvino.save_model(ov_tokenizer, path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, path / "openvino_detokenizer.xml")

        # to store tokenizer config jsons with special tokens
        processor.tokenizer.save_pretrained(path)

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

    opt_pipe = pipeline(
        "automatic-speech-recognition",
        model=opt_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )

    return (
        model_id,
        path,
        opt_pipe,
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


def compare_genai_opt_pipelines(opt_pipe, genai_pipe, dataset_id):
    ds = datasets.load_dataset(dataset_id, "clean", split="validation")

    for ds_row in ds:
        audio_sample = ds_row["audio"]
        genai_result = genai_pipe.generate(audio_sample["array"].tolist())
        result = opt_pipe(audio_sample)

        assert genai_result.texts[0] == result["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.precommit
def test_whisper(model_descr):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    compare_genai_opt_pipelines(
        opt_pipe, pipe, "hf-internal-testing/librispeech_asr_dummy"
    )
