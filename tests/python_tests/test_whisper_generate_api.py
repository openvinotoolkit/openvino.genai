# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import functools
import pytest
import openvino_tokenizers
import openvino
from ov_genai_test_utils import get_whisper_models_list
import datasets
from transformers import WhisperProcessor, pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import json
import time
import tracemalloc

tracemalloc.start()


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

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
        ov_genai.WhisperPipeline(
            str(path), device="CPU", config={"ENABLE_MMAP": False}
        ),
    )


def get_test_sample(dataset_id="hf-internal-testing/librispeech_asr_dummy"):
    ds = datasets.load_dataset(dataset_id, "clean", split="validation")
    return ds[0]["audio"]["array"]


# todo: implement sequential run with model unoading
# base model probably doesn't fit to memory
def compare_genai_and_opt_pipelines(opt_pipe, genai_pipe, dataset_id):
    ds = datasets.load_dataset(dataset_id, "clean", split="validation")
    opt_infer_time = 0
    genai_infer_time = 0
    failed = 0
    for ds_row in ds:
        audio_sample = ds_row["audio"]

        start = time.time()
        genai_result = genai_pipe.generate(audio_sample["array"].tolist())
        genai_infer_time += time.time() - start

        start = time.time()
        result = opt_pipe(audio_sample)
        opt_infer_time += time.time() - start

        if genai_result.texts[0] != result["text"]:
            print(f'HuggingFace: {result["text"]}\n genai: {genai_result.texts[0]}')
            failed += 1
    print(f"Inference time\nOpt: {opt_infer_time}\nGenAI: {genai_infer_time}")
    if failed > 0:
        print(f"Filed: {failed}")
    assert failed == 0


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize("dataset_id", ["hf-internal-testing/librispeech_asr_dummy"])
@pytest.mark.precommit
def test_whisper_on_hf_dataset(model_descr, dataset_id):
    model_id, path, opt_pipe, genai_pipe = read_whisper_model(model_descr)

    compare_genai_and_opt_pipelines(opt_pipe, genai_pipe, dataset_id)


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.precommit
def test_whisper_config_constructor(model_descr):
    model_id, path = model_descr

    config = ov_genai.WhisperGenerationConfig(str(path / "generation_config.json"))

    with open(path / "generation_config.json") as f:
        original_config = json.load(f)

    assert original_config["decoder_start_token_id"] == config.decoder_start_token_id
    assert original_config["max_length"] == config.max_length
    assert original_config["eos_token_id"] == config.eos_token_id
    assert original_config["pad_token_id"] == config.pad_token_id
    if "task_to_id" in original_config:
        assert original_config["task_to_id"]["translate"] == config.translate_token_id
        assert original_config["task_to_id"]["transcribe"] == config.transcribe_token_id
    assert original_config["no_timestamps_token_id"] == config.no_timestamps_token_id

    assert set(original_config["begin_suppress_tokens"]) == set(
        config.begin_suppress_tokens
    )

    assert set(original_config["suppress_tokens"]) == set(config.suppress_tokens)

    config = ov_genai.WhisperGenerationConfig(
        suppress_tokens=[1, 2], begin_suppress_tokens=[3, 4], max_new_tokens=100
    )

    assert set(config.suppress_tokens) == set([1, 2])
    assert set(config.begin_suppress_tokens) == set([3, 4])
    assert config.max_new_tokens == 100


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize("test_sample", [get_test_sample()])
@pytest.mark.precommit
def test_whisper_constructors(model_descr, test_sample):
    model_id, path = model_descr
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(test_sample)["text"]

    genai_result = ov_genai.WhisperPipeline(
        str(path), device="CPU", config={"ENABLE_MMAP": False}
    ).generate(test_sample)

    assert genai_result.texts[0] == expected

    genai_result = ov_genai.WhisperPipeline(str(path)).generate(test_sample)

    assert genai_result.texts[0] == expected

    tokenizer = ov_genai.Tokenizer(str(path))

    genai_result = ov_genai.WhisperPipeline(
        str(path), tokenizer=tokenizer, device="CPU", config={"ENABLE_MMAP": False}
    ).generate(test_sample)

    assert genai_result.texts[0] == expected


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize("test_sample", [get_test_sample()])
@pytest.mark.precommit
def test_max_new_tokens(model_descr, test_sample):
    model_id, path = model_descr
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(test_sample, max_new_tokens=10)["text"]

    genai_result = ov_genai.WhisperPipeline(str(path)).generate(
        test_sample, max_new_tokens=10
    )

    assert genai_result.texts[0] == expected

    tokenizer = ov_genai.Tokenizer(str(path))

    genai_pipeline = ov_genai.WhisperPipeline(
        str(path), tokenizer=tokenizer, device="CPU", config={"ENABLE_MMAP": False}
    )
    config = genai_pipeline.get_generation_config()
    config.max_new_tokens = 10
    genai_result = genai_pipeline.generate(test_sample, config)

    assert genai_result.texts[0] == expected
