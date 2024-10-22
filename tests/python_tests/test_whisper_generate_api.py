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
import typing


@functools.lru_cache(1)
def read_whisper_model(params, **tokenizer_kwargs):
    model_id, path = params

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
            tokenizer,
            with_detokenizer=True,
            clean_up_tokenization_spaces=False,
            **tokenizer_kwargs,
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

    processor = WhisperProcessor.from_pretrained(model_id, trust_remote_code=True)
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
        ov_genai.WhisperPipeline(path, "CPU", config={"ENABLE_MMAP": False}),
    )


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


def get_samples_from_dataset(
    language: str = "en", length: int = 30, long_form: bool = False
):
    if not long_form:
        ds = datasets.load_dataset(
            "mozilla-foundation/common_voice_11_0",
            language,
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
    else:
        ds = datasets.load_dataset(
            "distil-whisper/meanwhile",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )

    ds = typing.cast(datasets.IterableDataset, ds)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    ds = ds.take(length)

    return [x["audio"]["array"] for x in ds]


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize("dataset_id", ["hf-internal-testing/librispeech_asr_dummy"])
@pytest.mark.precommit
def test_whisper_on_hf_dataset(model_descr, dataset_id):
    model_id, path, opt_pipe, genai_pipe = read_whisper_model(model_descr)

    compare_genai_and_opt_pipelines(opt_pipe, genai_pipe, dataset_id)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample",
    get_samples_from_dataset(language="en", length=1),
)
@pytest.mark.precommit
def test_smoke(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(test_sample)

    genai_result = pipe.generate(test_sample)

    assert genai_result.texts[0] == expected["text"]

    assert "chunks" not in expected
    assert genai_result.chunks == None


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.precommit
def test_whisper_config_constructor(model_descr):
    model_id, path = model_descr

    config = ov_genai.WhisperGenerationConfig(path / "generation_config.json")

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
    assert original_config["is_multilingual"] == config.is_multilingual

    assert set(original_config["begin_suppress_tokens"]) == set(
        config.begin_suppress_tokens
    )

    assert set(original_config["suppress_tokens"]) == set(config.suppress_tokens)

    config = ov_genai.WhisperGenerationConfig(
        suppress_tokens=[1, 2],
        begin_suppress_tokens=[3, 4],
        max_new_tokens=100,
        lang_to_id={"<|_ru|>": 42},
    )

    assert set(config.suppress_tokens) == set([1, 2])
    assert set(config.begin_suppress_tokens) == set([3, 4])
    assert config.max_new_tokens == 100
    assert config.lang_to_id["<|_ru|>"] == 42


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("test_sample", get_samples_from_dataset(length=1))
@pytest.mark.precommit
def test_whisper_constructors(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(test_sample)["text"]

    genai_result = ov_genai.WhisperPipeline(
        path, device="CPU", config={"ENABLE_MMAP": False}
    ).generate(test_sample)

    assert genai_result.texts[0] == expected

    genai_result = ov_genai.WhisperPipeline(
        path, "CPU", {"ENABLE_MMAP": False}
    ).generate(test_sample)
    assert genai_result.texts[0] == expected


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("test_sample", get_samples_from_dataset(length=1))
@pytest.mark.precommit
def test_max_new_tokens(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(test_sample, max_new_tokens=30)["text"]

    genai_result = pipe.generate(test_sample, max_new_tokens=30)

    assert genai_result.texts[0] == expected

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    genai_result = pipe.generate(test_sample, config)
    assert genai_result.texts[0] == expected


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample", get_samples_from_dataset(language="fr", length=3)
)
@pytest.mark.precommit
def test_language_mode_fr(model_descr, test_sample):
    model_id, path = model_descr
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(
        test_sample, max_new_tokens=30, generate_kwargs={"language": "fr"}
    )

    genai_result = pipe.generate(test_sample, max_new_tokens=30, language="<|fr|>")

    assert genai_result.texts[0] == expected["text"]

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|fr|>"
    genai_result = pipe.generate(test_sample, config)

    assert genai_result.texts[0] == expected["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample", get_samples_from_dataset(language="de", length=3)
)
@pytest.mark.precommit
def test_language_mode_de(model_descr, test_sample):
    model_id, path = model_descr
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(
        test_sample, max_new_tokens=30, generate_kwargs={"language": "de"}
    )

    genai_result = pipe.generate(test_sample, max_new_tokens=30, language="<|de|>")

    assert genai_result.texts[0] == expected["text"]

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|de|>"
    genai_result = pipe.generate(test_sample, config)

    assert genai_result.texts[0] == expected["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample", get_samples_from_dataset(language="fr", length=3)
)
@pytest.mark.precommit
def test_task_mode(model_descr, test_sample):
    model_id, path = model_descr
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(
        test_sample,
        max_new_tokens=30,
        generate_kwargs={"language": "fr", "task": "translate"},
    )

    genai_result = pipe.generate(
        test_sample, max_new_tokens=30, language="<|fr|>", task="translate"
    )

    assert genai_result.texts[0] == expected["text"]

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|fr|>"
    config.task = "translate"
    genai_result = pipe.generate(test_sample, config)

    assert genai_result.texts[0] == expected["text"]

    expected = opt_pipe(
        test_sample,
        max_new_tokens=30,
        generate_kwargs={"language": "ru", "task": "translate"},
    )

    genai_result = pipe.generate(
        test_sample, max_new_tokens=30, language="<|ru|>", task="translate"
    )

    assert genai_result.texts[0] == expected["text"]

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|ru|>"
    config.task = "translate"
    genai_result = pipe.generate(test_sample, config)

    assert genai_result.texts[0] == expected["text"]

    # seems to be equivalent to translate task
    expected = opt_pipe(
        test_sample,
        max_new_tokens=30,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )

    genai_result = pipe.generate(
        test_sample, max_new_tokens=30, language="<|en|>", task="transcribe"
    )

    assert genai_result.texts[0] == expected["text"]

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|en|>"
    config.task = "transcribe"
    genai_result = pipe.generate(test_sample, config)

    assert genai_result.texts[0] == expected["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample",
    [
        *get_samples_from_dataset(language="fr", length=2),
        *get_samples_from_dataset(language="de", length=2),
        *get_samples_from_dataset(language="es", length=2),
    ],
)
@pytest.mark.precommit
def test_language_autodetect(model_descr, test_sample):
    model_id, path = model_descr
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    input_features = opt_pipe.feature_extractor(test_sample)
    language_id = opt_pipe.model.detect_language(input_features["input_features"])[0]
    # ensure detected language us not english
    assert language_id != pipe.get_generation_config().lang_to_id["<|en|>"]

    expected = opt_pipe(
        test_sample,
        max_new_tokens=30,
    )

    genai_result = pipe.generate(test_sample, max_new_tokens=30)

    assert genai_result.texts[0] == expected["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample",
    [
        *get_samples_from_dataset(language="en", length=10, long_form=True),
    ],
)
@pytest.mark.precommit
def test_return_timestamps_short_form(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)
    # long form audio not supported yet
    test_sample = test_sample[: 16000 * 30]

    expected = opt_pipe(
        test_sample,
        return_timestamps=True,
    )

    genai_result = pipe.generate(
        test_sample.tolist(),
        return_timestamps=True,
    )

    assert genai_result.texts[0] == expected["text"]

    assert len(genai_result.chunks) == len(expected["chunks"])

    for opt_chunk, genai_chunk in zip(expected["chunks"], genai_result.chunks):
        assert opt_chunk["text"] == genai_chunk.text
        assert opt_chunk["timestamp"][0] == round(genai_chunk.start_ts, 2)
        assert opt_chunk["timestamp"][1] == round(genai_chunk.end_ts, 2)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample",
    [
        *get_samples_from_dataset(language="en", length=10, long_form=True),
    ],
)
@pytest.mark.precommit
def test_return_timestamps_max_new_tokens_short_form(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)
    # long form audio not supported yet
    test_sample = test_sample[: 16000 * 30]

    expected = opt_pipe(
        test_sample,
        return_timestamps=True,
        max_new_tokens=15,
        generate_kwargs={"language": "en"},
    )

    genai_result = pipe.generate(
        test_sample.tolist(),
        max_new_tokens=15,
        return_timestamps=True,
        language="<|en|>",
    )

    assert genai_result.texts[0] == expected["text"]

    assert len(genai_result.chunks) == len(expected["chunks"])

    for opt_chunk, genai_chunk in zip(expected["chunks"], genai_result.chunks):
        assert opt_chunk["text"] == genai_chunk.text
        assert opt_chunk["timestamp"][0] == round(genai_chunk.start_ts, 2)
        if opt_chunk["timestamp"][1]:
            assert opt_chunk["timestamp"][1] == round(genai_chunk.end_ts, 2)
        else:
            assert opt_chunk["timestamp"][1] == None
            assert round(genai_chunk.end_ts, 2) == -1.0


@pytest.mark.parametrize("model_descr", get_whisper_models_list(multilingual=True))
@pytest.mark.parametrize(
    "test_sample",
    [
        *get_samples_from_dataset(language="en", length=10, long_form=True),
        *get_samples_from_dataset(language="fr", length=10, long_form=True),
    ],
)
@pytest.mark.precommit
def test_longform_audio_return_timestamps_multilingual(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(
        test_sample,
        return_timestamps=True,
    )

    genai_result = pipe.generate(
        test_sample,
        return_timestamps=True,
    )

    assert genai_result.texts[0] == expected["text"]

    assert len(genai_result.chunks) == len(expected["chunks"])

    for opt_chunk, genai_chunk in zip(expected["chunks"], genai_result.chunks):
        assert opt_chunk["text"] == genai_chunk.text
        assert opt_chunk["timestamp"][0] == round(genai_chunk.start_ts, 2)
        if opt_chunk["timestamp"][1]:
            assert opt_chunk["timestamp"][1] == round(genai_chunk.end_ts, 2)
        else:
            assert opt_chunk["timestamp"][1] == None
            assert round(genai_chunk.end_ts, 2) == -1.0


@pytest.mark.parametrize("model_descr", get_whisper_models_list(en_only=True))
@pytest.mark.parametrize(
    "test_sample",
    [
        *get_samples_from_dataset(language="en", length=10, long_form=True),
    ],
)
@pytest.mark.precommit
def test_longform_audio_return_timestamps_en(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(
        test_sample,
        return_timestamps=True,
    )

    genai_result = pipe.generate(
        test_sample,
        return_timestamps=True,
    )

    assert genai_result.texts[0] == expected["text"]

    assert len(genai_result.chunks) == len(expected["chunks"])

    for opt_chunk, genai_chunk in zip(expected["chunks"], genai_result.chunks):
        assert opt_chunk["text"] == genai_chunk.text
        assert opt_chunk["timestamp"][0] == round(genai_chunk.start_ts, 2)
        if opt_chunk["timestamp"][1]:
            assert opt_chunk["timestamp"][1] == round(genai_chunk.end_ts, 2)
        else:
            assert opt_chunk["timestamp"][1] == None
            assert round(genai_chunk.end_ts, 2) == -1.0


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample",
    [
        *get_samples_from_dataset(language="en", length=3, long_form=True),
        *get_samples_from_dataset(language="sp", length=3, long_form=True),
    ],
)
@pytest.mark.precommit
def test_longform_audio(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(test_sample, return_timestamps=True)

    genai_result = pipe.generate(test_sample)

    assert genai_result.texts[0] == expected["text"]

    assert genai_result.chunks == None


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample",
    [
        *get_samples_from_dataset(language="en", length=1),
    ],
)
@pytest.mark.precommit
def test_perf_metrics(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    result = pipe.generate(test_sample)

    perf_metrics = result.perf_metrics

    assert perf_metrics is not None

    assert perf_metrics.get_load_time() > 0
    assert perf_metrics.get_num_generated_tokens() > 0
    assert perf_metrics.get_num_input_tokens() == 0
    assert perf_metrics.get_ttft().mean > 0
    assert perf_metrics.get_tpot().mean > 0
    assert perf_metrics.get_ipot().mean > 0
    assert perf_metrics.get_throughput().mean > 0
    assert perf_metrics.get_inference_duration().mean > 0
    assert perf_metrics.get_generate_duration().mean > 0
    assert perf_metrics.get_tokenization_duration().mean == 0
    assert perf_metrics.get_detokenization_duration().mean > 0
