# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import functools
import pytest
import openvino_tokenizers
import openvino
import datasets
from transformers import WhisperProcessor, pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import gc
import json
import time
import typing
import numpy as np
import os
import pathlib
from dataclasses import dataclass


@pytest.fixture(scope="class", autouse=True)
def run_gc_after_test():
    """
    Fixture to run garbage collection after each test class.
    This is a workaround to minimize memory consumption during tests and allow the use of less powerful CI runners.
    """
    yield
    gc.collect()


def get_whisper_models_list(tiny_only=False):
    model_ids = [
        "openai/whisper-tiny",
        "distil-whisper/distil-small.en",
    ]

    if tiny_only:
        model_ids = ["openai/whisper-tiny"]

    if pytest.selected_model_ids:
        model_ids = [
            model_id
            for model_id in model_ids
            if model_id in pytest.selected_model_ids.split(" ")
        ]

    prefix = pathlib.Path(os.getenv("GENAI_MODELS_PATH_PREFIX", ""))
    return [(model_id, prefix / model_id.split("/")[1]) for model_id in model_ids]


# used whisper models are relatively small
# cache them in memory to speedup tests
@functools.lru_cache()
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
        processor.save_pretrained(path)

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
        ov_genai.WhisperPipeline(path, "CPU", **{"ENABLE_MMAP": False}),
    )


@dataclass
class GenerationConfig:
    task: str | None = None
    language: str | None = None
    return_timestamps: bool = False
    max_new_tokens: int | None = None
    streamer: typing.Callable[[str], bool] | None = None


def run_huggingface(
    pipeline,
    sample,
    config: GenerationConfig | None = None,
):
    if not config:
        config = GenerationConfig()

    return pipeline(
        sample,
        max_new_tokens=config.max_new_tokens,
        return_timestamps=config.return_timestamps,
        generate_kwargs={"language": config.language, "task": config.task},
    )


def run_genai(
    pipeline: ov_genai.WhisperPipeline,
    sample,
    config: GenerationConfig | None = None,
):
    if not config:
        config = GenerationConfig()

    genai_config = pipeline.get_generation_config()

    if config.max_new_tokens:
        genai_config.max_new_tokens = config.max_new_tokens
    genai_config.return_timestamps = config.return_timestamps
    genai_config.task = config.task
    genai_config.language = f"<|{config.language}|>" if config.language else None

    return pipeline.generate(sample, genai_config, streamer=config.streamer)


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


def run_pipeline_with_ref(
    model_id: str,
    tmp_path: str,
    sample: np.ndarray | list[np.ndarray],
    generation_config: GenerationConfig | None = None,
    print_infer_time=False,
):
    _, _, hf_pipe, genai_pipe = read_whisper_model((model_id, tmp_path))

    if type(sample) is np.ndarray and len(sample.shape) == 1:
        sample = np.expand_dims(sample, 0)

    hf_infer_time, genai_infer_time = 0, 0
    hf_result, genai_result = None, None
    for _sample in sample:
        start = time.time()
        genai_result = run_genai(genai_pipe, _sample, generation_config)
        genai_infer_time += time.time() - start

        start = time.time()
        hf_result = run_huggingface(hf_pipe, _sample, generation_config)
        hf_infer_time += time.time() - start

        compare_results(hf_result, genai_result)

    if print_infer_time:
        print(f"\nInference time HF: {hf_infer_time:.2f} GenAI: {genai_infer_time:.2f}")

    assert hf_result is not None
    assert genai_result is not None

    return hf_result, genai_result


def compare_results(hf_result, genai_result):
    assert genai_result.texts[0] == hf_result["text"]

    # transformers 4.47 updated return_timestamps implementation
    # enable once genai implementation aligned with trasformets. Ticket 160205.
    return

    if "chunks" not in hf_result and genai_result.chunks is None:
        return

    assert len(genai_result.chunks) == len(hf_result["chunks"])

    for opt_chunk, genai_chunk in zip(hf_result["chunks"], genai_result.chunks):
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
    get_samples_from_dataset(language="en", length=1),
)
@pytest.mark.precommit
def test_smoke(model_descr, test_sample):
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=test_sample,
    )


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
        models_path=path, device="CPU", **{"ENABLE_MMAP": False}
    ).generate(test_sample)

    assert genai_result.texts[0] == expected

    genai_result = ov_genai.WhisperPipeline(
        path, "CPU", **{"ENABLE_MMAP": False}
    ).generate(test_sample)
    assert genai_result.texts[0] == expected


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("test_sample", get_samples_from_dataset(length=1))
@pytest.mark.precommit
def test_max_new_tokens(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(test_sample, max_new_tokens=10)

    genai_result = pipe.generate(test_sample, max_new_tokens=10)

    compare_results(expected, genai_result)

    config = pipe.get_generation_config()
    config.max_new_tokens = 10
    genai_result = pipe.generate(test_sample, config)
    compare_results(expected, genai_result)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_samples",
    [
        (get_samples_from_dataset(language="fr", length=1), "fr"),
        (get_samples_from_dataset(language="de", length=1), "de"),
    ],
)
@pytest.mark.precommit
def test_language_mode(model_descr, test_samples):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)
    samples, language = test_samples

    expected = opt_pipe(
        samples[0], max_new_tokens=30, generate_kwargs={"language": language}
    )

    genai_result = pipe.generate(
        samples[0], max_new_tokens=30, language=f"<|{language}|>"
    )

    compare_results(expected, genai_result)

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = f"<|{language}|>"
    genai_result = pipe.generate(samples[0], config)

    compare_results(expected, genai_result)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample", get_samples_from_dataset(language="fr", length=1)
)
@pytest.mark.precommit
def test_task_mode(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    expected = opt_pipe(
        test_sample,
        max_new_tokens=30,
        generate_kwargs={"language": "fr", "task": "translate"},
    )

    genai_result = pipe.generate(
        test_sample, max_new_tokens=30, language="<|fr|>", task="translate"
    )

    compare_results(expected, genai_result)

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|fr|>"
    config.task = "translate"
    genai_result = pipe.generate(test_sample, config)

    compare_results(expected, genai_result)

    # seems to be equivalent to translate task
    expected = opt_pipe(
        test_sample,
        max_new_tokens=30,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )

    genai_result = pipe.generate(
        test_sample, max_new_tokens=30, language="<|en|>", task="transcribe"
    )

    compare_results(expected, genai_result)

    config = pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|en|>"
    config.task = "transcribe"
    genai_result = pipe.generate(test_sample, config)

    compare_results(expected, genai_result)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample",
    [
        *get_samples_from_dataset(language="fr", length=1),
        *get_samples_from_dataset(language="de", length=1),
        *get_samples_from_dataset(language="es", length=1),
    ],
)
@pytest.mark.precommit
def test_language_autodetect(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    input_features = opt_pipe.feature_extractor(test_sample)
    language_id = opt_pipe.model.detect_language(input_features["input_features"])[0]
    # ensure detected language us not english
    assert language_id != pipe.get_generation_config().lang_to_id["<|en|>"]

    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=test_sample,
        generation_config=GenerationConfig(max_new_tokens=30),
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("test_sample", get_samples_from_dataset(length=1))
@pytest.mark.precommit
def test_return_timestamps_short_form(model_descr, test_sample):
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=test_sample,
        generation_config=GenerationConfig(return_timestamps=True),
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("test_sample", get_samples_from_dataset(length=1))
@pytest.mark.precommit
def test_return_timestamps_max_new_tokens_short_form(model_descr, test_sample):
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=test_sample,
        generation_config=GenerationConfig(
            return_timestamps=True, language="en", max_new_tokens=30
        ),
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize(
    "test_sample", get_samples_from_dataset(length=10, long_form=True)
)
@pytest.mark.precommit
def test_longform_audio(model_descr, test_sample):
    streamer_result = []

    hf_result, genai_result = run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=test_sample,
        generation_config=GenerationConfig(
            return_timestamps=True,
            streamer=lambda x: streamer_result.append(x),
        ),
    )

    assert "".join(streamer_result) == hf_result["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.precommit
def test_shortform(model_descr):
    samples = []
    ds = datasets.load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )

    for ds_row in ds:
        samples.append(ds_row["audio"]["array"])

    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=samples,
        print_infer_time=True,
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "test_sample",
    get_samples_from_dataset(length=1),
)
@pytest.mark.precommit
def test_initial_prompt_hotwords(model_descr, test_sample):
    model_id, path, opt_pipe, pipe = read_whisper_model(model_descr)

    result = pipe.generate(test_sample)

    assert "Joel Keaton" in result.texts[0]
    assert "Joel Kyton" not in result.texts[0]

    result = pipe.generate(test_sample, initial_prompt="Joel Kyton")

    assert "Joel Keaton" not in result.texts[0]
    assert "Joel Kyton" in result.texts[0]

    result = pipe.generate(test_sample, hotwords="Joel Kyton")

    assert "Joel Keaton" not in result.texts[0]
    assert "Joel Kyton" in result.texts[0]


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
    assert perf_metrics.get_detokenization_duration().mean > 0
    assert perf_metrics.get_features_extraction_duration().mean > 0

    # assert that calculating statistics manually from the raw counters we get the same results as from PerfMetrics
    whisper_raw_metrics = perf_metrics.whisper_raw_metrics

    raw_dur = np.array(whisper_raw_metrics.features_extraction_durations) / 1000
    mean_dur, std_dur = perf_metrics.get_features_extraction_duration()
    assert np.allclose(mean_dur, np.mean(raw_dur))
    assert np.allclose(std_dur, np.std(raw_dur))
