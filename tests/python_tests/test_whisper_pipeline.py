# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import functools
import pytest
import sys
import openvino_tokenizers
import openvino
import datasets
from transformers import WhisperProcessor, pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import gc
import json
import typing
import numpy as np
import pathlib
import importlib.metadata as metadata
from packaging.version import parse
from utils.constants import get_ov_cache_models_dir, extra_generate_kwargs

from utils.network import retry_request
from typing import Any

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

    prefix = get_ov_cache_models_dir()
    return [(model_id, prefix / model_id.split("/")[1]) for model_id in model_ids]


# used whisper models are relatively small
# cache them in memory to speedup tests
@functools.lru_cache()
def read_whisper_model(params, stateful=True):
    model_id, path = params
    if not stateful:
        path = pathlib.Path(f"{path}_with_past")

    if not (path / "openvino_encoder_model.xml").exists():
        save_model(model_id=model_id, tmp_path=path, stateful=stateful)

    opt_model = retry_request(lambda: OVModelForSpeechSeq2Seq.from_pretrained(
        path,
        trust_remote_code=True,
        compile=False,
        device="CPU",
        load_in_8bit=False,
        local_files_only=True,
    ))

    processor = retry_request(lambda: WhisperProcessor.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
    ))

    hf_pipe = pipeline(
        "automatic-speech-recognition",
        model=opt_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )

    return (
        model_id,
        path,
        hf_pipe,
        ov_genai.WhisperPipeline(path, "CPU", ENABLE_MMAP=False),
    )


def save_model(model_id: str, tmp_path: pathlib.Path, stateful=True):
    tokenizer = retry_request(lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
        tokenizer,
        with_detokenizer=True,
        clean_up_tokenization_spaces=False,
    )

    openvino.save_model(ov_tokenizer, tmp_path / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, tmp_path / "openvino_detokenizer.xml")

    # to store tokenizer config jsons with special tokens
    tokenizer.save_pretrained(tmp_path)

    opt_model = retry_request(lambda: OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        export=True,
        trust_remote_code=True,
        stateful=stateful,
        compile=False,
        device="CPU",
        load_in_8bit=False,
    ))
    opt_model.generation_config.save_pretrained(tmp_path)
    opt_model.config.save_pretrained(tmp_path)
    opt_model.save_pretrained(tmp_path)

    processor = retry_request(lambda: WhisperProcessor.from_pretrained(model_id, trust_remote_code=True))
    processor.save_pretrained(tmp_path)


def run_huggingface(
    pipeline,
    sample,
    config: ov_genai.WhisperGenerationConfig | None = None,
):
    if not config:
        config = ov_genai.WhisperGenerationConfig()

    from optimum.intel.utils.import_utils import is_transformers_version
    if is_transformers_version(">=", "4.51"):
        if hasattr(pipeline.model.config, 'forced_decoder_ids'):
            pipeline.model.config.forced_decoder_ids = None

        if hasattr(pipeline.model, 'generation_config'):
            if hasattr(pipeline.model.generation_config, 'forced_decoder_ids'):
                pipeline.model.generation_config.forced_decoder_ids = None

    return pipeline(
        sample,
        return_timestamps=config.return_timestamps,
        generate_kwargs={
            "language": config.language,
            "task": config.task,
            "max_new_tokens": min(config.max_new_tokens, 444),
            "top_p": config.top_p,
            "do_sample": config.do_sample,
            "num_beams": config.num_beams,
        } | extra_generate_kwargs(),
    )


def run_genai(
    pipeline: ov_genai.WhisperPipeline,
    sample,
    config: ov_genai.WhisperGenerationConfig | None = None,
    streamer: typing.Callable[[str], bool] | None = None,
):
    if not config:
        config = ov_genai.WhisperGenerationConfig()

    genai_config = pipeline.get_generation_config()

    genai_config.max_new_tokens = config.max_new_tokens
    genai_config.return_timestamps = config.return_timestamps
    genai_config.task = config.task
    genai_config.language = f"<|{config.language}|>" if config.language else None
    genai_config.do_sample = config.do_sample
    genai_config.top_p = config.top_p
    genai_config.num_beams = config.num_beams

    return pipeline.generate(sample, genai_config, streamer=streamer)

MAX_DATASET_LENGTH = 30

@functools.lru_cache(16)
def get_whisper_dataset(language: str, long_form: bool) -> list:
    # TODO: temporary always use long_form for until "mozilla-foundation/common_voice_11_0" 
    # https://github.com/huggingface/datasets/issues/7647 dataset is fixed for streaming mode
    # if not long_form:
    if False:  
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
    ds = ds.take(MAX_DATASET_LENGTH)

    return [x["audio"]["array"] for x in ds]

@pytest.fixture
def sample_from_dataset(request):
    language = request.param.get("language", "en")
    long_form = request.param.get("long_form", False)

    sample_id = request.param.get("sample_id", 0)
    samples = get_whisper_dataset(language, long_form)
    assert sample_id < MAX_DATASET_LENGTH

    return samples[sample_id]

def get_fixture_params_for_n_whisper_dataset_samples(n: int, language: str = "en", long_form : bool = False) -> list[dict[str, Any]]:
    return [{"language": language, "long_form": long_form, "sample_id": i} for i in range(n)]

def run_pipeline_with_ref(
    model_id: str,
    tmp_path: str,
    sample: np.ndarray | list[np.ndarray],
    generation_config: ov_genai.WhisperGenerationConfig | None = None,
    streamer: typing.Callable[[str], bool] | None = None,
):
    _, _, hf_pipe, genai_pipe = read_whisper_model((model_id, tmp_path))
    _, _, _, genai_with_past_pipe = read_whisper_model(
        (model_id, tmp_path), stateful=False
    )

    if type(sample) is np.ndarray and len(sample.shape) == 1:
        sample = np.expand_dims(sample, 0)

    for _sample in sample:
        genai_result = run_genai(genai_pipe, _sample, generation_config, streamer)
        hf_result = run_huggingface(hf_pipe, _sample, generation_config)

        compare_results(hf_result, genai_result)

        genai_with_past_result = run_genai(
            genai_with_past_pipe, _sample, generation_config, streamer
        )

        compare_results(hf_result, genai_with_past_result)


def compare_results(hf_result, genai_result):
    assert genai_result.texts[0] == hf_result["text"]

    # transformers 4.47 updated return_timestamps implementation
    # remove once genai implementation aligned with transformers. Ticket 160205.
    transformers_version_greater_4_47 = parse(
        metadata.version("transformers")
    ) >= parse("4.47.0")

    if transformers_version_greater_4_47:
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
@pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
@pytest.mark.precommit
def test_smoke(model_descr, sample_from_dataset):
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=sample_from_dataset,
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.precommit
def test_whisper_config_constructor(model_descr):
    model_id, path = model_descr

    config = ov_genai.WhisperGenerationConfig(path / "generation_config.json")

    with open(path / "generation_config.json", encoding="utf-8") as f:
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
@pytest.mark.parametrize("sample_from_dataset", [{"language" : "en", "sample_id": 0}], indirect=True)
@pytest.mark.precommit
def test_whisper_constructors(model_descr, sample_from_dataset):
    model_id, path, hf_pipe, genai_pipe = read_whisper_model(model_descr)

    expected = hf_pipe(sample_from_dataset)["text"]

    genai_result = ov_genai.WhisperPipeline(
        models_path=path, device="CPU", **{"ENABLE_MMAP": False}
    ).generate(sample_from_dataset)

    assert genai_result.texts[0] == expected

    genai_result = ov_genai.WhisperPipeline(
        path, "CPU", **{"ENABLE_MMAP": False}
    ).generate(sample_from_dataset)
    assert genai_result.texts[0] == expected


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
@pytest.mark.precommit
def test_max_new_tokens(model_descr, sample_from_dataset):
    model_id, path, hf_pipe, genai_pipe = read_whisper_model(model_descr)

    expected = hf_pipe(sample_from_dataset, max_new_tokens=10)

    genai_result = genai_pipe.generate(sample_from_dataset, max_new_tokens=10)

    compare_results(expected, genai_result)

    config = genai_pipe.get_generation_config()
    config.max_new_tokens = 10
    genai_result = genai_pipe.generate(sample_from_dataset, config)
    compare_results(expected, genai_result)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("language", ["fr", "de"])
@pytest.mark.precommit
def test_language_mode(model_descr, language):
    model_id, path, hf_pipe, genai_pipe = read_whisper_model(model_descr)
    sample = get_whisper_dataset(language, long_form=False)[0]

    expected = hf_pipe(
        sample, max_new_tokens=30, generate_kwargs={"language": language}
    )

    genai_result = genai_pipe.generate(
        sample, max_new_tokens=30, language=f"<|{language}|>"
    )

    compare_results(expected, genai_result)

    config = genai_pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = f"<|{language}|>"
    genai_result = genai_pipe.generate(sample, config)

    compare_results(expected, genai_result)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", get_fixture_params_for_n_whisper_dataset_samples(n=1, language="fr"), indirect=True)
@pytest.mark.precommit
def test_task_mode(model_descr, sample_from_dataset):
    model_id, path, hf_pipe, genai_pipe = read_whisper_model(model_descr)

    expected = hf_pipe(
        sample_from_dataset,
        max_new_tokens=30,
        generate_kwargs={"language": "fr", "task": "translate"},
    )

    genai_result = genai_pipe.generate(
        sample_from_dataset, max_new_tokens=30, language="<|fr|>", task="translate"
    )

    compare_results(expected, genai_result)

    config = genai_pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|fr|>"
    config.task = "translate"
    genai_result = genai_pipe.generate(sample_from_dataset, config)

    compare_results(expected, genai_result)

    # seems to be equivalent to translate task
    expected = hf_pipe(
        sample_from_dataset,
        max_new_tokens=30,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )

    genai_result = genai_pipe.generate(
        sample_from_dataset, max_new_tokens=30, language="<|en|>", task="transcribe"
    )

    compare_results(expected, genai_result)

    config = genai_pipe.get_generation_config()
    config.max_new_tokens = 30
    config.language = "<|en|>"
    config.task = "transcribe"
    genai_result = genai_pipe.generate(sample_from_dataset, config)

    compare_results(expected, genai_result)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=1, language="fr"),
                                                 *get_fixture_params_for_n_whisper_dataset_samples(n=1, language="de"),
                                                 *get_fixture_params_for_n_whisper_dataset_samples(n=1, language="es")], indirect=True)
@pytest.mark.precommit
def test_language_autodetect(model_descr, sample_from_dataset):
    model_id, path, hf_pipe, genai_pipe = read_whisper_model(model_descr)

    input_features = hf_pipe.feature_extractor(sample_from_dataset)
    language_id = hf_pipe.model.detect_language(input_features["input_features"])[0]
    # ensure detected language us not english
    assert language_id != genai_pipe.get_generation_config().lang_to_id["<|en|>"]

    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=sample_from_dataset,
        generation_config=ov_genai.WhisperGenerationConfig(max_new_tokens=30),
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=1)], indirect=True)
@pytest.mark.precommit
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_return_timestamps_short_form(model_descr, sample_from_dataset):
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=sample_from_dataset,
        generation_config=ov_genai.WhisperGenerationConfig(return_timestamps=True),
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=1)], indirect=True)
@pytest.mark.precommit
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_return_timestamps_max_new_tokens_short_form(model_descr, sample_from_dataset):
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=sample_from_dataset,
        generation_config=ov_genai.WhisperGenerationConfig(
            return_timestamps=True, language="en", max_new_tokens=30
        ),
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=10, long_form=True)], indirect=True)
@pytest.mark.precommit
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_longform_audio(model_descr, sample_from_dataset):
    _, _, hf_pipe, genai_pipe = read_whisper_model(model_descr)

    streamer_result = []

    genai_result = run_genai(
        genai_pipe,
        sample_from_dataset,
        config=ov_genai.WhisperGenerationConfig(return_timestamps=True),
        streamer=lambda x: streamer_result.append(x),
    )

    hf_result = run_huggingface(
        hf_pipe,
        sample_from_dataset,
        config=ov_genai.WhisperGenerationConfig(return_timestamps=True),
    )

    compare_results(hf_result, genai_result)

    assert "".join(streamer_result) == hf_result["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=2, long_form=True)], indirect=True)
@pytest.mark.precommit
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_longform_audio_with_past(model_descr, sample_from_dataset):
    _, _, hf_pipe, genai_pipe = read_whisper_model(model_descr, stateful=True)

    streamer_result = []

    genai_result = run_genai(
        genai_pipe,
        sample_from_dataset,
        config=ov_genai.WhisperGenerationConfig(return_timestamps=True),
        streamer=lambda x: streamer_result.append(x),
    )

    hf_result = run_huggingface(
        hf_pipe,
        sample_from_dataset,
        config=ov_genai.WhisperGenerationConfig(return_timestamps=True),
    )

    compare_results(hf_result, genai_result)

    assert "".join(streamer_result) == hf_result["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.precommit
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
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
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=2, long_form=True)], indirect=True)
@pytest.mark.precommit
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_beam_search(model_descr, sample_from_dataset):
    # use only 30 seconds of audio due to beam search results wrong with enabled timestamps
    # ticket: 167239
    sample_from_dataset = sample_from_dataset[:30 * 16000]
    _, _, hf_pipe, genai_pipe = read_whisper_model(model_descr)
    generation_config=ov_genai.WhisperGenerationConfig(
        num_beams=2,
    )

    genai_result = run_genai(genai_pipe, sample_from_dataset, generation_config)
    hf_result = run_huggingface(hf_pipe, sample_from_dataset, generation_config)

    compare_results(hf_result, genai_result)

@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language" : "en", "sample_id": 0}], indirect=True)
@pytest.mark.precommit
def test_initial_prompt_hotwords(model_descr, sample_from_dataset):
    model_id, path, hf_pipe, genai_pipe = read_whisper_model(model_descr)

    result = genai_pipe.generate(sample_from_dataset)

    assert "Joel Keaton" in result.texts[0]
    assert "Joel Kyton" not in result.texts[0]

    result = genai_pipe.generate(sample_from_dataset, initial_prompt="Joel Kyton")

    assert "Joel Keaton" not in result.texts[0]
    assert "Joel Kyton" in result.texts[0]

    result = genai_pipe.generate(sample_from_dataset, hotwords="Joel Kyton")

    assert "Joel Keaton" not in result.texts[0]
    assert "Joel Kyton" in result.texts[0]


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language" : "en", "sample_id": 0}], indirect=True)
@pytest.mark.precommit
def test_random_sampling(model_descr, sample_from_dataset):
    _, _, hf_pipe, genai_pipe = read_whisper_model(model_descr)

    config = ov_genai.WhisperGenerationConfig(do_sample=True, top_p=0.01)

    genai_result = run_genai(
        genai_pipe,
        sample_from_dataset,
        config=config,
    )

    hf_result = run_huggingface(
        hf_pipe,
        sample_from_dataset,
        config=config,
    )

    compare_results(hf_result, genai_result)

    config.top_p = 0.6

    genai_result = run_genai(
        genai_pipe,
        sample_from_dataset,
        config=config,
    )

    hf_result = run_huggingface(
        hf_pipe,
        sample_from_dataset,
        config=config,
    )

    assert genai_result.texts[0] != hf_result["text"]


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language" : "en", "sample_id": 0}], indirect=True)
@pytest.mark.precommit
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_perf_metrics(model_descr, sample_from_dataset):
    model_id, path, hf_pipe, genai_pipe = read_whisper_model(model_descr)

    result = genai_pipe.generate(sample_from_dataset)

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


@pytest.fixture(params=[
    "DeprecatedBaseStreamer",
    "DeprecatedChunkStreamer",
    "DeprecatedChunkWriteStreamer",
    "Streamer",
    "streamer_callback",
    "streamer_bool_callback"
])
def streamer_for_test(request):
    class ResultHandler:
        def __init__(self, container: list[int] | list[str]):
            self.container: list[int] | list[str] = container

        def decode(self, tokenizer: ov_genai.Tokenizer) -> str:
            if type(self.container[0]) == int:
                return tokenizer.decode(typing.cast(list[int], self.container))
            return ''.join(typing.cast(list[str], self.container))

        def reset(self) -> None:
            self.container.clear()


    class DeprecatedBaseStreamer(ov_genai.StreamerBase):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = []

        def put(self, token: int) -> bool:
            self.tokens.append(token)
            return False

        def end(self) -> None:
            pass

    if request.param == 'DeprecatedBaseStreamer':
        streamer = DeprecatedBaseStreamer()
        return streamer, ResultHandler(streamer.tokens)


    class DeprecatedChunkStreamer(ov_genai.ChunkStreamerBase):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = []

        def put(self, token: int) -> bool:
            self.tokens.append(token)
            return False

        def put_chunk(self, tokens: list[int]) -> bool:
            self.tokens += tokens
            return False

        def end(self) -> None:
            pass

    if request.param == 'DeprecatedChunkStreamer':
        streamer = DeprecatedChunkStreamer()
        return streamer, ResultHandler(streamer.tokens)

    class DeprecatedChunkWriteStreamer(ov_genai.ChunkStreamerBase):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = []

        def write(self, token: int | list[int]) -> ov_genai.StreamingStatus:
            if type(token) == list:
                self.tokens += token
            else:
                self.tokens.append(token)
            return ov_genai.StreamingStatus.RUNNING

        def end(self) -> None:
            pass

    if request.param == 'DeprecatedChunkWriteStreamer':
        streamer = DeprecatedChunkWriteStreamer()
        return streamer, ResultHandler(streamer.tokens)
    
    class Streamer(ov_genai.StreamerBase):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = []

        def write(self, token: int | list[int]) -> ov_genai.StreamingStatus:
            if type(token) == list:
                self.tokens += token
            else:
                self.tokens.append(token)
            return ov_genai.StreamingStatus.RUNNING

        def end(self) -> None:
            pass

    if request.param == 'Streamer':
        streamer = Streamer()
        return streamer, ResultHandler(streamer.tokens)


    if request.param == 'streamer_callback':
        texts = []
        def streamer_callback(subword):
            texts.append(subword)
            return ov_genai.StreamingStatus.RUNNING
        return streamer_callback, ResultHandler(texts)

    if request.param == 'streamer_bool_callback':
        texts = []
        def streamer_bool_callback(subword):
            texts.append(subword)
            return False
        return streamer_bool_callback, ResultHandler(texts)

@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language" : "en", "sample_id": 0}], indirect=True)
@pytest.mark.precommit
def test_streamers(model_descr, sample_from_dataset, streamer_for_test):
    _, _, _, genai_pipe = read_whisper_model(model_descr)

    streamer, result_handler = streamer_for_test

    result = genai_pipe.generate(sample_from_dataset, streamer=streamer)

    expected = result.texts[0]

    assert expected == result_handler.decode(genai_pipe.get_tokenizer())
    result_handler.reset()

    config = genai_pipe.get_generation_config()
    genai_pipe.generate(sample_from_dataset, config, streamer)

    assert expected == result_handler.decode(genai_pipe.get_tokenizer())
    result_handler.reset()

    genai_pipe.generate(sample_from_dataset, config, streamer=streamer)

    assert expected == result_handler.decode(genai_pipe.get_tokenizer())
    result_handler.reset()

    genai_pipe.generate(sample_from_dataset, generation_config=config, streamer=streamer)

    assert expected == result_handler.decode(genai_pipe.get_tokenizer())
    result_handler.reset()

    genai_pipe.generate(sample_from_dataset, return_timestamps=True, streamer=streamer)

    assert expected == result_handler.decode(genai_pipe.get_tokenizer())
    result_handler.reset()
