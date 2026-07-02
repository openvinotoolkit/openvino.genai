# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import enum
import sys
import utils.patch_pyav_for_servercore as patch_pyav_for_servercore

patch_pyav_for_servercore.install_av_stub_module_for_windows()

# ruff: noqa: E402
import openvino_genai as ov_genai
import functools
import pytest
import openvino_tokenizers
import openvino
import datasets
from transformers import AutoProcessor, AutoTokenizer
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from huggingface_hub import snapshot_download
import gc
import json
import typing
import numpy as np
import pathlib
import importlib.metadata as metadata
from packaging.version import parse
from utils.constants import get_ov_cache_converted_models_dir, extra_generate_kwargs

from utils.network import retry_request
from utils.atomic_download import AtomicDownloadManager
from typing import Any, Literal
from difflib import SequenceMatcher

from utils.dataset_utils import load_dataset_via_snapshot
from utils.qwen3_asr import Qwen3ASROptimumPipeline, skip_if_qwen3_asr_package_is_unavailable


class PipelineType(enum.Enum):
    WHISPER = "whisper"
    ASR = "asr"


def get_pipeline_cls(pipeline_type: PipelineType):
    return ov_genai.ASRPipeline if pipeline_type == PipelineType.ASR else ov_genai.WhisperPipeline


def get_config_cls(pipeline_type: PipelineType):
    return ov_genai.ASRGenerationConfig if pipeline_type == PipelineType.ASR else ov_genai.WhisperGenerationConfig


def get_word_text(word, pipeline_type: PipelineType):
    return word.text if pipeline_type == PipelineType.ASR else word.word


def get_raw_metrics(perf_metrics, pipeline_type: PipelineType):
    return perf_metrics.asr_raw_metrics if pipeline_type == PipelineType.ASR else perf_metrics.whisper_raw_metrics


@pytest.fixture(params=[PipelineType.WHISPER, PipelineType.ASR])
def pipeline_type(request):
    return request.param


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
        model_ids = [model_id for model_id in model_ids if model_id in pytest.selected_model_ids.split(" ")]

    prefix = get_ov_cache_converted_models_dir()
    return [(model_id, prefix / model_id.split("/")[1]) for model_id in model_ids]


QWEN3_ASR_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-asr"


# used whisper models are relatively small
# cache them in memory to speedup tests
@functools.lru_cache()
def read_asr_model(params, word_timestamps=False, pipeline_type=PipelineType.WHISPER):
    model_id, path = params
    if model_id == QWEN3_ASR_MODEL_ID:
        skip_if_qwen3_asr_package_is_unavailable()

    manager = AtomicDownloadManager(path)
    if not manager.is_complete() and not (path / "openvino_encoder_model.xml").exists():
        save_model(model_id=model_id, tmp_path=path)

    opt_model = retry_request(
        lambda: OVModelForSpeechSeq2Seq.from_pretrained(
            path,
            trust_remote_code=True,
            compile=False,
            device="CPU",
            load_in_8bit=False,
            local_files_only=True,
        )
    )

    processor = retry_request(
        lambda: AutoProcessor.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )
    )

    if model_id == QWEN3_ASR_MODEL_ID:
        hf_pipe = Qwen3ASROptimumPipeline(
            model=opt_model,
            processor=processor,
        )
    else:
        hf_pipe = AutomaticSpeechRecognitionPipeline(
            model=opt_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )

    properties = {}
    if word_timestamps:
        properties["word_timestamps"] = True
    pipeline_cls = get_pipeline_cls(pipeline_type)

    return (
        model_id,
        path,
        hf_pipe,
        pipeline_cls(path, "CPU", **properties, ENABLE_MMAP=False),
    )


def save_model(model_id: str, tmp_path: pathlib.Path):
    manager = AtomicDownloadManager(tmp_path)

    def save_to_temp(temp_path: pathlib.Path) -> None:
        model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
        tokenizer = retry_request(lambda: AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True))
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
            tokenizer,
            with_detokenizer=True,
            clean_up_tokenization_spaces=False,
        )

        openvino.save_model(ov_tokenizer, temp_path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, temp_path / "openvino_detokenizer.xml")

        tokenizer.save_pretrained(temp_path)

        opt_model = retry_request(
            lambda: OVModelForSpeechSeq2Seq.from_pretrained(
                model_cached,
                export=True,
                trust_remote_code=True,
                compile=False,
                device="CPU",
                load_in_8bit=False,
            )
        )
        opt_model.generation_config.save_pretrained(temp_path)
        opt_model.config.save_pretrained(temp_path)
        opt_model.save_pretrained(temp_path)

        processor = retry_request(lambda: AutoProcessor.from_pretrained(model_cached, trust_remote_code=True))
        processor.save_pretrained(temp_path)

    manager.execute(save_to_temp)


def run_huggingface(
    pipeline,
    sample,
    config: ov_genai.ASRGenerationConfig | ov_genai.WhisperGenerationConfig | None = None,
):
    if not config:
        config = ov_genai.ASRGenerationConfig()

    from optimum.intel.utils.import_utils import is_transformers_version

    if is_transformers_version(">=", "4.51"):
        if hasattr(pipeline.model.config, "forced_decoder_ids"):
            pipeline.model.config.forced_decoder_ids = None

        if hasattr(pipeline.model, "generation_config"):
            if hasattr(pipeline.model.generation_config, "forced_decoder_ids"):
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
        }
        | extra_generate_kwargs(),
    )


def run_genai(
    pipeline,
    sample,
    config: ov_genai.ASRGenerationConfig | ov_genai.WhisperGenerationConfig | None = None,
    streamer: typing.Callable[[str], bool] | None = None,
):
    if not config:
        config = ov_genai.ASRGenerationConfig()

    genai_config = pipeline.get_generation_config()

    genai_config.max_new_tokens = config.max_new_tokens
    genai_config.return_timestamps = config.return_timestamps
    genai_config.task = config.task
    genai_config.language = f"<|{config.language}|>" if config.language else None
    genai_config.do_sample = config.do_sample
    genai_config.top_p = config.top_p
    genai_config.num_beams = config.num_beams
    genai_config.word_timestamps = config.word_timestamps
    if config.alignment_heads:
        genai_config.alignment_heads = config.alignment_heads

    return pipeline.generate(sample, genai_config, streamer=streamer)


MAX_DATASET_LENGTH = 30


@functools.lru_cache(16)
def get_audio_dataset(long_form: bool) -> list:
    if not long_form:
        ds = load_dataset_via_snapshot("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    else:
        ds = load_dataset_via_snapshot(
            "distil-whisper/meanwhile",
            split="test",
            streaming=True,
        )
    ds = typing.cast(datasets.IterableDataset, ds)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    ds = ds.take(MAX_DATASET_LENGTH)

    return [x["audio"]["array"] for x in ds]


@functools.lru_cache(16)
def get_multilingual_audio_dataset(language: Literal["de", "fr", "es"]) -> list:
    mls_config = {"de": "german", "fr": "french", "es": "spanish"}
    # dataset is too big (450gb) for snapshot download
    ds = retry_request(
        lambda: datasets.load_dataset(
            "facebook/multilingual_librispeech",
            mls_config[language],
            split="test",
            streaming=True,
        )
    )
    ds = typing.cast(datasets.IterableDataset, ds)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    ds = ds.take(1)

    return [x["audio"]["array"] for x in ds]


@pytest.fixture
def sample_from_dataset(request):
    long_form = request.param.get("long_form", False)

    sample_id = request.param.get("sample_id", 0)
    samples = get_audio_dataset(long_form)
    assert sample_id < MAX_DATASET_LENGTH

    return samples[sample_id]


def get_fixture_params_for_n_whisper_dataset_samples(n: int, long_form: bool = False) -> list[dict[str, Any]]:
    return [{"long_form": long_form, "sample_id": i} for i in range(n)]


@pytest.fixture
def sample_from_multilingual_dataset(request):
    language = request.param
    samples = get_multilingual_audio_dataset(language)
    return samples[0]


def run_pipeline_with_ref(
    model_id: str,
    tmp_path: str,
    sample: np.ndarray | list[np.ndarray],
    generation_config: ov_genai.ASRGenerationConfig | ov_genai.WhisperGenerationConfig | None = None,
    streamer: typing.Callable[[str], bool] | None = None,
    pipeline_type: PipelineType = PipelineType.WHISPER,
):
    _, _, hf_pipe, genai_pipe = read_asr_model((model_id, tmp_path), pipeline_type=pipeline_type)

    if type(sample) is np.ndarray and len(sample.shape) == 1:
        sample = np.expand_dims(sample, 0)

    for _sample in sample:
        genai_result = run_genai(genai_pipe, _sample, generation_config, streamer)
        hf_result = run_huggingface(hf_pipe, _sample, generation_config)

        compare_results(hf_result, genai_result)


def compare_results(hf_result, genai_result):
    assert genai_result.texts[0] == hf_result["text"]

    # transformers 4.47 updated return_timestamps implementation
    # remove once genai implementation aligned with transformers. Ticket 160205.
    transformers_version_greater_4_47 = parse(metadata.version("transformers")) >= parse("4.47.0")

    if transformers_version_greater_4_47:
        return

    if "chunks" not in hf_result and genai_result.chunks is None:
        return

    genai_chunks = (
        genai_result.chunks[0]
        if len(genai_result.chunks) and isinstance(genai_result.chunks[0], list)
        else genai_result.chunks
    )
    assert len(genai_chunks) == len(hf_result["chunks"])

    for opt_chunk, genai_chunk in zip(hf_result["chunks"], genai_chunks):
        assert opt_chunk["text"] == genai_chunk.text
        assert opt_chunk["timestamp"][0] == round(genai_chunk.start_ts, 2)
        if opt_chunk["timestamp"][1]:
            assert opt_chunk["timestamp"][1] == round(genai_chunk.end_ts, 2)
        else:
            assert opt_chunk["timestamp"][1] == None
            assert round(genai_chunk.end_ts, 2) == -1.0


MODEL_PIPELINE_PAIRS = [
    ("openai/whisper-tiny", PipelineType.ASR),
    ("distil-whisper/distil-small.en", PipelineType.ASR),
    (QWEN3_ASR_MODEL_ID, PipelineType.ASR),
    # test backward compatibility for tiny model only
    ("openai/whisper-tiny", PipelineType.WHISPER),
]


def get_model_pipeline_pair_id(model_pipeline_pair):
    model_id, pipeline_type = model_pipeline_pair[:2]
    return f"pipeline_{pipeline_type.name}_{model_id.split('/')[-1]}"


def get_model_pipeline_pair_params(model_pipeline_pairs=MODEL_PIPELINE_PAIRS):
    return [
        pytest.param(model_pipeline_pair, id=get_model_pipeline_pair_id(model_pipeline_pair))
        for model_pipeline_pair in model_pipeline_pairs
    ]


@pytest.fixture
def pipelines_fixture(request):
    model_id, pipeline_type = request.param[:2]
    options = request.param[2] if len(request.param) > 2 else {}
    model_name = model_id.split("/")[-1]
    model_path = get_ov_cache_converted_models_dir() / model_name
    model_id, _, hf_pipe, genai_pipe = read_asr_model((model_id, model_path), pipeline_type=pipeline_type, **options)
    return hf_pipe, genai_pipe, model_id, pipeline_type


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
def test_asr_config_constructor(model_descr, pipeline_type):
    model_id, path, _, _ = read_asr_model(model_descr, pipeline_type=pipeline_type)

    config_cls = get_config_cls(pipeline_type)
    config = config_cls(path / "generation_config.json")

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

    assert set(original_config["begin_suppress_tokens"]) == set(config.begin_suppress_tokens)

    assert set(original_config["suppress_tokens"]) == set(config.suppress_tokens)

    config = config_cls(
        suppress_tokens=[1, 2],
        begin_suppress_tokens=[3, 4],
        max_new_tokens=100,
        lang_to_id={"<|_ru|>": 42},
    )

    assert set(config.suppress_tokens) == set([1, 2])
    assert set(config.begin_suppress_tokens) == set([3, 4])
    assert config.max_new_tokens == 100
    assert config.lang_to_id["<|_ru|>"] == 42


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_asr_constructors(model_descr, sample_from_dataset, pipeline_type):
    model_id, path, hf_pipe, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

    expected = hf_pipe(sample_from_dataset)["text"]

    pipeline_cls = get_pipeline_cls(pipeline_type)

    genai_result = pipeline_cls(models_path=path, device="CPU", **{"ENABLE_MMAP": False}).generate(sample_from_dataset)

    assert genai_result.texts[0] == expected

    genai_result = pipeline_cls(path, "CPU", **{"ENABLE_MMAP": False}).generate(sample_from_dataset)
    assert genai_result.texts[0] == expected


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_max_new_tokens(model_descr, sample_from_dataset, pipeline_type):
    model_id, path, hf_pipe, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

    expected = hf_pipe(sample_from_dataset, max_new_tokens=10)

    genai_result = genai_pipe.generate(sample_from_dataset, max_new_tokens=10)

    compare_results(expected, genai_result)

    config = genai_pipe.get_generation_config()
    config.max_new_tokens = 10
    genai_result = genai_pipe.generate(sample_from_dataset, config)
    compare_results(expected, genai_result)


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("language", ["fr", "de"])
def test_language_mode(model_descr, language, pipeline_type):
    model_id, path, hf_pipe, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)
    sample = get_multilingual_audio_dataset(language)[0]

    config_cls = get_config_cls(pipeline_type)
    config = config_cls(max_new_tokens=30, language=language)

    expected = run_huggingface(hf_pipe, sample, config)

    genai_result = genai_pipe.generate(sample, max_new_tokens=30, language=f"<|{language}|>")

    compare_results(expected, genai_result)

    genai_config = genai_pipe.get_generation_config()
    genai_config.max_new_tokens = 30
    genai_config.language = f"<|{language}|>"
    genai_result = genai_pipe.generate(sample, genai_config)

    compare_results(expected, genai_result)


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["fr"], indirect=True)
def test_task_mode(model_descr, sample_from_multilingual_dataset, pipeline_type):
    model_id, path, hf_pipe, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

    config_cls = get_config_cls(pipeline_type)
    hf_config = config_cls(max_new_tokens=30, language="fr", task="translate")

    expected = run_huggingface(hf_pipe, sample_from_multilingual_dataset, hf_config)

    genai_result = genai_pipe.generate(
        sample_from_multilingual_dataset, max_new_tokens=30, language="<|fr|>", task="translate"
    )

    compare_results(expected, genai_result)

    genai_config = genai_pipe.get_generation_config()
    genai_config.max_new_tokens = 30
    genai_config.language = "<|fr|>"
    genai_config.task = "translate"
    genai_result = genai_pipe.generate(sample_from_multilingual_dataset, genai_config)

    compare_results(expected, genai_result)

    # seems to be equivalent to translate task
    hf_config = config_cls(max_new_tokens=30, language="en", task="transcribe")

    expected = run_huggingface(hf_pipe, sample_from_multilingual_dataset, hf_config)

    genai_result = genai_pipe.generate(
        sample_from_multilingual_dataset, max_new_tokens=30, language="<|en|>", task="transcribe"
    )

    compare_results(expected, genai_result)

    genai_config = genai_pipe.get_generation_config()
    genai_config.max_new_tokens = 30
    genai_config.language = "<|en|>"
    genai_config.task = "transcribe"
    genai_result = genai_pipe.generate(sample_from_multilingual_dataset, genai_config)

    compare_results(expected, genai_result)


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["fr", "de", "es"], indirect=True)
def test_language_autodetect(model_descr, sample_from_multilingual_dataset, pipeline_type):
    model_id, path, hf_pipe, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

    input_features = hf_pipe.feature_extractor(sample_from_multilingual_dataset)
    language_id = hf_pipe.model.detect_language(input_features["input_features"])[0]
    # ensure detected language is not english
    assert language_id != genai_pipe.get_generation_config().lang_to_id["<|en|>"]

    config_cls = get_config_cls(pipeline_type)
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=sample_from_multilingual_dataset,
        generation_config=config_cls(max_new_tokens=30),
        pipeline_type=pipeline_type,
    )


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_language_detection_en(model_descr, sample_from_dataset, pipeline_type):
    _, _, _, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

    result = genai_pipe.generate(sample_from_dataset)
    detected_language = result.languages[0] if hasattr(result, "languages") else result.language
    assert detected_language == "en"


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "sample_from_multilingual_dataset,language",
    [
        ("de", "de"),
        ("fr", "fr"),
        ("es", "es"),
    ],
    indirect=["sample_from_multilingual_dataset"],
)
def test_language_detection(model_descr, sample_from_multilingual_dataset, language, pipeline_type):
    _, _, _, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

    result = genai_pipe.generate(sample_from_multilingual_dataset)
    detected_language = result.languages[0] if hasattr(result, "languages") else result.language
    assert detected_language == language


@pytest.mark.parametrize(
    "pipelines_fixture",
    get_model_pipeline_pair_params(
        [
            ("openai/whisper-tiny", PipelineType.ASR),
            ("openai/whisper-tiny", PipelineType.WHISPER),
            (QWEN3_ASR_MODEL_ID, PipelineType.ASR),
        ]
    ),
    indirect=True,
)
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["fr"], indirect=True)
def test_forced_language(pipelines_fixture, sample_from_multilingual_dataset):
    _, genai_pipe, model_id, _ = pipelines_fixture

    config = {"language": "<|en|>"}
    if model_id == QWEN3_ASR_MODEL_ID:
        # tiny random model used for Qwen3-ASR testing. It was not trained to autodetect language.
        # Internal streamer suppresses language autodetection prefix, so if language is not forced all output is suppressed
        # also max_new_tokens have to be set as model cannot generate eos token
        config = {"language": "English", "max_new_tokens": 200}

    genai_result = genai_pipe.generate(sample_from_multilingual_dataset, **config)
    detected_language = genai_result.languages[0] if hasattr(genai_result, "languages") else genai_result.language
    expected_language = "en" if model_id != QWEN3_ASR_MODEL_ID else "English"
    assert detected_language == expected_language


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=1)], indirect=True)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_return_timestamps_short_form(model_descr, sample_from_dataset, pipeline_type):
    config_cls = get_config_cls(pipeline_type)
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=sample_from_dataset,
        generation_config=config_cls(return_timestamps=True),
        pipeline_type=pipeline_type,
    )


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 1}], indirect=True)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_return_timestamps_on_cut_sample(model_descr, sample_from_dataset, pipeline_type):
    sample_from_dataset = sample_from_dataset[: 30 * 16000]

    config_cls = get_config_cls(pipeline_type)
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=sample_from_dataset,
        generation_config=config_cls(return_timestamps=True),
        pipeline_type=pipeline_type,
    )


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=1)], indirect=True)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_return_timestamps_max_new_tokens_short_form(model_descr, sample_from_dataset, pipeline_type):
    config_cls = get_config_cls(pipeline_type)
    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=sample_from_dataset,
        generation_config=config_cls(return_timestamps=True, language="en", max_new_tokens=30),
        pipeline_type=pipeline_type,
    )


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize(
    "pipelines_fixture",
    get_model_pipeline_pair_params(
        [
            ("openai/whisper-tiny", PipelineType.ASR),
            ("openai/whisper-tiny", PipelineType.WHISPER),
            (QWEN3_ASR_MODEL_ID, PipelineType.ASR),
        ]
    ),
    indirect=True,
)
@pytest.mark.parametrize(
    "sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=10, long_form=True)], indirect=True
)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_longform_audio(pipelines_fixture, sample_from_dataset):
    hf_pipe, genai_pipe, model_id, pipeline_type = pipelines_fixture

    streamer_result = []

    config_cls = get_config_cls(pipeline_type)

    config = {}
    if model_id == QWEN3_ASR_MODEL_ID:
        # tiny random model used for Qwen3-ASR testing.
        # it cannot predict language so we have to force it to prevent streamer autodetection prefix suppression
        # also max_new_tokens have to be set as model cannot stop at eos token
        config = {"language": "English", "max_new_tokens": 200}

    genai_result = run_genai(
        genai_pipe,
        sample_from_dataset,
        config=config_cls(return_timestamps=True, **config),
        streamer=lambda x: streamer_result.append(x),
    )

    hf_result = run_huggingface(
        hf_pipe,
        sample_from_dataset,
        config=ov_genai.ASRGenerationConfig(return_timestamps=True, **config),
    )

    compare_results(hf_result, genai_result)

    assert "".join(streamer_result) == hf_result["text"]


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_shortform(model_descr, pipeline_type):
    if model_descr[0] == "openai/whisper-tiny":
        pytest.xfail("Accuracy issue. Ticket CVS-185132")
    samples = []
    ds = load_dataset_via_snapshot("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    for ds_row in ds:
        samples.append(ds_row["audio"]["array"])

    run_pipeline_with_ref(
        model_id=model_descr[0],
        tmp_path=model_descr[1],
        sample=samples,
        pipeline_type=pipeline_type,
    )


@pytest.fixture
def whisper_librispeech_10_openai_tiny_reference():
    json_path = pathlib.Path(__file__).parent / "data/whisper/librispeech_asr_dummy_10_openai_whisper_tiny_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        reference = json.load(f)
    return reference


def align_words_by_text(ref_words, test_words):
    """Align two word lists by matching their text content."""
    ref_texts = [w["word"].strip() for w in ref_words]
    test_texts = [w["word"].strip() for w in test_words]

    matcher = SequenceMatcher(None, ref_texts, test_texts)
    matches = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                matches.append((ref_words[i], test_words[j]))

    return matches


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_word_level_timestamps(model_descr, whisper_librispeech_10_openai_tiny_reference, pipeline_type):
    ds = load_dataset_via_snapshot("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").take(10)
    samples = [i["audio"]["array"] for i in ds]

    pipe = read_asr_model(model_descr, word_timestamps=True, pipeline_type=pipeline_type)[3]

    def openai_reference_to_words(reference):
        results = []
        for segment in reference["segments"]:
            for w in segment["words"]:
                results.append(
                    {
                        "word": w["word"],
                        "start_ts": w["start"],
                        "end_ts": w["end"],
                    }
                )
        return results

    matches = 0
    total_words = 0
    match_threshold = 0.02

    for i, sample in enumerate(samples):
        result = pipe.generate(
            sample,
            return_timestamps=True,
            word_timestamps=True,
        )
        words = result.words[0] if pipeline_type == PipelineType.ASR else result.words
        result_words = [
            {"word": get_word_text(w, pipeline_type), "start_ts": round(w.start_ts, 2), "end_ts": round(w.end_ts, 2)}
            for w in words
        ]

        reference = whisper_librispeech_10_openai_tiny_reference[i]
        reference_words = openai_reference_to_words(reference)

        aligned_words = align_words_by_text(reference_words, result_words)
        total_words += len(reference_words)

        for ref_word, test_word in aligned_words:
            start_diff = abs(ref_word["start_ts"] - test_word["start_ts"])
            end_diff = abs(ref_word["end_ts"] - test_word["end_ts"])

            if round(start_diff, 2) <= match_threshold and round(end_diff, 2) <= match_threshold:
                matches += 1

    assert total_words > 0
    accuracy = matches / total_words if total_words > 0 else 0
    assert accuracy > 0.95


@pytest.mark.parametrize("model_descr", get_whisper_models_list())
@pytest.mark.parametrize(
    "sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=2, long_form=True)], indirect=True
)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_longform_audio_with_word_level_timestamps(model_descr, sample_from_dataset, pipeline_type):
    genai_pipe = read_asr_model(model_descr, word_timestamps=True, pipeline_type=pipeline_type)[3]

    config_cls = get_config_cls(pipeline_type)
    config = config_cls(return_timestamps=True, word_timestamps=True)

    if model_descr[0] == "distil-whisper/distil-small.en":
        config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]

    genai_result = run_genai(
        genai_pipe,
        sample_from_dataset,
        config=config,
    )

    words = genai_result.words[0] if pipeline_type == PipelineType.ASR else genai_result.words
    assert len(words) > 0


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize(
    "sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=2, long_form=True)], indirect=True
)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_beam_search(model_descr, sample_from_dataset, pipeline_type):
    # use only 30 seconds of audio due to beam search results wrong with enabled timestamps
    # ticket: 167239
    sample_from_dataset = sample_from_dataset[: 30 * 16000]
    _, _, hf_pipe, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)
    config_cls = get_config_cls(pipeline_type)
    generation_config = config_cls(
        num_beams=2,
    )

    genai_result = run_genai(genai_pipe, sample_from_dataset, generation_config)
    hf_result = run_huggingface(hf_pipe, sample_from_dataset, generation_config)

    compare_results(hf_result, genai_result)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_initial_prompt_hotwords(model_descr, sample_from_dataset, pipeline_type):
    model_id, path, hf_pipe, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

    result = genai_pipe.generate(sample_from_dataset)
    assert "Kwilter" not in result.texts[0]
    assert "Quilter" in result.texts[0]

    # initial_prompt steers spelling of proper nouns
    result = genai_pipe.generate(sample_from_dataset, initial_prompt="Mr. Kwilter is known for his work.")
    assert "Kwilter" in result.texts[0]
    assert "Quilter" not in result.texts[0]

    result = genai_pipe.generate(sample_from_dataset, hotwords="Mr. Kwilter is known for his work.")
    assert "Kwilter" in result.texts[0]
    assert "Quilter" not in result.texts[0]


@pytest.mark.transformers_lower_v5(reason="CVS-185784")
@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_random_sampling(model_descr, sample_from_dataset, pipeline_type):
    _, _, hf_pipe, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

    config_cls = get_config_cls(pipeline_type)
    config = config_cls(do_sample=True, top_p=0.01)

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


@pytest.mark.parametrize(
    "pipelines_fixture",
    get_model_pipeline_pair_params(
        [
            ("openai/whisper-tiny", PipelineType.ASR, {"word_timestamps": True}),
            ("openai/whisper-tiny", PipelineType.WHISPER, {"word_timestamps": True}),
            (QWEN3_ASR_MODEL_ID, PipelineType.ASR),
        ]
    ),
    indirect=True,
)
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 173169")
def test_perf_metrics(pipelines_fixture, sample_from_dataset):
    _, genai_pipe, model_id, pipeline_type = pipelines_fixture

    if model_id == QWEN3_ASR_MODEL_ID:
        generate_kwargs = {"language": "English", "max_new_tokens": 200}
    else:
        generate_kwargs = {"return_timestamps": True, "word_timestamps": True}

    result = genai_pipe.generate(
        sample_from_dataset,
        **generate_kwargs,
    )

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
    if model_id == QWEN3_ASR_MODEL_ID:
        assert perf_metrics.get_tokenization_duration().mean > 0
    else:
        assert perf_metrics.get_tokenization_duration().mean == 0
    assert perf_metrics.get_detokenization_duration().mean > 0
    assert perf_metrics.get_features_extraction_duration().mean > 0
    if model_id != QWEN3_ASR_MODEL_ID:
        assert perf_metrics.get_word_level_timestamps_processing_duration().mean > 0
    assert perf_metrics.get_encode_inference_duration().mean > 0
    assert perf_metrics.get_decode_inference_duration().mean > 0
    assert perf_metrics.get_sampling_duration().mean > 0

    # assert that calculating statistics manually from the raw counters we get the same results as from PerfMetrics
    raw_metrics = get_raw_metrics(perf_metrics, pipeline_type)

    raw_dur = np.array(raw_metrics.features_extraction_durations) / 1000
    mean_dur, std_dur = perf_metrics.get_features_extraction_duration()
    assert np.allclose(mean_dur, np.mean(raw_dur))
    assert np.allclose(std_dur, np.std(raw_dur))

    if model_id != QWEN3_ASR_MODEL_ID:
        assert len(raw_metrics.word_level_timestamps_processing_durations) == 1
        word_ts_raw_dur = np.array(raw_metrics.word_level_timestamps_processing_durations) / 1000
        mean_dur, std_dur = perf_metrics.get_word_level_timestamps_processing_duration()
        assert np.allclose(mean_dur, np.mean(word_ts_raw_dur))
        assert np.allclose(std_dur, np.std(word_ts_raw_dur))

    enc_raw_dur = np.array(raw_metrics.encode_inference_durations) / 1000
    mean_dur, std_dur = perf_metrics.get_encode_inference_duration()
    assert len(enc_raw_dur) > 0
    assert np.allclose(mean_dur, np.mean(enc_raw_dur))
    assert np.allclose(std_dur, np.std(enc_raw_dur))

    dec_raw_dur = np.array(raw_metrics.decode_inference_durations) / 1000
    mean_dur, std_dur = perf_metrics.get_decode_inference_duration()
    assert len(dec_raw_dur) > 0
    assert np.allclose(mean_dur, np.mean(dec_raw_dur))
    assert np.allclose(std_dur, np.std(dec_raw_dur))

    smp_raw_dur = np.array(perf_metrics.raw_metrics.sampling_durations) / 1000
    mean_dur, std_dur = perf_metrics.get_sampling_duration()
    assert len(smp_raw_dur) > 0
    assert np.allclose(mean_dur, np.mean(smp_raw_dur))
    assert np.allclose(std_dur, np.std(smp_raw_dur))


@pytest.fixture(params=["Streamer", "streamer_callback", "streamer_bool_callback"])
def streamer_for_test(request):
    class ResultHandler:
        def __init__(self, container: list[int] | list[str]):
            self.container: list[int] | list[str] = container

        def decode(self, tokenizer: ov_genai.Tokenizer) -> str:
            if type(self.container[0]) == int:
                return tokenizer.decode(typing.cast(list[int], self.container))
            return "".join(typing.cast(list[str], self.container))

        def reset(self) -> None:
            self.container.clear()

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

    if request.param == "Streamer":
        streamer = Streamer()
        return streamer, ResultHandler(streamer.tokens)

    if request.param == "streamer_callback":
        texts = []

        def streamer_callback(subword):
            texts.append(subword)
            return ov_genai.StreamingStatus.RUNNING

        return streamer_callback, ResultHandler(texts)

    if request.param == "streamer_bool_callback":
        texts = []

        def streamer_bool_callback(subword):
            texts.append(subword)
            return False

        return streamer_bool_callback, ResultHandler(texts)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
@pytest.mark.xfail(sys.platform == "darwin", reason="Ticket - 182134", raises=AssertionError)
def test_streamers(model_descr, sample_from_dataset, streamer_for_test, pipeline_type):
    _, _, _, genai_pipe = read_asr_model(model_descr, pipeline_type=pipeline_type)

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
