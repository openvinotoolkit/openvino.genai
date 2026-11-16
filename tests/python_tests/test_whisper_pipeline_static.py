# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils.constants import NPUW_CPU_PROPERTIES
from utils.network import retry_request
from utils.atomic_download import AtomicDownloadManager
from test_asr_pipeline import (
    get_whisper_models_list,
    sample_from_multilingual_dataset,
    sample_from_dataset,
    get_fixture_params_for_n_whisper_dataset_samples,
)
from transformers import AutoTokenizer, WhisperProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from huggingface_hub import snapshot_download
import openvino_genai as ov_genai
import openvino_tokenizers
import openvino
import pytest
import pathlib
import json
import io
import numpy as np
import requests
import soundfile as sf

# This test suite is designed specifically to validate the functionality
# and robustness of the WhisperStaticPipeline on NPUW:CPU.
config = {**NPUW_CPU_PROPERTIES, "STATIC_PIPELINE": True}
dynamic_config = {**NPUW_CPU_PROPERTIES}


def load_and_save_whisper_model(params, stateful=False, **tokenizer_kwargs):
    model_id, path = params

    model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
    processor = retry_request(lambda: WhisperProcessor.from_pretrained(model_cached, trust_remote_code=True))
    if not stateful:
        path = pathlib.Path(f"{path}_with_past")

    manager = AtomicDownloadManager(path)

    if not manager.is_complete() and not (path / "openvino_encoder_model.xml").exists():

        def convert_to_temp(temp_path: pathlib.Path) -> None:
            tokenizer = AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True)
            ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
                tokenizer,
                with_detokenizer=True,
                clean_up_tokenization_spaces=False,
                **tokenizer_kwargs,
            )

            openvino.save_model(ov_tokenizer, temp_path / "openvino_tokenizer.xml")
            openvino.save_model(ov_detokenizer, temp_path / "openvino_detokenizer.xml")

            tokenizer.save_pretrained(temp_path)

            opt_model = retry_request(
                lambda: OVModelForSpeechSeq2Seq.from_pretrained(
                    model_cached,
                    export=True,
                    trust_remote_code=True,
                    stateful=stateful,
                    compile=False,
                    device="CPU",
                    load_in_8bit=False,
                )
            )
            opt_model.generation_config.save_pretrained(temp_path)
            opt_model.config.save_pretrained(temp_path)
            opt_model.save_pretrained(temp_path)
            processor.save_pretrained(temp_path)

        manager.execute(convert_to_temp)

    return model_id, path


def get_results_cpu_npu(model_path, audio_sample, **config_kwargs):
    cpu_pipe = ov_genai.WhisperPipeline(model_path, "CPU")
    expected = cpu_pipe.generate(audio_sample, **config_kwargs)

    npu_pipe = ov_genai.WhisperPipeline(model_path, "NPU", **config)
    actual_out = npu_pipe.generate(audio_sample, **config_kwargs)

    return expected, actual_out


def get_results_cpu_npu_dynamic(model_path, audio_sample, **config_kwargs):
    cpu_pipe = ov_genai.WhisperPipeline(model_path, "CPU")
    expected = cpu_pipe.generate(audio_sample, **config_kwargs)

    npu_pipe = ov_genai.WhisperPipeline(model_path, "NPU", **dynamic_config)
    actual_out = npu_pipe.generate(audio_sample, **config_kwargs)

    return expected, actual_out


def get_word_timestamps_results_cpu_npu(model_path, audio_sample, **config_kwargs):
    cpu_pipe = ov_genai.WhisperPipeline(model_path, "CPU", word_timestamps=True)
    expected = cpu_pipe.generate(audio_sample, **config_kwargs)

    npu_pipe = ov_genai.WhisperPipeline(model_path, "NPU", word_timestamps=True, **config)
    actual_out = npu_pipe.generate(audio_sample, **config_kwargs)

    return expected, actual_out


def get_results_npu_stateful_stateless(stateful_model_path, stateless_model_path, audio_sample, **config_kwargs):
    stateful_pipe = ov_genai.WhisperPipeline(stateful_model_path, "NPU", **config)
    stateful_out = stateful_pipe.generate(audio_sample, **config_kwargs)

    stateless_pipe = ov_genai.WhisperPipeline(stateless_model_path, "NPU", **config)
    stateless_out = stateless_pipe.generate(audio_sample, **config_kwargs)

    return stateful_out, stateless_out


def compare_results_with_assert(expected, actual_out):
    assert len(expected.texts) == len(actual_out.texts)

    for i in range(0, len(expected.texts)):
        if expected.texts[i] != actual_out.texts[i]:
            print(f"expected: {expected.texts[i]}\n")
            print(f"actual_out: {actual_out.texts[i]}")
        assert expected.texts[i] == actual_out.texts[i]


def compare_word_timestamps_results_with_assert(expected, actual_out, ts_tolerance=0.08):
    assert len(expected.words) == len(actual_out.words)

    for exp_word, act_word in zip(expected.words, actual_out.words):
        assert exp_word.word == act_word.word
        assert exp_word.start_ts - act_word.start_ts == pytest.approx(0.0, abs=ts_tolerance)
        assert exp_word.end_ts - act_word.end_ts == pytest.approx(0.0, abs=ts_tolerance)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_static_whisper_generation_compare_stateless(model_descr, sample_from_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_dataset
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["fr", "de", "es"], indirect=True)
def test_static_whisper_autodetect(model_descr, sample_from_multilingual_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_multilingual_dataset
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["de"], indirect=True)
# Both the plain code and the wrapped token must reach the static pipeline's
# configured-language path in pipeline_static.cpp (not just config validation).
@pytest.mark.parametrize("language", ["de", "<|de|>"])
def test_static_whisper_language_de(model_descr, sample_from_multilingual_dataset, language):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path,
        stateless_model_path,
        sample_from_multilingual_dataset,
        max_new_tokens=30,
        language=language,
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["fr"], indirect=True)
def test_static_whisper_language_fr(model_descr, sample_from_multilingual_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path,
        stateless_model_path,
        sample_from_multilingual_dataset,
        max_new_tokens=30,
        language="<|fr|>",
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0, "long_form": True}], indirect=True)
def test_static_whisper_generation_long(model_descr, sample_from_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_dataset
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_static_whisper_stateful_generation_compare_with_cpu(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_dataset)

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["fr", "de", "es"], indirect=True)
def test_static_whisper_stateful_autodetect(model_descr, sample_from_multilingual_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_multilingual_dataset)

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["de"], indirect=True)
def test_static_whisper_stateful_language_de(model_descr, sample_from_multilingual_dataset):
    if model_descr[0] == "openai/whisper-tiny":
        pytest.xfail("Accuracy issue. Ticket CVS-185132")
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(
        model_path, sample_from_multilingual_dataset, max_new_tokens=30, language="<|de|>"
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_multilingual_dataset", ["fr"], indirect=True)
def test_static_whisper_stateful_language_fr(model_descr, sample_from_multilingual_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(
        model_path, sample_from_multilingual_dataset, max_new_tokens=30, language="<|fr|>"
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0, "long_form": True}], indirect=True)
def test_static_whisper_stateful_generation_long(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_dataset)

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_static_whisper_stateful_word_timestamps(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)
    sample_from_dataset = sample_from_dataset[: 16000 * 30]

    expected, actual_out = get_word_timestamps_results_cpu_npu(model_path, sample_from_dataset)

    compare_results_with_assert(expected, actual_out)
    compare_word_timestamps_results_with_assert(expected, actual_out)


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
def test_language_detection(model_descr, sample_from_multilingual_dataset, language):
    _, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_word_timestamps_results_cpu_npu(model_path, sample_from_multilingual_dataset)
    compare_results_with_assert(expected, actual_out)
    assert expected.language == actual_out.language == language


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_language_detection_en(model_descr, sample_from_dataset):
    _, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_word_timestamps_results_cpu_npu(model_path, sample_from_dataset)
    compare_results_with_assert(expected, actual_out)
    assert expected.language == actual_out.language == "en"


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_dynamic_whisper_stateful_generation_compare_with_cpu(model_descr, sample_from_dataset):
    """
    Regression test for Issue #4222:
    Validates default dynamic WhisperPipelineStatefulImpl on NPU with NPUW CPU fallback.
    """
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu_dynamic(model_path, sample_from_dataset)

    compare_results_with_assert(expected, actual_out)


@pytest.fixture(scope="module")
def qwen3_asr_npu_pipelines():
    from utils.asr_utils.qwen3_asr import qwen3_asr_model_path

    model_path = pathlib.Path(qwen3_asr_model_path())
    model_config = json.loads((model_path / "config.json").read_text())
    preprocessor = json.loads((model_path / "preprocessor_config.json").read_text())
    chunk_frames = 2 * model_config["thinker_config"]["audio_config"].get("n_window", 100)
    cpu = ov_genai.ASRPipeline(model_path, "CPU")
    npu = ov_genai.ASRPipeline(model_path, "NPU", **NPUW_CPU_PROPERTIES)
    return cpu, npu, chunk_frames, preprocessor["hop_length"], preprocessor["sampling_rate"]


@pytest.fixture(scope="module")
def qwen3_asr_reference_speech():
    url = "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav"

    def download():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content

    audio, sampling_rate = sf.read(io.BytesIO(retry_request(download)), dtype="float32")
    assert sampling_rate == 16000 and audio.ndim == 1 and len(audio) > 0
    return audio


@pytest.mark.parametrize("encoder_chunks", [1.0, 1.5, 2.0])
def test_qwen3_asr_npu_full_partial_and_repeated_chunks(
    qwen3_asr_npu_pipelines, qwen3_asr_reference_speech, encoder_chunks
):
    cpu, npu, chunk_frames, hop_length, _ = qwen3_asr_npu_pipelines
    samples = int(chunk_frames * hop_length * encoder_chunks)
    repetitions = (samples + len(qwen3_asr_reference_speech) - 1) // len(qwen3_asr_reference_speech)
    audio = np.tile(qwen3_asr_reference_speech, repetitions)[:samples]
    generation = {"language": "English", "max_new_tokens": 8, "do_sample": False}

    expected = cpu.generate(audio.tolist(), **generation)
    first = npu.generate(audio.tolist(), **generation)
    repeated = npu.generate(audio.tolist(), **generation)

    assert first.texts == repeated.texts == expected.texts
    assert first.languages == repeated.languages == expected.languages


def test_qwen3_asr_npu_rejects_oversized_output_and_beam_search(
    qwen3_asr_npu_pipelines, qwen3_asr_reference_speech
):
    _, npu, _, _, _ = qwen3_asr_npu_pipelines
    audio = qwen3_asr_reference_speech[:3200].tolist()
    with pytest.raises(RuntimeError, match=r"requested .* new tokens.*KV-cache can accommodate only"):
        npu.generate(audio, language="English", max_new_tokens=1_000_000, ignore_eos=True)
    with pytest.raises(RuntimeError, match="does not support beam search"):
        npu.generate(audio, language="English", max_new_tokens=8, num_beams=2)


def test_qwen3_asr_npu_capacity_includes_chunk_boundary_search(qwen3_asr_npu_pipelines):
    _, npu, _, _, sampling_rate = qwen3_asr_npu_pipelines
    audio = np.full(66 * sampling_rate, 0.01, dtype=np.float32)
    audio[62 * sampling_rate : 63 * sampling_rate] = 0.0
    generation = {"language": "English", "max_new_tokens": 8}

    first = npu.generate(audio.tolist(), **generation)
    repeated = npu.generate(audio.tolist(), **generation)

    assert first.texts == repeated.texts
    assert len(first.perf_metrics.asr_raw_metrics.encode_inference_durations) == 2


def test_qwen3_asr_npu_single_token_reset_matches_fresh_decoder():
    from utils.asr_utils.qwen3_asr import qwen3_asr_model_path

    model_path = pathlib.Path(qwen3_asr_model_path())
    model_config = json.loads((model_path / "config.json").read_text())
    audio_token = model_config["thinker_config"]["audio_token_id"]
    hidden_size = model_config["thinker_config"]["audio_config"]["output_dim"]
    core = openvino.Core()
    model = core.read_model(model_path / "openvino_decoder_model.xml")
    hidden_shape = model.input("encoder_hidden_states").partial_shape
    hidden_shape[0] = 1
    hidden_shape[1] = -1
    model.reshape({"encoder_hidden_states": hidden_shape})
    compiled = core.compile_model(
        model, "NPU", {
            **NPUW_CPU_PROPERTIES,
            "NPUW_LLM": "YES",
            "NPUW_QWEN3_ASR": "YES",
            "NPUW_QWEN3_ASR_MAX_ENCODER_LEN": 16,
            "NPUW_LLM_MAX_PROMPT_LEN": 64,
            "NPUW_LLM_MIN_RESPONSE_LEN": 64,
            "NPUW_LLM_BATCH_DIM": 0,
            "NPUW_LLM_SEQ_LEN_DIM": 2,
            "NPUW_LLM_PREFILL_HINT": "STATIC",
            "NPUW_LLM_PREFILL_CHUNK_SIZE": 0,
        },
    )

    def stored_tokens(request):
        states = [state for state in request.query_state() if state.name == "npuw_stored_tokens_state"]
        assert len(states) == 1
        return int(np.asarray(states[0].state.data).reshape(-1)[0])

    def infer(request, ids, value):
        request.set_tensor("input_ids", openvino.Tensor(np.array([ids], dtype=np.int64)))
        request.set_tensor("beam_idx", openvino.Tensor(np.array([0], dtype=np.int32)))
        request.set_tensor(
            "encoder_hidden_states",
            openvino.Tensor(np.full((1, 1, hidden_size), value, dtype=np.float32)),
        )
        request.infer()
        return np.array(request.get_tensor("logits").data, copy=True)

    fresh = compiled.create_infer_request()
    expected = infer(fresh, [audio_token], 0.125)
    assert stored_tokens(fresh) == 1

    reused = compiled.create_infer_request()
    infer(reused, [audio_token, 1], 1.0)
    assert stored_tokens(reused) == 2
    infer(reused, [1], 1.0)
    assert stored_tokens(reused) == 3
    infer(fresh, [1], 0.125)
    assert stored_tokens(fresh) == 2
    assert stored_tokens(reused) == 3
    reused.reset_state()
    assert stored_tokens(reused) == 0
    actual = infer(reused, [audio_token], 0.125)
    assert stored_tokens(reused) == 1
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_qwen3_asr_npu_blob_export_import_preserves_decoder_path(tmp_path):
    from utils.asr_utils.qwen3_asr import qwen3_asr_model_path

    model_path = pathlib.Path(qwen3_asr_model_path())
    blob = tmp_path / "qwen3_asr_decoder.blob"
    audio = np.zeros(3200, dtype=np.float32).tolist()
    generation = {"language": "English", "max_new_tokens": 8}
    exported = ov_genai.ASRPipeline(
        model_path, "NPU", **NPUW_CPU_PROPERTIES,
        MAX_PROMPT_LEN=128, MIN_RESPONSE_LEN=64, BLOB_PATH=str(blob), EXPORT_BLOB=True,
    )
    expected = exported.generate(audio, **generation)
    assert blob.is_file()
    del exported

    imported = ov_genai.ASRPipeline(
        model_path, "NPU", **NPUW_CPU_PROPERTIES,
        BLOB_PATH=str(blob), WEIGHTS_PATH=str(model_path / "openvino_decoder_model.bin"),
    )
    actual = imported.generate(audio, **generation)
    assert actual.texts == expected.texts
    assert actual.languages == expected.languages
