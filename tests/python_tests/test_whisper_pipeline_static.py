# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils.network import retry_request
from utils.atomic_download import AtomicDownloadManager
from test_whisper_pipeline import get_whisper_models_list, sample_from_dataset, get_fixture_params_for_n_whisper_dataset_samples
from transformers import WhisperProcessor, AutoTokenizer
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from huggingface_hub import snapshot_download
import openvino_genai as ov_genai
import openvino_tokenizers
import openvino
import pytest
import pathlib

# This test suite is designed specifically to validate the functionality 
# and robustness of the WhisperStaticPipeline on NPUW:CPU.
config = {"NPU_USE_NPUW" : "YES",
          "NPUW_DEVICES" : "CPU",
          "NPUW_ONLINE_PIPELINE" : "NONE",
          "STATIC_PIPELINE": True}

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
            print(f'expected: {expected.texts[i]}\n')
            print(f'actual_out: {actual_out.texts[i]}')
        assert expected.texts[i] == actual_out.texts[i]


def compare_word_timestamps_results_with_assert(expected, actual_out, ts_tolerance=0.07):
    assert len(expected.words) == len(actual_out.words)

    for exp_word, act_word in zip(expected.words, actual_out.words):
        assert exp_word.word == act_word.word
        assert exp_word.start_ts - act_word.start_ts == pytest.approx(0.0, abs=ts_tolerance)
        assert exp_word.end_ts - act_word.end_ts == pytest.approx(0.0, abs=ts_tolerance)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
def test_static_whisper_generation_compare_stateless(model_descr, sample_from_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_dataset
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=2, language="fr"),
                                                 *get_fixture_params_for_n_whisper_dataset_samples(n=2, language="de"),
                                                 *get_fixture_params_for_n_whisper_dataset_samples(n=2, language="es")], indirect=True)
def test_static_whisper_autodetect(model_descr, sample_from_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_dataset
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(language='de', n=3)], indirect=True)
def test_static_whisper_language_de(model_descr, sample_from_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_dataset, max_new_tokens=30, language="<|de|>"
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(language='fr', n=3)], indirect=True)
def test_static_whisper_language_fr(model_descr, sample_from_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_dataset, max_new_tokens=30, language="<|fr|>"
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(language='ru', n=3)], indirect=True)
def test_static_whisper_language_ru(model_descr, sample_from_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_dataset, max_new_tokens=30, language="<|ru|>"
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0, "long_form": True}], indirect=True)
def test_static_whisper_generation_long(model_descr, sample_from_dataset):
    model_id, stateful_model_path = load_and_save_whisper_model(model_descr, stateful=True)
    model_id, stateless_model_path = load_and_save_whisper_model(model_descr, stateful=False)

    expected, actual_out = get_results_npu_stateful_stateless(
        stateful_model_path, stateless_model_path, sample_from_dataset
    )

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0}], indirect=True)
def test_static_whisper_stateful_generation_compare_with_cpu(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_dataset)

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(n=2, language="fr"),
                                                 *get_fixture_params_for_n_whisper_dataset_samples(n=2, language="de"),
                                                 *get_fixture_params_for_n_whisper_dataset_samples(n=2, language="es")], indirect=True)
def test_static_whisper_stateful_autodetect(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_dataset)

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(language='de', n=3)], indirect=True)
def test_static_whisper_stateful_language_de(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_dataset, max_new_tokens=30, language="<|de|>")

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(language='fr', n=3)], indirect=True)
def test_static_whisper_stateful_language_fr(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_dataset, max_new_tokens=30, language="<|fr|>")

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [*get_fixture_params_for_n_whisper_dataset_samples(language='ru', n=3)], indirect=True)
def test_static_whisper_stateful_language_ru(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_dataset, max_new_tokens=30, language="<|ru|>")

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0, "long_form": True}], indirect=True)
def test_static_whisper_stateful_generation_long(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)

    expected, actual_out = get_results_cpu_npu(model_path, sample_from_dataset)

    compare_results_with_assert(expected, actual_out)


@pytest.mark.parametrize("model_descr", get_whisper_models_list(tiny_only=True))
@pytest.mark.parametrize("sample_from_dataset", [{"language": "en", "sample_id": 0, "long_form": False}], indirect=True)
def test_static_whisper_stateful_word_timestamps(model_descr, sample_from_dataset):
    model_id, model_path = load_and_save_whisper_model(model_descr, stateful=True)
    sample_from_dataset = sample_from_dataset[: 16000 * 30]

    expected, actual_out = get_word_timestamps_results_cpu_npu(model_path, sample_from_dataset)

    compare_results_with_assert(expected, actual_out)
    compare_word_timestamps_results_with_assert(expected, actual_out)
