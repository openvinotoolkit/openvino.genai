# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import subprocess  # nosec B404
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

import openvino as ov
import openvino_genai as ov_genai
from optimum.intel import OVModelForTextToSpeechSeq2Seq

from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir
from utils.network import retry_request

logger = logging.getLogger(__name__)

QWEN3_TTS_BASE_TINY_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-tts"
QWEN3_TTS_CUSTOMVOICE_TINY_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-tts-customvoice"
QWEN3_TTS_VOICEDESIGN_TINY_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-tts-voicedesign"

SAMPLE_RATE = 24000
LANGUAGES = ["English", "Chinese", "Russian"]
LANGUAGE_TEST_TEXTS = {
    "English": {
        "prompt": "Okay, let's run some tests! Is everything okay?",
        "reference_text": "This is a dummy-generated waveform for parity testing",
        "instruct": "Speak with a calm, friendly, and reassuring tone.",
        "customvoice_speaker": "serena",
    },
    "Chinese": {
        "prompt": "好了，我们来做些测试吧！一切都正常吗？",
        "reference_text": "这是一个用于奇偶校验测试的模拟生成波形。",
        "instruct": "请用平静、亲切且令人安心的语气说话。",
        "customvoice_speaker": "uncle_fu",
    },
    "Russian": {
        "prompt": "Хорошо, давайте проведем несколько тестов! Всё в порядке?",
        "reference_text": "Это сгенерированный тестовый сигнал для проверки четности.",
        "instruct": "Говорите спокойным, дружелюбным и обнадеживающим тоном.",
        "customvoice_speaker": "vivian",
    },
}
GENERATION_KWARGS = {
    "do_sample": False,
    "subtalker_dosample": False,
    "max_new_tokens": 32,
}


def _generation_kwargs(non_streaming_mode: bool) -> dict:
    return {**GENERATION_KWARGS, "non_streaming_mode": non_streaming_mode}


def _prepare_qwen3_tts_ov_model(
    model_id: str,
    cache_name: str,
) -> Path:
    models_dir = get_ov_cache_converted_models_dir()
    model_dir = models_dir / cache_name
    manager = AtomicDownloadManager(model_dir)

    if manager.is_complete() or (model_dir / "openvino_talker_model.xml").exists():
        return model_dir

    def convert_to_temp(temp_dir: Path) -> None:
        command = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            model_id,
            "--trust-remote-code",
            str(temp_dir),
        ]
        logger.info("Qwen3 TTS export command: %s", " ".join(command))
        retry_request(
            lambda: subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        )

    try:
        manager.execute(convert_to_temp)
    except FileNotFoundError:
        pytest.fail("optimum-cli is not available in PATH")
    except subprocess.CalledProcessError as error:
        pytest.fail(f"Failed to export {model_id}: returncode={error.returncode}, stderr={error.stderr}")

    return model_dir


def _make_ref_audio() -> np.ndarray:
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / np.float32(SAMPLE_RATE)
    audio = 0.18 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t + 0.3)
    return np.ascontiguousarray(audio.astype(np.float32))


def _to_waveform_array(output) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        assert len(output) == 1, f"Expected single-sample output, got batch size {len(output)}"
        output = output[0]

    if hasattr(output, "detach"):
        array = output.detach().cpu().numpy()
    else:
        array = np.asarray(output)

    return np.asarray(array, dtype=np.float32).reshape(-1)


def _assert_waveform_equal(expected: np.ndarray, actual: np.ndarray, context: str) -> None:
    assert expected.shape == actual.shape, (
        f"Shape mismatch for {context}: expected={expected.shape}, actual={actual.shape}"
    )

    if not np.array_equal(expected, actual):
        diff_idx = np.flatnonzero(expected != actual)
        first = int(diff_idx[0]) if diff_idx.size else -1
        max_diff = float(np.max(np.abs(expected - actual)))
        mean_diff = float(np.mean(np.abs(expected - actual)))
        pytest.fail(
            f"Waveform mismatch for {context}: first_diff_index={first}, "
            f"expected={expected[first] if first >= 0 else 'n/a'}, "
            f"actual={actual[first] if first >= 0 else 'n/a'}, "
            f"max_abs_diff={max_diff:.7f}, mean_abs_diff={mean_diff:.7f}"
        )


@pytest.fixture(scope="module")
def tiny_qwen3_tts_base_ov_path() -> Path:
    return _prepare_qwen3_tts_ov_model(
        QWEN3_TTS_BASE_TINY_MODEL_ID,
        QWEN3_TTS_BASE_TINY_MODEL_ID.replace("/", "_"),
    )


@pytest.fixture(scope="module")
def tiny_qwen3_tts_customvoice_ov_path() -> Path:
    return _prepare_qwen3_tts_ov_model(
        QWEN3_TTS_CUSTOMVOICE_TINY_MODEL_ID,
        "optimum-intel-internal-testing_tiny-random-qwen3-tts-customvoice",
    )


@pytest.fixture(scope="module")
def tiny_qwen3_tts_voicedesign_ov_path() -> Path:
    return _prepare_qwen3_tts_ov_model(
        QWEN3_TTS_VOICEDESIGN_TINY_MODEL_ID,
        "optimum-intel-internal-testing_tiny-random-qwen3-tts-voicedesign",
    )


@pytest.fixture(scope="module")
def optimum_qwen3_tts_base_model(tiny_qwen3_tts_base_ov_path: Path):
    try:
        return retry_request(
            lambda: OVModelForTextToSpeechSeq2Seq.from_pretrained(
                str(tiny_qwen3_tts_base_ov_path),
                trust_remote_code=True,
                local_files_only=True,
                compile=False,
                device="CPU",
            )
        )
    except Exception as error:
        pytest.fail(f"Failed to load optimum Qwen3-TTS model: {error}")


@pytest.fixture
def genai_qwen3_tts_base_pipe(tiny_qwen3_tts_base_ov_path: Path):
    return ov_genai.Text2SpeechPipeline(str(tiny_qwen3_tts_base_ov_path), "CPU")


@pytest.fixture(scope="module")
def optimum_qwen3_tts_customvoice_model(tiny_qwen3_tts_customvoice_ov_path: Path):
    try:
        return retry_request(
            lambda: OVModelForTextToSpeechSeq2Seq.from_pretrained(
                str(tiny_qwen3_tts_customvoice_ov_path),
                trust_remote_code=True,
                local_files_only=True,
                compile=False,
                device="CPU",
            )
        )
    except Exception as error:
        pytest.fail(f"Failed to load optimum Qwen3-TTS CustomVoice model: {error}")


@pytest.fixture
def genai_qwen3_tts_customvoice_pipe(tiny_qwen3_tts_customvoice_ov_path: Path):
    return ov_genai.Text2SpeechPipeline(str(tiny_qwen3_tts_customvoice_ov_path), "CPU")


@pytest.fixture(scope="module")
def optimum_qwen3_tts_voicedesign_model(tiny_qwen3_tts_voicedesign_ov_path: Path):
    try:
        return retry_request(
            lambda: OVModelForTextToSpeechSeq2Seq.from_pretrained(
                str(tiny_qwen3_tts_voicedesign_ov_path),
                trust_remote_code=True,
                local_files_only=True,
                compile=False,
                device="CPU",
            )
        )
    except Exception as error:
        pytest.fail(f"Failed to load optimum Qwen3-TTS VoiceDesign model: {error}")


@pytest.fixture
def genai_qwen3_tts_voicedesign_pipe(tiny_qwen3_tts_voicedesign_ov_path: Path):
    return ov_genai.Text2SpeechPipeline(str(tiny_qwen3_tts_voicedesign_ov_path), "CPU")


def _run_optimum_base_generate(
    model: OVModelForTextToSpeechSeq2Seq,
    prompt: str,
    language: str,
    ref_audio: np.ndarray,
    ref_text: Optional[str],
    x_vector_only_mode: bool,
    non_streaming_mode: bool,
) -> tuple[np.ndarray, int]:
    preprocess_kwargs = {
        "text": prompt,
        "language": language,
        "ref_audio": (ref_audio, SAMPLE_RATE),
        "x_vector_only_mode": x_vector_only_mode,
    }
    if ref_text is not None:
        preprocess_kwargs["ref_text"] = ref_text

    inputs = model.preprocess_input(**preprocess_kwargs)
    output = model.generate(**inputs, **_generation_kwargs(non_streaming_mode))
    sample_rate = int(getattr(model, "sampling_rate", SAMPLE_RATE))
    return _to_waveform_array(output), sample_rate


def _run_genai_base_generate(
    pipe: ov_genai.Text2SpeechPipeline,
    prompt: str,
    language: str,
    ref_audio: np.ndarray,
    **generation_kwargs,
) -> tuple[np.ndarray, int]:
    result = pipe.generate(
        prompt,
        language=language,
        ref_audio=ov.Tensor(ref_audio),
        **generation_kwargs,
    )
    speech = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)
    return speech, int(result.output_sample_rate)


@pytest.mark.speech_generation
def test_qwen3_tts_base_icl_model_properties_change_genai_output(
    tiny_qwen3_tts_base_ov_path: Path,
):
    stack_ov_config = {"KV_CACHE_PRECISION": "f32"}
    component_properties = {
        "MODEL_PROPERTIES": {
            "talker_model": stack_ov_config,
            "code_predictor_model": {**stack_ov_config, "INFERENCE_PRECISION_HINT": "f32"},
        }
    }

    genai_pipe_with_props = ov_genai.Text2SpeechPipeline(
        str(tiny_qwen3_tts_base_ov_path),
        "CPU",
        **component_properties,
    )
    genai_pipe_default = ov_genai.Text2SpeechPipeline(str(tiny_qwen3_tts_base_ov_path), "CPU")

    language = "English"
    prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
    ref_text = LANGUAGE_TEST_TEXTS[language]["reference_text"]
    ref_audio = _make_ref_audio()

    with_props_result = genai_pipe_with_props.generate(
        prompt,
        language=language,
        ref_audio=ov.Tensor(ref_audio),
        ref_text=ref_text,
        **_generation_kwargs(non_streaming_mode=True),
    )
    with_props_speech = np.array(with_props_result.speeches[0].data, dtype=np.float32).reshape(-1)

    default_result = genai_pipe_default.generate(
        prompt,
        language=language,
        ref_audio=ov.Tensor(ref_audio),
        ref_text=ref_text,
        **_generation_kwargs(non_streaming_mode=True),
    )
    default_speech = np.array(default_result.speeches[0].data, dtype=np.float32).reshape(-1)

    assert with_props_result.output_sample_rate == SAMPLE_RATE, (
        f"GenAI output sample rate mismatch for MODEL_PROPERTIES test (with props): "
        f"expected={SAMPLE_RATE}, actual={with_props_result.output_sample_rate}"
    )
    assert default_result.output_sample_rate == SAMPLE_RATE, (
        f"GenAI output sample rate mismatch for MODEL_PROPERTIES test (default): "
        f"expected={SAMPLE_RATE}, actual={default_result.output_sample_rate}"
    )

    if np.array_equal(with_props_speech, default_speech):
        pytest.fail(
            "MODEL_PROPERTIES sanity check failed: waveform with component properties "
            "matches waveform from default pipeline for the same English ICL input."
        )


@pytest.mark.speech_generation
def test_qwen3_tts_base_sampling_default_seeded_is_deterministic(
    genai_qwen3_tts_base_pipe,
):
    language = "English"
    prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
    ref_audio = _make_ref_audio()

    cases = [
        ("do_sample=false", {"max_new_tokens": 128, "rng_seed": 2025}),
        ("do_sample=true", {"max_new_tokens": 128, "rng_seed": 2026, "do_sample": True}),
    ]

    for case_name, generation_kwargs in cases:
        first_speech, first_sr = _run_genai_base_generate(
            genai_qwen3_tts_base_pipe,
            prompt,
            language,
            ref_audio,
            **generation_kwargs,
        )
        second_speech, second_sr = _run_genai_base_generate(
            genai_qwen3_tts_base_pipe,
            prompt,
            language,
            ref_audio,
            **generation_kwargs,
        )

        assert first_sr == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, got {first_sr}"
        assert second_sr == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, got {second_sr}"
        _assert_waveform_equal(first_speech, second_speech, f"seeded reproducibility ({case_name})")


@pytest.mark.speech_generation
def test_qwen3_tts_base_sampling_subtalker_knobs_change_output(
    genai_qwen3_tts_base_pipe,
):
    language = "English"
    prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
    ref_audio = _make_ref_audio()

    baseline_kwargs = {
        "max_new_tokens": 128,
        "rng_seed": 2027,
    }
    baseline_speech, baseline_sr = _run_genai_base_generate(
        genai_qwen3_tts_base_pipe,
        prompt,
        language,
        ref_audio,
        **baseline_kwargs,
    )
    assert baseline_sr == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, got {baseline_sr}"

    variants = [
        {"subtalker_top_k": 1},
        {"subtalker_top_p": 0.2},
        {"subtalker_temperature": 0.2},
    ]

    differs = False
    for variant in variants:
        variant_speech, variant_sr = _run_genai_base_generate(
            genai_qwen3_tts_base_pipe,
            prompt,
            language,
            ref_audio,
            **baseline_kwargs,
            **variant,
        )
        assert variant_sr == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, got {variant_sr}"
        if not np.array_equal(baseline_speech, variant_speech):
            differs = True
            break

    assert differs, (
        "Expected at least one subtalker sampling knob variant (top_k/top_p/temperature) "
        "to produce a different waveform from seeded baseline."
    )


@pytest.mark.speech_generation
def test_qwen3_tts_base_sampling_talker_knobs_change_output(
    genai_qwen3_tts_base_pipe,
):
    language = "English"
    prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
    ref_audio = _make_ref_audio()

    baseline_kwargs = {
        "max_new_tokens": 128,
        "rng_seed": 2029,
        "do_sample": True,
        "subtalker_dosample": False,
    }
    baseline_speech, baseline_sr = _run_genai_base_generate(
        genai_qwen3_tts_base_pipe,
        prompt,
        language,
        ref_audio,
        **baseline_kwargs,
    )
    assert baseline_sr == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, got {baseline_sr}"

    variants = [
        {"top_k": 1},
        {"top_p": 0.2},
        {"temperature": 0.2},
    ]

    differs = False
    for variant in variants:
        variant_speech, variant_sr = _run_genai_base_generate(
            genai_qwen3_tts_base_pipe,
            prompt,
            language,
            ref_audio,
            **baseline_kwargs,
            **variant,
        )
        assert variant_sr == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, got {variant_sr}"
        if not np.array_equal(baseline_speech, variant_speech):
            differs = True
            break

    assert differs, (
        "Expected at least one talker sampling knob variant (top_k/top_p/temperature) "
        "to produce a different waveform from seeded baseline."
    )


@pytest.mark.speech_generation
@pytest.mark.parametrize(
    "invalid_kwargs,expected_message",
    [
        ({"subtalker_top_k": 0}, "subtalker_top_k must be positive"),
        ({"subtalker_top_p": -0.1}, "subtalker_top_p must be in the range [0; 1]"),
        ({"subtalker_top_p": 1.1}, "subtalker_top_p must be in the range [0; 1]"),
        ({"subtalker_temperature": 0.0}, "subtalker_temperature must be positive"),
    ],
)
def test_qwen3_tts_base_sampling_invalid_subtalker_settings_raises(
    genai_qwen3_tts_base_pipe,
    invalid_kwargs,
    expected_message,
):
    language = "English"
    prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
    ref_audio = _make_ref_audio()

    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        genai_qwen3_tts_base_pipe.generate(
            prompt,
            language=language,
            ref_audio=ov.Tensor(ref_audio),
            max_new_tokens=128,
            rng_seed=2028,
            subtalker_dosample=True,
            **invalid_kwargs,
        )


@pytest.mark.speech_generation
@pytest.mark.parametrize(
    "invalid_kwargs,expected_message",
    [
        ({"top_p": 0.0}, "When 'do_sample' is true, top_p must be a positive float > 0.0 and <= 1.0"),
        ({"top_p": 1.1}, "When 'do_sample' is true, top_p must be a positive float > 0.0 and <= 1.0"),
        ({"temperature": 0.0}, "When 'do_sample' is true, temperature must be a strictly positive float"),
    ],
)
def test_qwen3_tts_base_sampling_invalid_talker_settings_raises(
    genai_qwen3_tts_base_pipe,
    invalid_kwargs,
    expected_message,
):
    language = "English"
    prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
    ref_audio = _make_ref_audio()

    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        genai_qwen3_tts_base_pipe.generate(
            prompt,
            language=language,
            ref_audio=ov.Tensor(ref_audio),
            max_new_tokens=128,
            rng_seed=2030,
            do_sample=True,
            subtalker_dosample=False,
            **invalid_kwargs,
        )


@pytest.mark.speech_generation
def test_qwen3_tts_base_optimum_vs_genai_without_language(
    optimum_qwen3_tts_base_model,
    genai_qwen3_tts_base_pipe,
):
    prompt = LANGUAGE_TEST_TEXTS["English"]["prompt"]
    ref_audio = _make_ref_audio()

    inputs = optimum_qwen3_tts_base_model.preprocess_input(
        text=prompt,
        ref_audio=(ref_audio, SAMPLE_RATE),
        x_vector_only_mode=True,
    )
    optimum_output = optimum_qwen3_tts_base_model.generate(
        **inputs,
        **_generation_kwargs(non_streaming_mode=True),
    )
    optimum_speech = _to_waveform_array(optimum_output)
    optimum_sr = int(getattr(optimum_qwen3_tts_base_model, "sampling_rate", SAMPLE_RATE))

    result = genai_qwen3_tts_base_pipe.generate(
        prompt,
        ref_audio=ov.Tensor(ref_audio),
        **_generation_kwargs(non_streaming_mode=True),
    )
    genai_speech = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

    assert result.output_sample_rate == SAMPLE_RATE, (
        f"GenAI Base sample rate mismatch without language: "
        f"expected={SAMPLE_RATE}, actual={result.output_sample_rate}"
    )
    assert result.output_sample_rate == optimum_sr, (
        f"Base sample rate mismatch without language: "
        f"optimum={optimum_sr}, genai={result.output_sample_rate}"
    )
    _assert_waveform_equal(optimum_speech, genai_speech, "base parity without language")


@pytest.mark.speech_generation
def test_qwen3_tts_customvoice_optimum_vs_genai_without_language(
    optimum_qwen3_tts_customvoice_model,
    genai_qwen3_tts_customvoice_pipe,
):
    prompt = LANGUAGE_TEST_TEXTS["English"]["prompt"]
    instruct = LANGUAGE_TEST_TEXTS["English"]["instruct"]
    speaker = LANGUAGE_TEST_TEXTS["English"]["customvoice_speaker"]

    inputs = optimum_qwen3_tts_customvoice_model.preprocess_input(
        text=prompt,
        speaker=speaker,
        instruct=instruct,
    )
    optimum_output = optimum_qwen3_tts_customvoice_model.generate(
        **inputs,
        **_generation_kwargs(non_streaming_mode=True),
    )
    optimum_speech = _to_waveform_array(optimum_output)
    optimum_sr = int(getattr(optimum_qwen3_tts_customvoice_model, "sampling_rate", SAMPLE_RATE))

    result = genai_qwen3_tts_customvoice_pipe.generate(
        prompt,
        speaker=speaker,
        instruct=instruct,
        **_generation_kwargs(non_streaming_mode=True),
    )
    genai_speech = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

    assert result.output_sample_rate == SAMPLE_RATE, (
        f"GenAI CustomVoice sample rate mismatch without language: "
        f"expected={SAMPLE_RATE}, actual={result.output_sample_rate}"
    )
    assert result.output_sample_rate == optimum_sr, (
        f"CustomVoice sample rate mismatch without language: "
        f"optimum={optimum_sr}, genai={result.output_sample_rate}"
    )
    _assert_waveform_equal(optimum_speech, genai_speech, "customvoice parity without language")


@pytest.mark.speech_generation
def test_qwen3_tts_customvoice_invalid_language_hides_dialects_in_supported_list(
    genai_qwen3_tts_customvoice_pipe,
):
    prompt = LANGUAGE_TEST_TEXTS["English"]["prompt"]
    instruct = LANGUAGE_TEST_TEXTS["English"]["instruct"]
    speaker = LANGUAGE_TEST_TEXTS["English"]["customvoice_speaker"]

    with pytest.raises(RuntimeError) as error:
        genai_qwen3_tts_customvoice_pipe.generate(
            prompt,
            language="SomeInvalidLanguage",
            speaker=speaker,
            instruct=instruct,
            **_generation_kwargs(non_streaming_mode=True),
        )

    message = str(error.value)
    assert "'beijing_dialect'" not in message
    assert "'sichuan_dialect'" not in message
    assert "'auto'" in message
    assert "'chinese'" in message


@pytest.mark.speech_generation
@pytest.mark.parametrize(
    "language",
    ["Chinese", "Auto"],
    ids=["chinese", "auto"],
)
def test_qwen3_tts_customvoice_dialect_speaker_language_parity_optimum_vs_genai(
    optimum_qwen3_tts_customvoice_model,
    genai_qwen3_tts_customvoice_pipe,
    language: str,
):
    dialect_speaker = "eric"
    get_supported_speakers = getattr(optimum_qwen3_tts_customvoice_model, "get_supported_speakers", None)
    if callable(get_supported_speakers):
        supported_speakers = {str(s).lower() for s in get_supported_speakers()}
        if dialect_speaker not in supported_speakers:
            pytest.fail(f"Speaker '{dialect_speaker}' is not available in this model fixture")

    prompt = LANGUAGE_TEST_TEXTS["Chinese"]["prompt"]
    instruct = LANGUAGE_TEST_TEXTS["Chinese"]["instruct"]

    inputs = optimum_qwen3_tts_customvoice_model.preprocess_input(
        text=prompt,
        language=language,
        speaker=dialect_speaker,
        instruct=instruct,
    )
    optimum_output = optimum_qwen3_tts_customvoice_model.generate(
        **inputs,
        **_generation_kwargs(non_streaming_mode=True),
    )
    optimum_speech = _to_waveform_array(optimum_output)
    optimum_sr = int(getattr(optimum_qwen3_tts_customvoice_model, "sampling_rate", SAMPLE_RATE))

    result = genai_qwen3_tts_customvoice_pipe.generate(
        prompt,
        language=language,
        speaker=dialect_speaker,
        instruct=instruct,
        **_generation_kwargs(non_streaming_mode=True),
    )
    genai_speech = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

    assert result.output_sample_rate == SAMPLE_RATE, (
        f"GenAI CustomVoice sample rate mismatch for language={language}+dialect speaker parity: "
        f"expected={SAMPLE_RATE}, actual={result.output_sample_rate}"
    )
    assert result.output_sample_rate == optimum_sr, (
        f"CustomVoice sample rate mismatch for language={language}+dialect speaker parity: "
        f"optimum={optimum_sr}, genai={result.output_sample_rate}"
    )
    _assert_waveform_equal(optimum_speech, genai_speech, f"customvoice {language}+dialect parity")


@pytest.mark.speech_generation
def test_qwen3_tts_voicedesign_optimum_vs_genai_without_language(
    optimum_qwen3_tts_voicedesign_model,
    genai_qwen3_tts_voicedesign_pipe,
):
    prompt = LANGUAGE_TEST_TEXTS["English"]["prompt"]
    instruct = LANGUAGE_TEST_TEXTS["English"]["instruct"]

    inputs = optimum_qwen3_tts_voicedesign_model.preprocess_input(
        text=prompt,
        instruct=instruct,
    )
    optimum_output = optimum_qwen3_tts_voicedesign_model.generate(
        **inputs,
        **_generation_kwargs(non_streaming_mode=True),
    )
    optimum_speech = _to_waveform_array(optimum_output)
    optimum_sr = int(getattr(optimum_qwen3_tts_voicedesign_model, "sampling_rate", SAMPLE_RATE))

    result = genai_qwen3_tts_voicedesign_pipe.generate(
        prompt,
        instruct=instruct,
        **_generation_kwargs(non_streaming_mode=True),
    )
    genai_speech = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

    assert result.output_sample_rate == SAMPLE_RATE, (
        f"GenAI VoiceDesign sample rate mismatch without language: "
        f"expected={SAMPLE_RATE}, actual={result.output_sample_rate}"
    )
    assert result.output_sample_rate == optimum_sr, (
        f"VoiceDesign sample rate mismatch without language: "
        f"optimum={optimum_sr}, genai={result.output_sample_rate}"
    )
    _assert_waveform_equal(optimum_speech, genai_speech, "voicedesign parity without language")


@pytest.mark.speech_generation
@pytest.mark.parametrize(
    "non_streaming_mode",
    [True, False],
    ids=["non-streaming", "streaming"],
)
@pytest.mark.parametrize("language", LANGUAGES)
class TestQwen3TTSPipelineBase:
    @pytest.mark.parametrize(
        "mode,use_reference_text,x_vector_only_mode",
        [
            pytest.param("xvector", False, True, id="optimum-vs-genai-xvector"),
            pytest.param("icl", True, False, id="optimum-vs-genai-icl"),
        ],
    )
    def test_qwen3_tts_base_optimum_vs_genai(
        self,
        optimum_qwen3_tts_base_model,
        genai_qwen3_tts_base_pipe,
        mode: str,
        language: str,
        use_reference_text: bool,
        x_vector_only_mode: bool,
        non_streaming_mode: bool,
    ):
        ref_audio = _make_ref_audio()
        prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
        ref_text = LANGUAGE_TEST_TEXTS[language]["reference_text"] if use_reference_text else None

        optimum_speech, optimum_sr = _run_optimum_base_generate(
            optimum_qwen3_tts_base_model,
            prompt,
            language,
            ref_audio,
            ref_text,
            x_vector_only_mode,
            non_streaming_mode,
        )

        generation_properties = {
            "language": language,
            "ref_audio": ov.Tensor(ref_audio),
            **_generation_kwargs(non_streaming_mode),
        }
        if ref_text is not None:
            generation_properties["ref_text"] = ref_text

        result = genai_qwen3_tts_base_pipe.generate(prompt, **generation_properties)
        genai_speech = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

        assert result.output_sample_rate == SAMPLE_RATE, (
            f"GenAI output sample rate mismatch for mode={mode}, language={language}: "
            f"expected={SAMPLE_RATE}, actual={result.output_sample_rate}"
        )

        assert result.output_sample_rate == optimum_sr, (
            f"Output sample rate mismatch for mode={mode}: optimum={optimum_sr}, genai={result.output_sample_rate}"
        )

        _assert_waveform_equal(optimum_speech, genai_speech, f"mode={mode}")

    def test_qwen3_tts_base_reuse_artifacts_icl(
        self,
        genai_qwen3_tts_base_pipe,
        non_streaming_mode: bool,
        language: str,
    ):
        ref_audio = _make_ref_audio()
        prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
        ref_text = LANGUAGE_TEST_TEXTS[language]["reference_text"]
        generation_properties = {
            "language": language,
            "ref_audio": ov.Tensor(ref_audio),
            "ref_text": ref_text,
            **_generation_kwargs(non_streaming_mode),
        }

        first = genai_qwen3_tts_base_pipe.generate(prompt, **generation_properties)
        assert first.speaker_embedding, "First ICL generation did not return speaker_embedding"
        assert first.ref_codec_ids, "First ICL generation did not return ref_codec_ids"
        assert first.output_sample_rate == SAMPLE_RATE, (
            f"First ICL sample rate mismatch for language={language}: "
            f"expected={SAMPLE_RATE}, actual={first.output_sample_rate}"
        )

        second = genai_qwen3_tts_base_pipe.generate(
            prompt,
            first.speaker_embedding,
            language=language,
            ref_text=ref_text,
            ref_codec_ids=first.ref_codec_ids,
            **_generation_kwargs(non_streaming_mode),
        )
        assert second.output_sample_rate == SAMPLE_RATE, (
            f"Second ICL sample rate mismatch for language={language}: "
            f"expected={SAMPLE_RATE}, actual={second.output_sample_rate}"
        )

        first_speech = np.array(first.speeches[0].data, dtype=np.float32).reshape(-1)
        second_speech = np.array(second.speeches[0].data, dtype=np.float32).reshape(-1)

        assert first.output_sample_rate == second.output_sample_rate
        _assert_waveform_equal(first_speech, second_speech, "icl artifact reuse")

    def test_qwen3_tts_base_reuse_artifacts_xvector(
        self,
        genai_qwen3_tts_base_pipe,
        non_streaming_mode: bool,
        language: str,
    ):
        ref_audio = _make_ref_audio()
        prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
        first = genai_qwen3_tts_base_pipe.generate(
            prompt,
            language=language,
            ref_audio=ov.Tensor(ref_audio),
            **_generation_kwargs(non_streaming_mode),
        )
        assert first.speaker_embedding, "First x-vector generation did not return speaker_embedding"
        assert first.output_sample_rate == SAMPLE_RATE, (
            f"First x-vector sample rate mismatch for language={language}: "
            f"expected={SAMPLE_RATE}, actual={first.output_sample_rate}"
        )

        second = genai_qwen3_tts_base_pipe.generate(
            prompt,
            first.speaker_embedding,
            language=language,
            **_generation_kwargs(non_streaming_mode),
        )
        assert second.output_sample_rate == SAMPLE_RATE, (
            f"Second x-vector sample rate mismatch for language={language}: "
            f"expected={SAMPLE_RATE}, actual={second.output_sample_rate}"
        )

        first_speech = np.array(first.speeches[0].data, dtype=np.float32).reshape(-1)
        second_speech = np.array(second.speeches[0].data, dtype=np.float32).reshape(-1)

        assert first.output_sample_rate == second.output_sample_rate
        _assert_waveform_equal(first_speech, second_speech, "x-vector artifact reuse")


@pytest.mark.speech_generation
@pytest.mark.parametrize(
    "non_streaming_mode",
    [True, False],
    ids=["non-streaming", "streaming"],
)
@pytest.mark.parametrize("language", LANGUAGES)
class TestQwen3TTSPipelineCustomVoice:
    @pytest.mark.parametrize("use_instruct", [True, False], ids=["with-instruct", "without-instruct"])
    def test_qwen3_tts_customvoice_optimum_vs_genai(
        self,
        optimum_qwen3_tts_customvoice_model,
        genai_qwen3_tts_customvoice_pipe,
        language: str,
        non_streaming_mode: bool,
        use_instruct: bool,
    ):
        prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
        instruct = LANGUAGE_TEST_TEXTS[language]["instruct"] if use_instruct else None
        speaker = LANGUAGE_TEST_TEXTS[language]["customvoice_speaker"]

        preprocess_kwargs = dict(
            text=prompt,
            language=language,
            speaker=speaker,
        )
        if instruct is not None:
            preprocess_kwargs["instruct"] = instruct

        inputs = optimum_qwen3_tts_customvoice_model.preprocess_input(**preprocess_kwargs)
        optimum_output = optimum_qwen3_tts_customvoice_model.generate(
            **inputs,
            **_generation_kwargs(non_streaming_mode),
        )
        optimum_speech = _to_waveform_array(optimum_output)
        optimum_sr = int(getattr(optimum_qwen3_tts_customvoice_model, "sampling_rate", SAMPLE_RATE))

        generation_kwargs = {
            "text": prompt,
            "language": language,
            "speaker": speaker,
            **_generation_kwargs(non_streaming_mode),
        }
        if instruct is not None:
            generation_kwargs["instruct"] = instruct

        result = genai_qwen3_tts_customvoice_pipe.generate(**generation_kwargs)
        genai_speech = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

        assert result.output_sample_rate == SAMPLE_RATE, (
            f"GenAI CustomVoice sample rate mismatch for language={language}: "
            f"expected={SAMPLE_RATE}, actual={result.output_sample_rate}"
        )
        assert result.output_sample_rate == optimum_sr, (
            f"CustomVoice sample rate mismatch for language={language}: "
            f"optimum={optimum_sr}, genai={result.output_sample_rate}"
        )
        _assert_waveform_equal(optimum_speech, genai_speech, "customvoice parity")


@pytest.mark.speech_generation
@pytest.mark.parametrize(
    "non_streaming_mode",
    [True, False],
    ids=["non-streaming", "streaming"],
)
@pytest.mark.parametrize("language", LANGUAGES)
class TestQwen3TTSPipelineVoiceDesign:
    def test_qwen3_tts_voicedesign_optimum_vs_genai(
        self,
        optimum_qwen3_tts_voicedesign_model,
        genai_qwen3_tts_voicedesign_pipe,
        language: str,
        non_streaming_mode: bool,
    ):
        prompt = LANGUAGE_TEST_TEXTS[language]["prompt"]
        instruct = LANGUAGE_TEST_TEXTS[language]["instruct"]

        inputs = optimum_qwen3_tts_voicedesign_model.preprocess_input(
            text=prompt,
            language=language,
            instruct=instruct,
        )
        optimum_output = optimum_qwen3_tts_voicedesign_model.generate(
            **inputs,
            **_generation_kwargs(non_streaming_mode),
        )
        optimum_speech = _to_waveform_array(optimum_output)
        optimum_sr = int(getattr(optimum_qwen3_tts_voicedesign_model, "sampling_rate", SAMPLE_RATE))

        result = genai_qwen3_tts_voicedesign_pipe.generate(
            prompt,
            language=language,
            instruct=instruct,
            **_generation_kwargs(non_streaming_mode),
        )
        genai_speech = np.array(result.speeches[0].data, dtype=np.float32).reshape(-1)

        assert result.output_sample_rate == SAMPLE_RATE, (
            f"GenAI VoiceDesign sample rate mismatch for language={language}: "
            f"expected={SAMPLE_RATE}, actual={result.output_sample_rate}"
        )
        assert result.output_sample_rate == optimum_sr, (
            f"VoiceDesign sample rate mismatch for language={language}: "
            f"optimum={optimum_sr}, genai={result.output_sample_rate}"
        )
        _assert_waveform_equal(optimum_speech, genai_speech, "voicedesign parity")
