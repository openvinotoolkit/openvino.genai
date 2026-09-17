# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess  # nosec B404
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

import openvino as ov
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForTextToSpeechSeq2Seq

from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir
from utils.network import retry_request

logger = logging.getLogger(__name__)

QWEN3_TTS_BASE_TINY_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-tts"
SAMPLE_RATE = 24000
LANGUAGES = ["English", "Chinese", "Russian"]
LANGUAGE_TEST_TEXTS = {
    "English": {
        "prompt": "Okay, let's run some tests! Is everything okay?",
        "reference_text": "This is a dummy-generated waveform for parity testing",
    },
    "Chinese": {
        "prompt": "好了，我们来做些测试吧！一切都正常吗？",
        "reference_text": "这是一个用于奇偶校验测试的模拟生成波形。",
    },
    "Russian": {
        "prompt": "Хорошо, давайте проведем несколько тестов! Всё в порядке?",
        "reference_text": "Это сгенерированный тестовый сигнал для проверки четности.",
    },
}
GENERATION_KWARGS = {
    "do_sample": False,
    "subtalker_dosample": False,
    "max_new_tokens": 32,
}


def _generation_kwargs(non_streaming_mode: bool) -> dict:
    return {**GENERATION_KWARGS, "non_streaming_mode": non_streaming_mode}


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
    models_dir = get_ov_cache_converted_models_dir()
    model_dir = models_dir / QWEN3_TTS_BASE_TINY_MODEL_ID.replace("/", "_")
    manager = AtomicDownloadManager(model_dir)

    if manager.is_complete() or (model_dir / "openvino_talker_model.xml").exists():
        return model_dir

    def convert_to_temp(temp_dir: Path) -> None:
        model_cached = snapshot_download(QWEN3_TTS_BASE_TINY_MODEL_ID)
        command = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            model_cached,
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
        pytest.skip("optimum-cli is not available in PATH")
    except subprocess.CalledProcessError as error:
        pytest.skip(
            f"Failed to export {QWEN3_TTS_BASE_TINY_MODEL_ID}: returncode={error.returncode}, stderr={error.stderr}"
        )

    return model_dir


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
        pytest.skip(f"Failed to load optimum Qwen3-TTS model: {error}")


@pytest.fixture
def genai_qwen3_tts_base_pipe(tiny_qwen3_tts_base_ov_path: Path):
    return ov_genai.Text2SpeechPipeline(str(tiny_qwen3_tts_base_ov_path), "CPU")


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
