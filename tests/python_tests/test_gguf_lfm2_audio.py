# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optional real-checkpoint LFM2.5 Audio Omni API regression tests."""
import os
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as genai
import pytest


@pytest.fixture(scope="module")
def lfm2_bundle():
    path = os.environ.get("OV_LFM2_AUDIO_BUNDLE")
    if not path:
        pytest.skip("Set OV_LFM2_AUDIO_BUNDLE to the pinned LFM2.5 Audio GGUF bundle")
    path = Path(path)
    return path, {
        "mmproj_path": str(path / "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"),
        "vocoder_path": str(path / "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf"),
        "audio_detokenizer_path": str(path / "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"),
        "INFERENCE_PRECISION_HINT": "f32",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
        "INFERENCE_NUM_THREADS": 4,
    }


@pytest.fixture(scope="module")
def lfm2_speech(lfm2_bundle):
    path, properties = lfm2_bundle
    return genai.OmniPipeline(str(path / "LFM2.5-Audio-1.5B-Q4_0.gguf"), "CPU", **properties)


def test_lfm2_speech_repeat_streaming_and_text_only(lfm2_speech):
    config = genai.GenerationConfig(max_new_tokens=24, do_sample=False)
    chunks = []
    result = lfm2_speech.generate("Say hello.", text_config=config,
                                 speech_streamer=lambda chunk: chunks.append(chunk.data.copy()))
    waveform = result.speech_result.waveforms[0].data.copy()
    assert waveform.shape[0] == 1 and waveform.size > 0 and np.isfinite(waveform).all()
    assert np.array_equal(np.concatenate(chunks, axis=1), waveform)
    repeated = lfm2_speech.generate("Say hello.", text_config=config)
    assert repeated.texts == result.texts
    assert np.array_equal(repeated.speech_result.waveforms[0].data, waveform)
    speech = genai.OmniTalkerSpeechConfig()
    speech.return_audio = False
    assert not lfm2_speech.generate("Say hello.", text_config=config, talker_speech_config=speech).speech_result.waveforms


def test_lfm2_speech_rejects_unsupported_options_and_recovers(lfm2_speech):
    for settings in ({"num_beams": 2}, {"repetition_penalty": 1.2}, {"min_p": 0.1}):
        with pytest.raises(RuntimeError):
            lfm2_speech.generate("Say hello.", text_config=genai.GenerationConfig(max_new_tokens=8, **settings))
    speech = genai.OmniTalkerSpeechConfig()
    speech.speaker = "another voice"
    with pytest.raises(RuntimeError):
        lfm2_speech.generate("Say hello.", text_config=genai.GenerationConfig(max_new_tokens=8), talker_speech_config=speech)
    class Cancel(genai.StreamerBase):
        def write(self, tokens):
            return genai.StreamingStatus.CANCEL
        def end(self):
            pass
    result = lfm2_speech.generate("Say hello.", text_config=genai.GenerationConfig(max_new_tokens=8), streamer=Cancel())
    assert not result.speech_result.waveforms
    assert lfm2_speech.generate("Say hello.", text_config=genai.GenerationConfig(max_new_tokens=8)).texts


def test_lfm2_speech_audio_chat_reuses_media(lfm2_speech):
    samples = np.fromfile(Path(__file__).parents[1] / "cpp/data/lfm2_audio/waveform-16000.bin", dtype=np.float32)
    history = genai.ChatHistory([{"role": "user", "content": "<audio> Transcribe the audio."}])
    speech = genai.OmniTalkerSpeechConfig()
    speech.return_audio = False
    config = genai.GenerationConfig(max_new_tokens=8)
    result = lfm2_speech.generate(history, audios=[ov.Tensor(samples)], text_config=config, talker_speech_config=speech)
    history.append({"role": "assistant", "content": result.texts[0]})
    history.append({"role": "user", "content": "What did the speaker say?"})
    assert lfm2_speech.generate(history, text_config=config, talker_speech_config=speech).texts


def test_lfm2_sequential_speech(lfm2_bundle):
    path, properties = lfm2_bundle
    pipe = genai.OmniPipeline(str(path / "LFM2.5-Audio-1.5B-Q4_0.gguf"), "CPU",
                             lfm2_audio_mode="sequential", **properties)
    result = pipe.generate("Hello.", text_config=genai.GenerationConfig(max_new_tokens=48))
    assert result.speech_result.waveforms
    assert np.isfinite(result.speech_result.waveforms[0].data).all()
