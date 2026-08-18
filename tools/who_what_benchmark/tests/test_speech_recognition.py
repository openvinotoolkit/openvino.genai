# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from whowhatbench import model_loaders
from whowhatbench.whowhat_metrics import WordSimilarity
from whowhatbench.wwb import to_mono_16k
from whowhatbench.speech_recognition_evaluator import (
    DEFAULT_ASR_INSTRUCTION,
    FunASRGenAITranscriber,
    FunASROptimumTranscriber,
    FunASRSourceTranscriber,
    GenAIMultimodalTranscriber,
    MultimodalTranscriber,
    SpeechRecognitionEvaluator,
)


def _frame(answers):
    return pd.DataFrame({"prompts": [str(i) for i in range(len(answers))], "answers": answers})


def _csv(tmp_path, name, prompts, answers):
    path = tmp_path / name
    pd.DataFrame({"prompts": prompts, "answers": answers}).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize(
    "references, hypotheses, corpus, per_prompt",
    [
        # 1 insertion over 6 reference words: the corpus value is 5/6, not the 0.75 mean of the two.
        (["the quick brown fox", "hello world"], ["the quick brown fox", "hello there world"], 5 / 6, [1.0, 0.5]),
        (["hello world"], ["hello world"], 1.0, [1.0]),
        # 3 insertions over 1 reference word: 1 - WER is negative and must clamp to 0.
        (["hello", ""], ["hello", "spurious extra words"], 0.0, [1.0, 0.0]),
        ([], [], 1.0, []),
    ],
)
def test_word_similarity(references, hypotheses, corpus, per_prompt):
    aggregate, per_utterance = WordSimilarity().evaluate(_frame(references), _frame(hypotheses))
    assert per_utterance == {"similarity": per_prompt}
    assert math.isclose(aggregate["similarity"], corpus)


def test_word_similarity_rejects_length_mismatch():
    with pytest.raises(ValueError, match="lengths must match"):
        WordSimilarity().evaluate(_frame(["x", "y"]), _frame(["x"]))


@pytest.mark.parametrize(
    "prompts, answers, error",
    [
        (["a"], ["x"], "differ in length"),
        (["a", "c"], ["x", "y"], "do not match"),
    ],
)
def test_score_rejects_inconsistent_predictions(tmp_path, prompts, answers, error):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a", "b"], ["x", "y"]))
    with pytest.raises(ValueError, match=error):
        evaluator.score(_csv(tmp_path, "target.csv", prompts, answers))


def test_score_rejects_missing_column(tmp_path):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a"], ["x"]))
    bad = tmp_path / "target.csv"
    pd.DataFrame({"prompts": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        evaluator.score(str(bad))


def test_score_reports_similarity(tmp_path):
    gt = _csv(tmp_path, "gt.csv", ["a", "b"], ["the quick brown fox", "hello world"])
    target = _csv(tmp_path, "target.csv", ["a", "b"], ["the quick brown fox", "hello there world"])
    evaluator = SpeechRecognitionEvaluator(gt_data=gt)

    per_prompt, aggregate = evaluator.score(target)
    assert per_prompt.columns.tolist() == ["similarity"]
    assert per_prompt["similarity"].tolist() == [1.0, 0.5]
    assert math.isclose(aggregate["similarity"].iloc[0], 5 / 6)
    assert [example["prompt"] for example in evaluator.worst_examples(top_k=1)] == ["b"]


class _FakeTranscriber:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.calls = []

    def transcribe(self, audio, max_new_tokens):
        self.calls.append((len(audio), max_new_tokens))
        return self.transcripts[len(self.calls) - 1]


AUDIO_DATA = {"prompts": ["a", "b"], "audio": [np.zeros(4, dtype=np.float32), np.zeros(8, dtype=np.float32)]}


def test_evaluator_delegates_to_transcriber():
    base = _FakeTranscriber(["hello world", "second one"])
    evaluator = SpeechRecognitionEvaluator(base_model=base, test_data=AUDIO_DATA, max_new_tokens=42)

    assert base.calls == [(4, 42), (8, 42)]
    assert list(evaluator.gt_data["prompts"]) == ["a", "b"]
    assert list(evaluator.gt_data["answers"]) == ["hello world", "second one"]

    target = _FakeTranscriber(["hello world", "second two"])
    _, aggregate = evaluator.score(target)
    assert target.calls == [(4, 42), (8, 42)]
    assert 0.0 < aggregate["similarity"].iloc[0] < 1.0


def test_evaluator_honours_num_samples():
    base = _FakeTranscriber(["only one", "unused"])
    evaluator = SpeechRecognitionEvaluator(base_model=base, test_data=AUDIO_DATA, num_samples=1)
    assert base.calls == [(4, 256)]
    assert list(evaluator.gt_data["prompts"]) == ["a"]


def test_evaluator_rejects_model_without_transcribe():
    with pytest.raises(TypeError, match="transcribe"):
        SpeechRecognitionEvaluator(base_model=object(), test_data=AUDIO_DATA)


def test_evaluator_uses_explicit_gen_answer_fn():
    calls = []

    def gen_answer_fn(model, audio, max_new_tokens):
        calls.append(max_new_tokens)
        return "custom"

    evaluator = SpeechRecognitionEvaluator(
        base_model=object(), test_data=AUDIO_DATA, max_new_tokens=7, gen_answer_fn=gen_answer_fn
    )
    assert calls == [7, 7]
    assert list(evaluator.gt_data["answers"]) == ["custom", "custom"]
    assert evaluator.get_generation_fn() is gen_answer_fn


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


DETECTION_CASES = [
    ({"configuration.json": {"framework": "pytorch", "model": {"type": "funasr"}}}, "source"),
    ({"config.json": {"model_type": "fun_asr"}}, "export"),
    ({}, None),
    ({"config.json": {"model_type": "gemma4_unified"}}, None),
    ({"configuration.json": {"model": {"type": "sensevoice"}}}, None),
    ({"configuration.json": {"model": "funasr"}}, None),
]


@pytest.mark.parametrize("files, kind", DETECTION_CASES)
def test_funasr_model_kind_local(tmp_path, files, kind):
    for name, payload in files.items():
        _write_json(tmp_path / name, payload)
    assert model_loaders.funasr_model_kind(str(tmp_path)) == kind


@pytest.mark.parametrize("files, kind", DETECTION_CASES)
def test_funasr_model_kind_remote(monkeypatch, tmp_path, files, kind):
    import huggingface_hub

    for name, payload in files.items():
        _write_json(tmp_path / name, payload)

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        if filename not in files:
            raise FileNotFoundError(filename)  # the hub 404s for a file the repo does not have
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    assert model_loaders.funasr_model_kind("FunAudioLLM/Fun-ASR-Nano-2512") == kind


class _FakeFunASRAutoModel:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.generate_kwargs = None
        _FakeFunASRAutoModel.instances.append(self)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [{"text": " hello world "}]


@pytest.fixture
def fake_funasr(monkeypatch):
    _FakeFunASRAutoModel.instances = []
    module = types.ModuleType("funasr")
    module.AutoModel = _FakeFunASRAutoModel
    monkeypatch.setitem(sys.modules, "funasr", module)
    return _FakeFunASRAutoModel


class _FakeOVSpeechSeq2Seq:
    from_pretrained_kwargs = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.from_pretrained_kwargs = {"model_id": model_id, **kwargs}
        return cls()

    def preprocess_input(self, waveform, sampling_rate, **kwargs):
        self.preprocess_call = {"sampling_rate": sampling_rate, **kwargs}
        return {
            "input_features": torch.zeros(1, 2, 3),
            "decoder_input_ids": torch.tensor([[1, 2, 3]]),
        }

    def generate(self, **kwargs):
        self.generate_call = kwargs
        return torch.tensor([[1, 2, 3, 7, 8]])


class _FakeTokenizer:
    def __init__(self, location, **kwargs):
        self.location = location
        self.kwargs = kwargs
        self.decoded = None

    def batch_decode(self, ids, **kwargs):
        self.decoded = ids
        return [" decoded "]


@pytest.fixture
def fake_optimum(monkeypatch):
    module = types.ModuleType("optimum.intel.openvino")
    module.OVModelForSpeechSeq2Seq = _FakeOVSpeechSeq2Seq
    monkeypatch.setitem(sys.modules, "optimum.intel.openvino", module)
    monkeypatch.setattr(model_loaders, "AutoTokenizer", types.SimpleNamespace(from_pretrained=_FakeTokenizer))
    return module


class _FakeASRPipeline:
    def __init__(self, models_path, device, **properties):
        self.models_path = models_path
        self.device = device
        self.properties = properties
        self.generate_call = None

    def generate(self, audio, **kwargs):
        self.generate_call = {"audio": audio, **kwargs}
        return types.SimpleNamespace(texts=[" transcript "])


@pytest.fixture
def fake_genai(monkeypatch):
    module = types.ModuleType("openvino_genai")
    module.ASRPipeline = _FakeASRPipeline
    monkeypatch.setitem(sys.modules, "openvino_genai", module)
    return module


def test_load_speech_recognition_model_dispatches_funasr_source(monkeypatch, fake_funasr):
    monkeypatch.setattr(model_loaders, "funasr_model_kind", lambda model_id: "source")
    model = model_loaders.load_speech_recognition_model(
        "FunAudioLLM/Fun-ASR-Nano-2512", use_hf=True, speech_language="zh"
    )

    assert isinstance(model, FunASRSourceTranscriber)
    assert fake_funasr.instances[0].init_kwargs["model"] == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert fake_funasr.instances[0].init_kwargs["hub"] == "hf"
    assert "trust_remote_code" not in fake_funasr.instances[0].init_kwargs
    assert model.language == "zh"


@pytest.mark.parametrize("kind, subfolder", [("source", "Qwen3-0.6B"), ("export", "")])
def test_load_speech_recognition_model_dispatches_funasr_optimum(monkeypatch, fake_optimum, kind, subfolder):
    monkeypatch.setattr(model_loaders, "funasr_model_kind", lambda model_id: kind)
    model = model_loaders.load_speech_recognition_model(
        "fun-asr-ov", device="CPU", ov_config={"CACHE_DIR": ""}, speech_language="en"
    )

    assert isinstance(model, FunASROptimumTranscriber)
    assert _FakeOVSpeechSeq2Seq.from_pretrained_kwargs == {
        "model_id": "fun-asr-ov",
        "device": "CPU",
        "ov_config": {"CACHE_DIR": ""},
    }
    assert model.preprocess_kwargs == {"language": "en"}
    # a source repo keeps the LLM tokenizer in a subfolder, an export keeps it next to the model
    assert (model.tokenizer.location, model.tokenizer.kwargs["subfolder"]) == ("fun-asr-ov", subfolder)


@pytest.mark.parametrize("requested, language", [("zh", "zh"), ("", "en")])
def test_load_speech_recognition_model_dispatches_funasr_genai(monkeypatch, fake_genai, requested, language):
    monkeypatch.setattr(model_loaders, "funasr_model_kind", lambda model_id: "export")
    model = model_loaders.load_speech_recognition_model(
        "fun-asr-ov", device="cpu", ov_config={"CACHE_DIR": "cache"}, use_genai=True, speech_language=requested
    )

    assert isinstance(model, FunASRGenAITranscriber)
    assert (model.pipeline.models_path, model.pipeline.device) == ("fun-asr-ov", "CPU")
    assert model.pipeline.properties == {"CACHE_DIR": "cache"}
    assert model.language == language


def test_load_speech_recognition_model_dispatches_audio_vlm(monkeypatch):
    loaded = {}

    def fake_load_visual_text_model(model_id, device, ov_config, use_hf, use_genai, **kwargs):
        loaded.update({"model_id": model_id, "use_hf": use_hf, "use_genai": use_genai, "kwargs": kwargs})
        return "vlm-model"

    monkeypatch.setattr(model_loaders, "funasr_model_kind", lambda model_id: None)
    monkeypatch.setattr(model_loaders, "load_visual_text_model", fake_load_visual_text_model)
    monkeypatch.setattr(model_loaders, "_load_audio_vlm_processor", lambda model_id: "processor")

    model = model_loaders.load_speech_recognition_model("google/gemma-4-E4B-it", use_hf=True, speech_language="")
    assert isinstance(model, MultimodalTranscriber)
    assert (model.model, model.processor) == ("vlm-model", "processor")
    assert model.instruction == "Transcribe this audio in English."
    # the audio VLM path reuses the visual-text loaders and must not receive ASR-only arguments
    assert loaded["kwargs"] == {"model_type": "visual-text"}

    genai_model = model_loaders.load_speech_recognition_model("gemma-4-ov", use_genai=True, speech_language="Japanese")
    assert isinstance(genai_model, GenAIMultimodalTranscriber)
    assert genai_model.instruction == "Transcribe this audio in Japanese."


@pytest.mark.parametrize("requested, forwarded", [("", None), ("zh", "zh")])
def test_funasr_source_transcriber_maps_arguments(fake_funasr, requested, forwarded):
    transcriber = FunASRSourceTranscriber("FunAudioLLM/Fun-ASR-Nano-2512", requested)
    assert transcriber.transcribe(np.zeros(16, dtype=np.float32), 64) == "hello world"

    generate_kwargs = fake_funasr.instances[0].generate_kwargs
    assert generate_kwargs["language"] is forwarded
    assert generate_kwargs["max_length"] == 64
    assert generate_kwargs["itn"] is True
    assert generate_kwargs["batch_size"] == 1


def test_funasr_source_transcriber_requires_funasr(monkeypatch):
    monkeypatch.setitem(sys.modules, "funasr", None)
    with pytest.raises(ModuleNotFoundError, match="pip install funasr"):
        FunASRSourceTranscriber("FunAudioLLM/Fun-ASR-Nano-2512")


@pytest.mark.parametrize(
    "language, preprocess_call",
    [("en", {"sampling_rate": 16000, "language": "en"}), ("", {"sampling_rate": 16000})],
)
def test_funasr_optimum_transcriber_decodes_generated_ids_only(language, preprocess_call):
    model, tokenizer = _FakeOVSpeechSeq2Seq(), _FakeTokenizer("export")

    assert (
        FunASROptimumTranscriber(model, tokenizer, language).transcribe(np.zeros(16, dtype=np.float32), 32) == "decoded"
    )
    assert model.preprocess_call == preprocess_call
    assert model.generate_call["max_new_tokens"] == 32
    # the 3 prompt ids are dropped, only the generated ids are decoded
    assert tokenizer.decoded.tolist() == [[7, 8]]

    model.generate = lambda **kwargs: types.SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 7, 8]]))
    assert (
        FunASROptimumTranscriber(model, tokenizer, language).transcribe(np.zeros(16, dtype=np.float32), 32) == "decoded"
    )
    assert tokenizer.decoded.tolist() == [[7, 8]]


@pytest.mark.parametrize(
    "language, generate_call",
    [
        ("", {"audio": [0.0, 0.0, 0.0], "max_new_tokens": 16}),
        ("en", {"audio": [0.0, 0.0, 0.0], "max_new_tokens": 16, "language": "en"}),
    ],
)
def test_funasr_genai_transcriber_forwards_language(language, generate_call):
    pipeline = _FakeASRPipeline("dir", "CPU")
    assert FunASRGenAITranscriber(pipeline, language).transcribe(np.zeros(3, dtype=np.float32), 16) == "transcript"
    assert pipeline.generate_call == generate_call


class _FakeVLMProcessor:
    def __init__(self):
        self.messages = None
        self.decoded = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return {"input_ids": torch.tensor([[1, 2]]), "audio_values": torch.zeros(1, 2)}

    def batch_decode(self, ids, **kwargs):
        self.decoded = ids
        return ["multimodal transcript"]


class _FakeVLMModel:
    def generate(self, **kwargs):
        self.generate_call = kwargs
        return torch.tensor([[1, 2, 5, 6]])


@pytest.mark.parametrize(
    "language, instruction",
    [("", DEFAULT_ASR_INSTRUCTION), ("English", "Transcribe this audio in English.")],
)
def test_multimodal_transcriber_prompts_and_slices_prompt(language, instruction):
    model, processor = _FakeVLMModel(), _FakeVLMProcessor()
    audio = np.zeros(8, dtype=np.float32)

    assert MultimodalTranscriber(model, processor, language).transcribe(audio, 24) == "multimodal transcript"
    content = processor.messages[0]["content"]
    assert content[0]["type"] == "audio" and content[0]["audio"] is audio
    assert content[1] == {"type": "text", "text": instruction}
    assert model.generate_call["max_new_tokens"] == 24
    assert model.generate_call["do_sample"] is False
    assert processor.decoded.tolist() == [[5, 6]]


def test_genai_multimodal_transcriber_passes_audio_tensor():
    calls = {}

    class _FakeVLMPipeline:
        def generate(self, prompt, **kwargs):
            calls.update({"prompt": prompt, **kwargs})
            return types.SimpleNamespace(texts=["genai transcript"])

    transcriber = GenAIMultimodalTranscriber(_FakeVLMPipeline())
    assert transcriber.transcribe(np.zeros(4, dtype=np.float32), 12) == "genai transcript"
    assert calls["prompt"] == DEFAULT_ASR_INSTRUCTION
    assert calls["max_new_tokens"] == 12
    assert calls["audios"][0].get_shape() == [4]


@pytest.mark.parametrize("sampling_rate, length", [(48000, 1600), (44100, 1742), (16000, 4800)])
def test_to_mono_16k_resamples(sampling_rate, length):
    audio = np.sin(np.arange(4800, dtype=np.float64) / 10.0)
    resampled = to_mono_16k(audio, sampling_rate)
    assert resampled.shape == (length,)
    assert resampled.dtype == np.float32


def test_to_mono_16k_downmixes_channels():
    stereo = np.stack([np.ones(96, dtype=np.float64), -np.ones(96, dtype=np.float64)], axis=1)
    mono = to_mono_16k(stereo, 48000)
    assert mono.shape == (32,)
    assert mono.dtype == np.float32
    assert np.allclose(mono, 0.0)  # the two channels cancel out
