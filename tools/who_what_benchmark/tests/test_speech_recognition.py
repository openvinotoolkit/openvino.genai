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
from whowhatbench.whowhat_metrics import WordErrorRate
from whowhatbench.speech_recognition_evaluator import (
    DEFAULT_ASR_INSTRUCTION,
    FunASRGenAITranscriber,
    FunASROptimumTranscriber,
    FunASRSourceTranscriber,
    GenAIMultimodalTranscriber,
    MultimodalTranscriber,
    SpeechRecognitionEvaluator,
)


def _frame(prompts, answers):
    return pd.DataFrame({"prompts": prompts, "answers": answers})


def _csv(tmp_path, name, prompts, answers):
    path = tmp_path / name
    _frame(prompts, answers).to_csv(path, index=False)
    return str(path)


def test_wer_corpus_and_per_utterance():
    gt = _frame(["a", "b"], ["the quick brown fox", "hello world"])
    pred = _frame(["a", "b"], ["the quick brown fox", "hello there world"])
    aggregate, per_prompt = WordErrorRate().evaluate(gt, pred)
    # 1 insertion over 6 reference words -> corpus 1/6; mean-utterance would be 0.25.
    assert per_prompt["WER"] == [0.0, 0.5]
    assert math.isclose(aggregate["WER"], 1 / 6)


def test_wer_identical_is_zero():
    gt = _frame(["a"], ["hello world"])
    assert WordErrorRate().evaluate(gt, _frame(["a"], ["hello world"]))[0]["WER"] == 0.0


def test_wer_empty_reference_counts_insertions():
    gt = _frame(["a", "b"], ["hello", ""])
    pred = _frame(["a", "b"], ["hello", "spurious words"])
    aggregate, per_prompt = WordErrorRate().evaluate(gt, pred)
    assert aggregate["WER"] == 2.0
    assert per_prompt["WER"] == [0.0, 2.0]


def test_wer_empty_data_is_zero():
    aggregate, per_prompt = WordErrorRate().evaluate(_frame([], []), _frame([], []))
    assert aggregate["WER"] == 0.0
    assert per_prompt["WER"] == []


def test_wer_length_mismatch_raises():
    with pytest.raises(ValueError, match="counts differ"):
        WordErrorRate().evaluate(_frame(["a", "b"], ["x", "y"]), _frame(["a"], ["x"]))


def test_score_row_count_mismatch(tmp_path):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a", "b"], ["x", "y"]))
    with pytest.raises(ValueError, match="differ in length"):
        evaluator.score(_csv(tmp_path, "target.csv", ["a"], ["x"]))


def test_score_missing_column(tmp_path):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a"], ["x"]))
    bad = tmp_path / "target.csv"
    pd.DataFrame({"prompts": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        evaluator.score(str(bad))


def test_score_prompt_ids_mismatch(tmp_path):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a", "b"], ["x", "y"]))
    with pytest.raises(ValueError, match="do not match"):
        evaluator.score(_csv(tmp_path, "target.csv", ["a", "c"], ["x", "y"]))


def test_score_end_to_end(tmp_path):
    gt = _csv(tmp_path, "gt.csv", ["a", "b"], ["the quick brown fox", "hello world"])
    target = _csv(tmp_path, "target.csv", ["a", "b"], ["the quick brown fox", "hello there world"])
    evaluator = SpeechRecognitionEvaluator(gt_data=gt)
    per_prompt, aggregate = evaluator.score(target)
    assert per_prompt["similarity"].tolist() == [1.0, 0.5]
    assert per_prompt.columns.tolist() == ["similarity"]
    assert math.isclose(aggregate["similarity"].iloc[0], 5 / 6)


def test_score_clamps_similarity_at_zero(tmp_path):
    evaluator = SpeechRecognitionEvaluator(gt_data=_csv(tmp_path, "gt.csv", ["a"], [""]))
    _, aggregate = evaluator.score(_csv(tmp_path, "target.csv", ["a"], ["spurious words"]))
    assert aggregate["similarity"].iloc[0] == 0.0


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_is_funasr_model_detects_local_source(tmp_path):
    _write_json(tmp_path / "configuration.json", {"framework": "pytorch", "model": {"type": "funasr"}})
    assert model_loaders.is_funasr_model(str(tmp_path))


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "fun_asr"},
        {"model_type": "fun-asr"},
        {"export_model_type": "fun_asr"},
        {"model_type": "", "export_model_type": "FUN_ASR"},
    ],
)
def test_is_funasr_model_detects_local_export(tmp_path, config):
    _write_json(tmp_path / "config.json", config)
    assert model_loaders.is_funasr_model(str(tmp_path))


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"config.json": {"model_type": "gemma4_unified"}},
        {"configuration.json": {"model": {"type": "sensevoice"}}},
        {"configuration.json": {"model": "funasr"}},
    ],
)
def test_is_funasr_model_rejects_other_models(tmp_path, files):
    for name, payload in files.items():
        _write_json(tmp_path / name, payload)
    assert not model_loaders.is_funasr_model(str(tmp_path))


def test_is_funasr_model_ignores_empty_model_id():
    assert not model_loaders.is_funasr_model(None)
    assert not model_loaders.is_funasr_model("")


def test_is_funasr_model_detects_remote_source(monkeypatch, tmp_path):
    import huggingface_hub

    source = tmp_path / "configuration.json"
    _write_json(source, {"model": {"type": "funasr"}})
    requested = []

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        requested.append((repo_id, filename))
        if filename == "configuration.json":
            return str(source)
        raise FileNotFoundError(filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    assert model_loaders.is_funasr_model("FunAudioLLM/Fun-ASR-Nano-2512")
    assert ("FunAudioLLM/Fun-ASR-Nano-2512", "configuration.json") in requested


def test_is_funasr_model_survives_missing_remote_files(monkeypatch):
    import huggingface_hub

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        raise FileNotFoundError(filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    assert not model_loaders.is_funasr_model("google/gemma-4-E4B-it")


class _FakeFunASRAutoModel:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.generate_kwargs = None
        _FakeFunASRAutoModel.instances.append(self)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [{"text": " 你好世界 "}]


@pytest.fixture
def fake_funasr(monkeypatch):
    _FakeFunASRAutoModel.instances = []
    module = types.ModuleType("funasr")
    module.AutoModel = _FakeFunASRAutoModel
    monkeypatch.setitem(sys.modules, "funasr", module)
    return _FakeFunASRAutoModel


class _FakeOVSpeechSeq2Seq:
    model_save_dir = "/export/dir"
    from_pretrained_kwargs = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.from_pretrained_kwargs = {"model_id": model_id, **kwargs}
        return cls()

    def preprocess_input(self, waveform, sampling_rate, **kwargs):
        self.preprocess_call = {"sampling_rate": sampling_rate, **kwargs}
        return {
            "input_features": torch.zeros(1, 2, 3),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "decoder_input_ids": torch.tensor([[1, 2, 3]]),
            "decoder_attention_mask": torch.ones(1, 3, dtype=torch.long),
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
    monkeypatch.setattr(model_loaders, "is_funasr_model", lambda model_id: True)
    model = model_loaders.load_speech_recognition_model(
        "FunAudioLLM/Fun-ASR-Nano-2512", use_hf=True, speech_language="zh"
    )

    assert isinstance(model, FunASRSourceTranscriber)
    assert fake_funasr.instances[0].init_kwargs["model"] == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert fake_funasr.instances[0].init_kwargs["hub"] == "hf"
    assert "trust_remote_code" not in fake_funasr.instances[0].init_kwargs
    assert model.language == "zh"


def test_load_speech_recognition_model_dispatches_funasr_optimum(monkeypatch, fake_optimum):
    monkeypatch.setattr(model_loaders, "is_funasr_model", lambda model_id: True)
    model = model_loaders.load_speech_recognition_model(
        "fun-asr-ov", device="CPU", ov_config={"CACHE_DIR": ""}, speech_language="en"
    )

    assert isinstance(model, FunASROptimumTranscriber)
    assert _FakeOVSpeechSeq2Seq.from_pretrained_kwargs["model_id"] == "fun-asr-ov"
    assert _FakeOVSpeechSeq2Seq.from_pretrained_kwargs["device"] == "CPU"
    assert "trust_remote_code" not in _FakeOVSpeechSeq2Seq.from_pretrained_kwargs
    assert model.preprocess_kwargs == {"language": "en"}
    assert model.tokenizer.location == "fun-asr-ov"


def test_load_speech_recognition_model_dispatches_funasr_genai(monkeypatch, fake_genai):
    monkeypatch.setattr(model_loaders, "is_funasr_model", lambda model_id: True)
    model = model_loaders.load_speech_recognition_model(
        "fun-asr-ov", device="cpu", ov_config={"CACHE_DIR": "cache"}, use_genai=True, speech_language="en"
    )

    assert isinstance(model, FunASRGenAITranscriber)
    assert model.pipeline.models_path == "fun-asr-ov"
    assert model.pipeline.device == "CPU"
    assert model.pipeline.properties == {"CACHE_DIR": "cache"}
    assert model.language == "en"


def test_load_speech_recognition_model_defaults_funasr_to_english(monkeypatch, fake_genai):
    monkeypatch.setattr(model_loaders, "is_funasr_model", lambda model_id: True)
    model = model_loaders.load_speech_recognition_model("fun-asr-ov", use_genai=True, speech_language="")
    assert model.language == "en"


def test_load_funasr_tokenizer_falls_back_to_llm_subfolder(monkeypatch):
    attempts = []

    def from_pretrained(location, **kwargs):
        attempts.append((location, kwargs.get("subfolder")))
        if kwargs.get("subfolder") is None:
            raise OSError("no tokenizer here")
        return _FakeTokenizer(location, **kwargs)

    monkeypatch.setattr(model_loaders, "AutoTokenizer", types.SimpleNamespace(from_pretrained=from_pretrained))
    tokenizer = model_loaders._load_funasr_tokenizer("FunAudioLLM/Fun-ASR-Nano-2512", _FakeOVSpeechSeq2Seq())

    assert tokenizer.kwargs["subfolder"] == model_loaders.FUNASR_TOKENIZER_SUBFOLDER
    assert attempts == [
        ("FunAudioLLM/Fun-ASR-Nano-2512", None),
        ("/export/dir", None),
        ("FunAudioLLM/Fun-ASR-Nano-2512", model_loaders.FUNASR_TOKENIZER_SUBFOLDER),
    ]


def test_load_funasr_tokenizer_falls_back_to_exported_detokenizer(monkeypatch, tmp_path):
    def from_pretrained(location, **kwargs):
        raise OSError("no tokenizer here")

    detokenizer_path = tmp_path / model_loaders.FUNASR_DETOKENIZER_NAME
    detokenizer_path.write_text("<net/>", encoding="utf-8")
    monkeypatch.setattr(model_loaders, "AutoTokenizer", types.SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setattr(
        "whowhatbench.speech_recognition_evaluator.OVDetokenizer", lambda path: ("detokenizer", str(path))
    )

    assert model_loaders._load_funasr_tokenizer(str(tmp_path), _FakeOVSpeechSeq2Seq()) == (
        "detokenizer",
        str(detokenizer_path),
    )


def test_load_funasr_tokenizer_reports_all_attempts(monkeypatch):
    def from_pretrained(location, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(model_loaders, "AutoTokenizer", types.SimpleNamespace(from_pretrained=from_pretrained))
    with pytest.raises(ValueError, match="decoder for FunASR transcripts"):
        model_loaders._load_funasr_tokenizer("FunAudioLLM/Fun-ASR-Nano-2512", _FakeOVSpeechSeq2Seq())


def test_load_speech_recognition_model_dispatches_audio_vlm(monkeypatch):
    loaded = {}

    def fake_load_visual_text_model(model_id, device, ov_config, use_hf, use_genai, **kwargs):
        loaded.update({"model_id": model_id, "use_hf": use_hf, "use_genai": use_genai, "kwargs": kwargs})
        return "vlm-model"

    monkeypatch.setattr(model_loaders, "is_funasr_model", lambda model_id: False)
    monkeypatch.setattr(model_loaders, "load_visual_text_model", fake_load_visual_text_model)
    monkeypatch.setattr(model_loaders, "_load_audio_vlm_processor", lambda model_id: "processor")

    model = model_loaders.load_speech_recognition_model("google/gemma-4-E4B-it", use_hf=True, speech_language="")
    assert isinstance(model, MultimodalTranscriber)
    assert (model.model, model.processor) == ("vlm-model", "processor")
    assert model.instruction == "Transcribe this audio in English."
    # the audio VLM path reuses the visual-text loaders and must not receive ASR-only arguments
    assert loaded["kwargs"] == {"model_type": "visual-text"}

    genai_model = model_loaders.load_speech_recognition_model("gemma-4-ov", use_genai=True, speech_language="English")
    assert isinstance(genai_model, GenAIMultimodalTranscriber)
    assert genai_model.instruction == "Transcribe this audio in English."


def test_funasr_source_transcriber_maps_arguments(fake_funasr):
    transcriber = FunASRSourceTranscriber("FunAudioLLM/Fun-ASR-Nano-2512")
    assert transcriber.transcribe(np.zeros(16, dtype=np.float32), 64) == "你好世界"

    generate_kwargs = fake_funasr.instances[0].generate_kwargs
    assert generate_kwargs["language"] is None  # neutral funasr prompt
    assert generate_kwargs["max_length"] == 64
    assert generate_kwargs["itn"] is True
    assert generate_kwargs["batch_size"] == 1


def test_funasr_source_transcriber_forwards_language(fake_funasr):
    FunASRSourceTranscriber("FunAudioLLM/Fun-ASR-Nano-2512", "zh").transcribe(np.zeros(16, dtype=np.float32), 8)
    assert fake_funasr.instances[0].generate_kwargs["language"] == "zh"


def test_funasr_source_transcriber_requires_funasr(monkeypatch):
    monkeypatch.setitem(sys.modules, "funasr", None)
    with pytest.raises(ModuleNotFoundError, match="pip install funasr"):
        FunASRSourceTranscriber("FunAudioLLM/Fun-ASR-Nano-2512")


def test_funasr_optimum_transcriber_decodes_generated_ids_only():
    model = _FakeOVSpeechSeq2Seq()
    tokenizer = _FakeTokenizer("export")
    transcriber = FunASROptimumTranscriber(model, tokenizer, "en")

    assert transcriber.transcribe(np.zeros(16, dtype=np.float32), 32) == "decoded"
    assert model.preprocess_call == {"sampling_rate": 16000, "language": "en"}
    assert model.generate_call["max_new_tokens"] == 32
    # the 3 prompt ids are dropped, only the generated ids are decoded
    assert tokenizer.decoded.tolist() == [[7, 8]]


def test_funasr_optimum_transcriber_keeps_optimum_language_default():
    model = _FakeOVSpeechSeq2Seq()
    FunASROptimumTranscriber(model, _FakeTokenizer("export")).transcribe(np.zeros(4, dtype=np.float32), 8)
    assert model.preprocess_call == {"sampling_rate": 16000}


def test_funasr_genai_transcriber_omits_unset_language():
    pipeline = _FakeASRPipeline("dir", "CPU")
    assert FunASRGenAITranscriber(pipeline).transcribe(np.zeros(3, dtype=np.float32), 16) == "transcript"
    assert pipeline.generate_call == {"audio": [0.0, 0.0, 0.0], "max_new_tokens": 16}


def test_funasr_genai_transcriber_forwards_language():
    pipeline = _FakeASRPipeline("dir", "CPU")
    FunASRGenAITranscriber(pipeline, "en").transcribe(np.zeros(1, dtype=np.float32), 16)
    assert pipeline.generate_call["language"] == "en"


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
    def __init__(self):
        self.generate_call = None

    def generate(self, **kwargs):
        self.generate_call = kwargs
        return torch.tensor([[1, 2, 5, 6]])


def test_multimodal_transcriber_prompts_and_slices_prompt():
    model, processor = _FakeVLMModel(), _FakeVLMProcessor()
    transcriber = MultimodalTranscriber(model, processor)
    audio = np.zeros(8, dtype=np.float32)

    assert transcriber.transcribe(audio, 24) == "multimodal transcript"
    content = processor.messages[0]["content"]
    assert content[0]["type"] == "audio" and content[0]["audio"] is audio
    assert content[1] == {"type": "text", "text": DEFAULT_ASR_INSTRUCTION}
    assert model.generate_call["max_new_tokens"] == 24
    assert model.generate_call["do_sample"] is False
    assert processor.decoded.tolist() == [[5, 6]]


def test_multimodal_transcriber_uses_language_in_instruction():
    processor = _FakeVLMProcessor()
    MultimodalTranscriber(_FakeVLMModel(), processor, "English").transcribe(np.zeros(2, dtype=np.float32), 8)
    assert processor.messages[0]["content"][1]["text"] == "Transcribe this audio in English."


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
