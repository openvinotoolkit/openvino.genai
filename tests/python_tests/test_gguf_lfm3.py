# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU checkpoint/API checks; independent outputs use the pinned llama.cpp reference manifest."""
import json
import gc
import os
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as genai
import pytest


@pytest.fixture(scope="module")
def bundle():
    path = os.environ.get("OV_LFM3_BUNDLE")
    if not path:
        pytest.skip("Set OV_LFM3_BUNDLE to the pinned LFM3 model directory")
    root = Path(path)
    return root, {
        "INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
        "INFERENCE_NUM_THREADS": 4, "ATTENTION_BACKEND": "SDPA",
    }


@pytest.fixture(scope="module", params=["Audio", "Omni"])
def speech_pipeline(bundle, request):
    root, properties = bundle
    family = request.param
    pipeline = genai.OmniPipeline(str(root / f"LFM3-{family}-2B-Q4_0.gguf"), "CPU",
                                  mmproj_path=str(root / f"mmproj-LFM3-{family}-2B-Q8_0.gguf"), **properties)
    return family, pipeline


def speech_config(return_audio=True, frames=1):
    config = genai.OmniTalkerSpeechConfig()
    config.return_audio = return_audio
    config.audio_chunk_frames = frames
    config.cp_top_k = 1
    config.rng_seed = 7
    return config


def test_speech_streaming_repeat_and_cancel_recovery(speech_pipeline):
    _, pipeline = speech_pipeline
    text = genai.GenerationConfig(max_new_tokens=24, do_sample=False)
    speech = speech_config()
    chunks = []
    result = pipeline.generate("Say hello.", text_config=text, talker_speech_config=speech,
                               speech_streamer=lambda chunk: chunks.append(chunk.data.copy()))
    waveform = result.speech_result.waveforms[0].data.copy()
    assert waveform.shape[0] == 1 and waveform.size > 0 and np.isfinite(waveform).all()
    np.testing.assert_array_equal(np.concatenate(chunks, axis=1), waveform)
    assert waveform.size % 1920 == 0
    repeat = pipeline.generate("Say hello.", text_config=text, talker_speech_config=speech)
    assert repeat.texts == result.texts
    np.testing.assert_array_equal(repeat.speech_result.waveforms[0].data, waveform)
    cancelled = pipeline.generate("Say hello.", text_config=text, talker_speech_config=speech,
                                  speech_streamer=lambda _: genai.StreamingStatus.CANCEL)
    assert not cancelled.speech_result.waveforms
    recovered = pipeline.generate("Say hello.", text_config=text, talker_speech_config=speech)
    assert recovered.texts == result.texts
    np.testing.assert_array_equal(recovered.speech_result.waveforms[0].data, waveform)


def test_audio_chat_and_text_only(speech_pipeline):
    _, pipeline = speech_pipeline
    samples = np.fromfile(Path(__file__).parents[1] / "cpp/data/lfm3/waveform-16000.bin", dtype=np.float32)
    history = genai.ChatHistory([{"role": "user", "content": "<audio> Describe the sound."}])
    config = genai.GenerationConfig(max_new_tokens=8)
    result = pipeline.generate(history, audios=[ov.Tensor(samples)], text_config=config,
                               talker_speech_config=speech_config(False))
    assert result.texts and not result.speech_result.waveforms
    history.append({"role": "assistant", "content": result.texts[0]})
    history.append({"role": "user", "content": "What did you hear?"})
    continuation = pipeline.generate(history, text_config=config, talker_speech_config=speech_config(False))
    assert continuation.texts
    spoken = pipeline.generate("<audio> Describe the sound.", audios=[ov.Tensor(samples)],
                               text_config=genai.GenerationConfig(max_new_tokens=24),
                               talker_speech_config=speech_config())
    assert spoken.speech_result.waveforms


def test_omni_combined_media_and_modality_rejections(speech_pipeline):
    family, pipeline = speech_pipeline
    image = ov.Tensor(np.full((1, 256, 256, 3), 128, dtype=np.uint8))
    video = ov.Tensor(np.full((4, 128, 128, 3), 96, dtype=np.uint8))
    metadata = genai.VideoMetadata()
    metadata.fps = 2.0
    samples = ov.Tensor(np.zeros(16000, dtype=np.float32))
    config = genai.GenerationConfig(max_new_tokens=8)
    if family == "Audio":
        with pytest.raises(RuntimeError, match="does not accept image|accepts text and audio"):
            pipeline.generate("<image>", images=[image], text_config=config)
        with pytest.raises(RuntimeError, match="does not accept video"):
            pipeline.generate("<video>", videos=[video], text_config=config)
    else:
        result = pipeline.generate("<audio> <image> <video> Describe the inputs.", images=[image], videos=[video],
                                   videos_metadata=[metadata], audios=[samples],
                                   text_config=genai.GenerationConfig(max_new_tokens=24),
                                   talker_speech_config=speech_config())
        assert result.texts and result.speech_result.waveforms
        assert np.isfinite(result.speech_result.waveforms[0].data).all()
        with pytest.raises(RuntimeError, match="fps"):
            pipeline.generate("<video>", videos=[video], text_config=config)
    with pytest.raises(RuntimeError, match="mono"):
        pipeline.generate("<audio>", audios=[ov.Tensor(np.zeros((2, 16000), dtype=np.float32))], text_config=config)
    assert pipeline.generate("Hello.", text_config=config, talker_speech_config=speech_config(False)).texts


def test_vl_image_video_chat_and_text_pipeline(bundle):
    root, properties = bundle
    text = genai.LLMPipeline(str(root / "LFM3-1.7B-QAD-Q4_0.gguf"), "CPU", GGUF_READER="FRONTEND", **properties)
    assert text.generate("Hello", genai.GenerationConfig(max_new_tokens=2))
    pipeline = genai.VLMPipeline(str(root / "LFM3-VL-2B-QAD-Q4_0.gguf"), "CPU",
                                 mmproj_path=str(root / "mmproj-LFM3-VL-2B-Q8_0.gguf"), **properties)
    image = ov.Tensor(np.full((1, 256, 320, 3), 160, dtype=np.uint8))
    video = ov.Tensor(np.full((4, 128, 160, 3), 96, dtype=np.uint8))
    metadata = genai.VideoMetadata()
    metadata.fps = 2.0
    config = genai.GenerationConfig(max_new_tokens=2)
    assert pipeline.generate("<image> Describe.", images=[image], generation_config=config).texts
    assert pipeline.generate("<video> Describe.", videos=[video], videos_metadata=[metadata], generation_config=config).texts
    with pytest.raises(RuntimeError):
        pipeline.generate("<audio>", audios=[ov.Tensor(np.zeros(16000, dtype=np.float32))], generation_config=config)


@pytest.mark.parametrize("family", ["VL", "Audio", "Omni"])
def test_saved_ir_directory_and_serialized_models(bundle, family):
    saved_root = os.environ.get("OV_LFM3_SAVED_IR_BUNDLE")
    if not saved_root:
        pytest.skip("Set OV_LFM3_SAVED_IR_BUNDLE to the exported family bundle directory")
    root, properties = bundle
    saved = Path(saved_root) / family
    language = root / ("LFM3-VL-2B-QAD-Q4_0.gguf" if family == "VL" else f"LFM3-{family}-2B-Q4_0.gguf")
    projector = root / f"mmproj-LFM3-{family}-2B-Q8_0.gguf"
    image = ov.Tensor(np.full((1, 256, 256, 3), 128, dtype=np.uint8))
    video = ov.Tensor(np.full((4, 128, 128, 3), 96, dtype=np.uint8))
    clock = genai.VideoMetadata()
    clock.fps = 2.0
    samples = ov.Tensor(np.fromfile(Path(__file__).parents[1] / "cpp/data/lfm3/waveform-16000.bin", dtype=np.float32))
    if family == "VL":
        scenarios = [("<image> Describe.", {"images": [image]}),
                     ("<video> Describe.", {"videos": [video], "videos_metadata": [clock]})]
    elif family == "Audio":
        scenarios = [("<audio> Describe the sound.", {"audios": [samples]})]
    else:
        scenarios = [("<audio> <image> <video> Describe the inputs.",
                      {"audios": [samples], "images": [image], "videos": [video], "videos_metadata": [clock]})]
    config = genai.GenerationConfig(max_new_tokens=8, do_sample=False)
    direct = genai.VLMPipeline(str(language), "CPU", mmproj_path=str(projector), **properties)
    reference_texts = [direct.generate(prompt, generation_config=config, **media).texts for prompt, media in scenarios]
    del direct
    gc.collect()
    directory = genai.VLMPipeline(str(saved), "CPU", **properties)
    assert [directory.generate(prompt, generation_config=config, **media).texts
            for prompt, media in scenarios] == reference_texts
    del directory
    gc.collect()
    models = {}
    for name in ("language", "text_embeddings", "vision_embeddings", "audio_embeddings"):
        xml = saved / f"openvino_{name}_model.xml"
        if xml.exists():
            models[name] = (xml.read_text(), ov.Tensor(np.fromfile(xml.with_suffix(".bin"), dtype=np.uint8)))
    serialized = genai.VLMPipeline(models, genai.Tokenizer(str(saved)), str(saved), "CPU", **properties)
    assert [serialized.generate(prompt, generation_config=config, **media).texts
            for prompt, media in scenarios] == reference_texts
    del serialized, models
    gc.collect()
    if family != "VL":
        prompt, media = scenarios[0]
        text = genai.GenerationConfig(max_new_tokens=24, do_sample=False)
        direct = genai.OmniPipeline(str(language), "CPU", mmproj_path=str(projector), **properties)
        reference = direct.generate(prompt, text_config=text, talker_speech_config=speech_config(), **media)
        assert reference.speech_result.waveforms
        expected = reference.speech_result.waveforms[0].data.copy()
        reference_text = reference.texts
        del direct, reference
        gc.collect()
        directory = genai.OmniPipeline(str(saved), "CPU", **properties)
        replay = directory.generate(prompt, text_config=text, talker_speech_config=speech_config(), **media)
        assert replay.texts == reference_text
        np.testing.assert_array_equal(replay.speech_result.waveforms[0].data, expected)


def test_independent_cpu_reference_and_saved_ir(bundle):
    manifest_path = os.environ.get("OV_LFM3_REFERENCE_MANIFEST")
    if not manifest_path:
        pytest.skip("Set OV_LFM3_REFERENCE_MANIFEST to pinned independent CPU generation scenarios")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["reference_revision"] == "604cc430239c76ff636c443e3400bf552b750421"
    root, properties = bundle
    assert manifest["scenarios"]
    class TokenCollector(genai.StreamerBase):
        def __init__(self):
            super().__init__()
            self.tokens = []

        def write(self, token):
            self.tokens.extend(token if isinstance(token, (list, tuple)) else [token])
            return genai.StreamingStatus.RUNNING

        def end(self):
            pass

    for scenario in manifest["scenarios"]:
        family = scenario["family"]
        model = root / scenario["model"]
        media = {}
        for name in ("images", "videos", "audios"):
            media[name] = [ov.Tensor(np.load(manifest_path.parent / path)) for path in scenario.get(name, [])]
        clocks = []
        for values in scenario.get("videos_metadata", []):
            clock = genai.VideoMetadata()
            clock.fps = values["fps"]
            clock.frames_indices = values.get("frames_indices", [])
            clocks.append(clock)
        history = genai.ChatHistory(scenario.get("history", [{"role": "user", "content": scenario["prompt"]}]))
        if "formatted_prompt" in scenario:
            tokenizer = genai.GGUFTokenizer(str(model))
            formatted = tokenizer.apply_chat_template(history, True)
            assert formatted == (manifest_path.parent / scenario["formatted_prompt"]).read_text()
            if "reference_input_ids" in scenario:
                ids = tokenizer.encode(formatted, add_special_tokens=False).input_ids.data.reshape(-1).tolist()
                assert ids == scenario["reference_input_ids"]
        collector = TokenCollector()
        if family == "text":
            pipeline = genai.LLMPipeline(str(model), "CPU", GGUF_READER="FRONTEND", **properties)
            result = pipeline.generate(history, genai.GenerationConfig(**scenario["generation_config"]), streamer=collector)
        else:
            factory = genai.VLMPipeline if family == "VL" else genai.OmniPipeline
            pipeline = factory(str(model), "CPU", mmproj_path=str(root / scenario["mmproj"]), **properties)
            settings = {"generation_config" if family == "VL" else "text_config": genai.GenerationConfig(**scenario["generation_config"])}
            if family != "VL":
                settings["talker_speech_config"] = speech_config(scenario.get("return_audio", False))
            result = pipeline.generate(history, videos_metadata=clocks, streamer=collector, **media, **settings)
        assert result.texts == scenario["reference_texts"]
        if "reference_generated_ids" in scenario:
            assert collector.tokens == scenario["reference_generated_ids"]
        if "reference_waveform" in scenario:
            expected = np.load(manifest_path.parent / scenario["reference_waveform"])
            np.testing.assert_allclose(result.speech_result.waveforms[0].data, expected,
                                       rtol=scenario["waveform_rtol"], atol=scenario["waveform_atol"])
        if "saved_ir" in scenario:
            if family == "text":
                saved = genai.LLMPipeline(str(manifest_path.parent / scenario["saved_ir"]), "CPU", **properties)
                replay = saved.generate(history, genai.GenerationConfig(**scenario["generation_config"]))
            else:
                saved = factory(str(manifest_path.parent / scenario["saved_ir"]), "CPU", **properties)
                replay = saved.generate(history, videos_metadata=clocks, **media, **settings)
            assert replay.texts == result.texts
            if family not in ("VL", "text") and result.speech_result.waveforms:
                np.testing.assert_array_equal(replay.speech_result.waveforms[0].data, result.speech_result.waveforms[0].data)
