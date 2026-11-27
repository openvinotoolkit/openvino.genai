# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import functools
import json
import shutil

import numpy as np
import openvino
import openvino_genai as ov_genai
import pytest

from test_asr_pipeline import (
    PipelineType,
    check_perf_metrics,
    check_streamers,
    get_audio_dataset,
    get_model_pipeline_pair_id,
    read_asr_model as read_cpu_asr_model,
    sample_from_dataset,
    streamer_for_test,
)
from utils.asr_utils.qwen3_asr import QWEN3_ASR_MODEL_ID, QWEN3_ASR_SPLIT_EXPORT_SUFFIX
from utils.constants import NPUW_CPU_PROPERTIES, get_ov_cache_converted_models_dir
from utils.ov_genai_pipelines import should_skip_npuw_tests

QWEN3_ASR_NPU_PIPELINE = (QWEN3_ASR_MODEL_ID, PipelineType.ASR, {"device": "NPU", "attention_backend": "SDPA"})
QWEN3_ASR_NPU_MARKS = [
    pytest.mark.qwen3_asr_npu,
    pytest.mark.qwen3_asr_split_export,
    pytest.mark.skipif(**should_skip_npuw_tests()),
]
QWEN3_ASR_NPU_PARAMS = [
    pytest.param(
        QWEN3_ASR_NPU_PIPELINE,
        id=f"{get_model_pipeline_pair_id(QWEN3_ASR_NPU_PIPELINE)}-split-NPUW-CPU",
        marks=QWEN3_ASR_NPU_MARKS,
    )
]


@functools.lru_cache()
def read_asr_model(params, word_timestamps=False, pipeline_type=PipelineType.ASR, attention_backend=None, device="CPU"):
    model_id, path, hf_pipe, cpu_pipe = read_cpu_asr_model(
        params, word_timestamps=word_timestamps, pipeline_type=pipeline_type, attention_backend=attention_backend
    )
    if device == "CPU":
        return model_id, path, hf_pipe, cpu_pipe
    if device != "NPU" or pipeline_type != PipelineType.ASR:
        raise ValueError(f"Unsupported Qwen3-ASR test device/pipeline: {device}, {pipeline_type}")
    properties = dict(NPUW_CPU_PROPERTIES, MAX_PROMPT_LEN=1024, MIN_RESPONSE_LEN=256)
    if attention_backend:
        properties["ATTENTION_BACKEND"] = attention_backend
    return model_id, path, hf_pipe, ov_genai.ASRPipeline(path, "NPU", **properties)


@pytest.fixture
def pipelines_fixture(request):
    model_id, pipeline_type, options = request.param
    path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    model_id, _, hf_pipe, genai_pipe = read_asr_model((model_id, path), pipeline_type=pipeline_type, **options)
    return hf_pipe, genai_pipe, model_id, pipeline_type


@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_npu_perf_metrics(pipelines_fixture, sample_from_dataset):
    check_perf_metrics(pipelines_fixture, sample_from_dataset, has_features_extraction=False)


@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_npu_streamers(pipelines_fixture, sample_from_dataset, streamer_for_test):
    check_streamers(pipelines_fixture, sample_from_dataset, streamer_for_test)


@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
def test_npu_streamer_rejects_multi_chunk_audio(pipelines_fixture):
    _, genai_pipe, _, _ = pipelines_fixture
    audio = np.zeros(70 * 16000, dtype=np.float32)
    with pytest.raises(RuntimeError, match="Qwen3-ASR streaming supports audio up to 60 seconds"):
        genai_pipe.generate(audio, language="English", max_new_tokens=1, streamer=lambda subword: False)


@pytest.mark.parametrize("audio_seconds", [0.2, 0.5], ids=["short-padded", "minimum-length"])
@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
def test_qwen3_asr_direct_vlm_npu_generation_and_capacity(pipelines_fixture, capfd, audio_seconds):
    _, _, model_id, _ = pipelines_fixture
    model_path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, model_path, _, _ = read_asr_model(
        (model_id, model_path), pipeline_type=PipelineType.ASR, device="NPU", attention_backend="SDPA"
    )
    properties = {
        "DEVICE_PROPERTIES": {"NPU": NPUW_CPU_PROPERTIES},
        "MODEL_PROPERTIES": {"text_embeddings": {"NPU_COMPILER_TYPE": "PLUGIN"}},
        "ATTENTION_BACKEND": "SDPA",
        "MAX_PROMPT_LEN": 64,
        "MIN_RESPONSE_LEN": 16,
    }
    asr_pipe = ov_genai.ASRPipeline(model_path, "NPU", **properties)
    vlm_properties = {
        **properties,
        "DEVICE_PROPERTIES": {"NPU": NPUW_CPU_PROPERTIES, "AUTO": {"MULTI_DEVICE_PRIORITIES": "NPU"}},
    }
    vlm_pipe = ov_genai.VLMPipeline(model_path, "NPU", **vlm_properties)
    with open(model_path / "preprocessor_config.json", encoding="utf-8") as file:
        sample_rate = json.load(file)["sampling_rate"]
    audio = np.zeros(int(audio_seconds * sample_rate), dtype=np.float32)
    # ASRPipeline pads short chunks to 0.5 seconds; direct VLMPipeline does not.
    vlm_audio = np.pad(audio, (0, max(0, (sample_rate + 1) // 2 - audio.size)))
    prompt = (
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>"
        "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>"
    )
    config = ov_genai.GenerationConfig(max_new_tokens=8, do_sample=False, apply_chat_template=False)
    asr_config = asr_pipe.get_generation_config()
    config.stop_token_ids = asr_config.stop_token_ids
    config.set_eos_token_id(asr_config.eos_token_id)
    expected = asr_pipe.generate(audio, language="English", max_new_tokens=8, do_sample=False)
    result = vlm_pipe.generate(prompt, audios=[openvino.Tensor(vlm_audio)], generation_config=config)
    assert result.texts == expected.texts
    if audio.size < vlm_audio.size:
        unpadded = vlm_pipe.generate(prompt, audios=[openvino.Tensor(audio)], generation_config=config)
        assert unpadded.perf_metrics.get_num_input_tokens() < result.perf_metrics.get_num_input_tokens()

    available = 64 + 16 - result.perf_metrics.get_num_input_tokens()
    assert available >= 8
    config.max_new_tokens = available
    config.ignore_eos = True
    result = vlm_pipe.generate(prompt, audios=[openvino.Tensor(vlm_audio)], generation_config=config)
    assert result.perf_metrics.get_num_generated_tokens() == available

    config.max_new_tokens = available + 1
    with pytest.raises(RuntimeError, match=rf"requested {available + 1} new tokens.*accommodate only {available}"):
        vlm_pipe.generate(prompt, audios=[openvino.Tensor(vlm_audio)], generation_config=config)

    config.max_new_tokens = (1 << 64) - 1
    config.max_length = (1 << 64) - 1
    config.ignore_eos = False
    config.min_new_tokens = available
    result = vlm_pipe.generate(prompt, audios=[openvino.Tensor(vlm_audio)], generation_config=config)
    assert result.perf_metrics.get_num_generated_tokens() == available
    assert "Qwen3-ASR NPU KV-cache capacity was reached" in capfd.readouterr().err

    config.max_new_tokens = 8
    config.min_new_tokens = 0
    recovered = vlm_pipe.generate(prompt, audios=[openvino.Tensor(vlm_audio)], generation_config=config)
    assert recovered.texts == expected.texts


@pytest.mark.parametrize("pipeline_cls", [ov_genai.ASRPipeline, ov_genai.VLMPipeline], ids=["ASR", "VLM"])
@pytest.mark.parametrize("role", ["audio_encoder", "text_embeddings"])
@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
def test_qwen3_asr_npu_routes_split_components(pipelines_fixture, pipeline_cls, role, tmp_path):
    from openvino import opset13 as ops

    _, _, model_id, _ = pipelines_fixture
    model_path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, model_path, _, _ = read_asr_model(
        (model_id, model_path), pipeline_type=PipelineType.ASR, device="NPU", attention_backend="SDPA"
    )
    copied_model = tmp_path / "model"
    shutil.copytree(model_path, copied_model)
    model_name = "openvino_audio_encoder_model.xml" if role == "audio_encoder" else "openvino_text_embeddings_model.xml"
    component = openvino.Core().read_model(model_path / model_name)
    output = component.get_results()[0]
    if role == "audio_encoder":
        lengths = component.input("aftercnn_lens")
        assert not lengths.get_target_inputs()
        length_sum = ops.reduce_sum(
            ops.convert(lengths, openvino.Type.f32), ops.constant(np.array([0], dtype=np.int64)), False
        )
        zero = ops.multiply(length_sum, ops.constant(np.float32(0)))
        error = "requires independent windows; input 'aftercnn_lens' is used"
    else:
        zero = ops.constant(np.float32(0))
        error = "require a token-wise vocabulary lookup"
    output.input(0).replace_source_output(ops.add(output.input_value(0), zero).output(0))
    component.validate_nodes_and_infer_types()
    openvino.save_model(component, copied_model / model_name, compress_to_fp16=False)

    device_properties = {"NPU": NPUW_CPU_PROPERTIES}
    if pipeline_cls is ov_genai.VLMPipeline:
        device_properties["AUTO"] = {"MULTI_DEVICE_PRIORITIES": "NPU"}
    with pytest.raises(RuntimeError, match=error):
        pipeline_cls(
            copied_model,
            "NPU",
            DEVICE_PROPERTIES=device_properties,
            ATTENTION_BACKEND="SDPA",
            MODEL_PROPERTIES={"text_embeddings": {"NPU_COMPILER_TYPE": "PLUGIN"}},
        )


@pytest.mark.parametrize("role", [None, "audio_encoder", "text_embeddings"], ids=["global", "audio", "text"])
@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
def test_qwen3_asr_npu_rejects_cpu_input_device(pipelines_fixture, role):
    _, _, model_id, _ = pipelines_fixture
    model_path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, model_path, _, _ = read_asr_model(
        (model_id, model_path), pipeline_type=PipelineType.ASR, device="NPU", attention_backend="SDPA"
    )
    auto_priority = "CPU" if role is None else "NPU"
    model_properties = {} if role is None else {role: {"MULTI_DEVICE_PRIORITIES": "CPU"}}
    with pytest.raises(RuntimeError, match="requires MULTI_DEVICE_PRIORITIES=NPU for"):
        ov_genai.ASRPipeline(
            model_path,
            "NPU",
            DEVICE_PROPERTIES={"NPU": NPUW_CPU_PROPERTIES, "AUTO": {"MULTI_DEVICE_PRIORITIES": auto_priority}},
            MODEL_PROPERTIES=model_properties,
            ATTENTION_BACKEND="SDPA",
        )


@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
def test_qwen3_asr_npu_accepts_role_npu_priorities(pipelines_fixture):
    _, _, model_id, _ = pipelines_fixture
    model_path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, model_path, _, _ = read_asr_model(
        (model_id, model_path), pipeline_type=PipelineType.ASR, device="NPU", attention_backend="SDPA"
    )
    pipe = ov_genai.ASRPipeline(
        model_path,
        "NPU",
        DEVICE_PROPERTIES={"NPU": NPUW_CPU_PROPERTIES, "AUTO": {"MULTI_DEVICE_PRIORITIES": "CPU"}},
        MODEL_PROPERTIES={role: {"MULTI_DEVICE_PRIORITIES": "NPU"} for role in ("audio_encoder", "text_embeddings")},
        ATTENTION_BACKEND="SDPA",
        MAX_PROMPT_LEN=64,
        MIN_RESPONSE_LEN=16,
    )
    assert len(pipe.generate(np.zeros(8000, dtype=np.float32), language="English", max_new_tokens=2).texts) == 1


@pytest.mark.parametrize("auto_priority,role_priority", [("CPU", "NPU"), ("NPU", "CPU")], ids=["role-npu", "role-cpu"])
@pytest.mark.parametrize("role", ["audio_encoder", "text_embeddings"], ids=["audio", "text"])
@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
def test_qwen3_asr_direct_vlm_resolves_role_priorities(pipelines_fixture, role, auto_priority, role_priority):
    _, _, model_id, _ = pipelines_fixture
    model_path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, model_path, _, _ = read_asr_model(
        (model_id, model_path), pipeline_type=PipelineType.ASR, device="NPU", attention_backend="SDPA"
    )
    pipe = ov_genai.VLMPipeline(
        model_path,
        "NPU",
        DEVICE_PROPERTIES={"NPU": NPUW_CPU_PROPERTIES, "AUTO": {"MULTI_DEVICE_PRIORITIES": auto_priority}},
        MODEL_PROPERTIES={role: {"MULTI_DEVICE_PRIORITIES": role_priority}},
        ATTENTION_BACKEND="SDPA",
        MAX_PROMPT_LEN=64,
        MIN_RESPONSE_LEN=16,
    )
    prompt = (
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>"
        "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>"
    )
    audio = openvino.Tensor(np.zeros(8000, dtype=np.float32))
    config = ov_genai.GenerationConfig(max_new_tokens=2, do_sample=False, apply_chat_template=False)
    assert len(pipe.generate(prompt, audios=[audio], generation_config=config).texts) == 1


@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
def test_qwen3_asr_direct_vlm_keeps_default_auxiliary_device(pipelines_fixture):
    _, _, model_id, _ = pipelines_fixture
    model_path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, model_path, _, _ = read_asr_model(
        (model_id, model_path), pipeline_type=PipelineType.ASR, device="NPU", attention_backend="SDPA"
    )
    vlm = ov_genai.VLMPipeline(
        model_path,
        "NPU",
        DEVICE_PROPERTIES={"NPU": NPUW_CPU_PROPERTIES},
        MODEL_PROPERTIES={"text_embeddings": {"CPU_DENORMALS_OPTIMIZATION": True}},
        ATTENTION_BACKEND="SDPA",
    )
    audio = openvino.Tensor(np.zeros(8000, dtype=np.float32))
    prompt = (
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>"
        "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>"
    )
    config = ov_genai.GenerationConfig(max_new_tokens=8, do_sample=False, apply_chat_template=False)
    assert len(vlm.generate(prompt, audios=[audio], generation_config=config).texts) == 1


@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
def test_npu_asr_constructors(pipelines_fixture, tmp_path):
    _, _, model_id, pipeline_type = pipelines_fixture
    path = get_ov_cache_converted_models_dir() / model_id.split("/")[-1]
    path = path.with_name(path.name + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, path, _, _ = read_asr_model(
        (model_id, path), pipeline_type=pipeline_type, device="NPU", attention_backend="SDPA"
    )
    blob = tmp_path / "qwen3_asr_decoder.blob"
    audio = np.zeros(3200, dtype=np.float32).tolist()
    generation = {"language": "English", "max_new_tokens": 8}
    exported = ov_genai.ASRPipeline(
        path,
        "NPU",
        **NPUW_CPU_PROPERTIES,
        MAX_PROMPT_LEN=128,
        MIN_RESPONSE_LEN=64,
        BLOB_PATH=str(blob),
        EXPORT_BLOB=True,
    )
    expected = exported.generate(audio, **generation)
    assert blob.is_file()
    del exported
    imported = ov_genai.ASRPipeline(
        path,
        "NPU",
        **NPUW_CPU_PROPERTIES,
        BLOB_PATH=str(blob),
        WEIGHTS_PATH=str(path / "openvino_language_model.bin"),
    )
    actual = imported.generate(audio, **generation)
    assert actual.texts == expected.texts
    assert actual.languages == expected.languages
    default = ov_genai.ASRPipeline(
        path,
        "NPU",
        **NPUW_CPU_PROPERTIES,
        MAX_PROMPT_LEN=128,
        MIN_RESPONSE_LEN=64,
    )
    assert default.generate(audio, **generation).texts == expected.texts
    with pytest.raises(RuntimeError, match="supports only the SDPA attention backend"):
        ov_genai.ASRPipeline(path, "NPU", **NPUW_CPU_PROPERTIES, ATTENTION_BACKEND="PA")


@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
@pytest.mark.parametrize("sample_from_dataset", [{"sample_id": 0}], indirect=True)
def test_npu_max_new_tokens(pipelines_fixture, sample_from_dataset):
    _, genai_pipe, model_id, _ = pipelines_fixture
    audio = sample_from_dataset[:3200]
    expected = genai_pipe.generate(audio, language="English", max_new_tokens=8)
    config = genai_pipe.get_generation_config()
    config.language = "English"
    config.max_new_tokens = 8
    actual = genai_pipe.generate(audio, config)
    assert actual.texts == expected.texts
    assert actual.languages == expected.languages
    with pytest.raises(RuntimeError, match=r"requested .* new tokens.*KV-cache can accommodate only"):
        genai_pipe.generate(audio, language="English", max_new_tokens=1_000_000, ignore_eos=True)
    with pytest.raises(RuntimeError, match="does not support beam search"):
        genai_pipe.generate(audio, language="English", max_new_tokens=8, num_beams=2)

    model_path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, model_path, _, _ = read_asr_model(
        (model_id, model_path), pipeline_type=PipelineType.ASR, device="NPU", attention_backend="SDPA"
    )
    small_pipe = ov_genai.ASRPipeline(
        model_path,
        "NPU",
        **NPUW_CPU_PROPERTIES,
        ATTENTION_BACKEND="SDPA",
        MAX_PROMPT_LEN=64,
        MIN_RESPONSE_LEN=16,
    )
    small_expected = small_pipe.generate(audio, language="English", max_new_tokens=8)
    with open(model_path / "preprocessor_config.json", encoding="utf-8") as file:
        preprocessor = json.load(file)
    with open(model_path / "config.json", encoding="utf-8") as file:
        audio_config = json.load(file)["thinker_config"]["audio_config"]
    frames = max(len(audio), preprocessor["sampling_rate"] // 2) // preprocessor["hop_length"]
    chunk_frames = 2 * audio_config["n_window"]
    full_chunks, tail_frames = divmod(frames, chunk_frames)
    audio_tokens = full_chunks * ((chunk_frames + 7) // 8) + (tail_frames + 7) // 8
    prompt = (
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
        + "<|audio_pad|>" * audio_tokens
        + "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>"
    )
    prompt_len = small_pipe.get_tokenizer().encode(prompt, add_special_tokens=False).input_ids.size
    available = 64 + 16 - prompt_len
    result = small_pipe.generate(audio, language="English", max_new_tokens=available, ignore_eos=True)
    assert result.perf_metrics.get_num_generated_tokens() == available
    with pytest.raises(RuntimeError, match=rf"requested {available + 1} new tokens.*accommodate only {available}"):
        small_pipe.generate(audio, language="English", max_new_tokens=available + 1, ignore_eos=True)
    recovered = small_pipe.generate(audio, language="English", max_new_tokens=8)
    assert recovered.texts == small_expected.texts


@pytest.mark.parametrize("pipelines_fixture", QWEN3_ASR_NPU_PARAMS, indirect=True)
@pytest.mark.parametrize("audio_case", ["short", "full", "partial", "two-full", "boundary"])
def test_npu_qwen3_asr_multi_chunk_audio(pipelines_fixture, audio_case):
    _, genai_pipe, model_id, pipeline_type = pipelines_fixture
    model_path = get_ov_cache_converted_models_dir() / (model_id.split("/")[-1] + QWEN3_ASR_SPLIT_EXPORT_SUFFIX)
    _, model_path, _, cpu_pipe = read_asr_model((model_id, model_path), pipeline_type=pipeline_type)
    with open(model_path / "config.json", encoding="utf-8") as file:
        model_config = json.load(file)
    with open(model_path / "preprocessor_config.json", encoding="utf-8") as file:
        preprocessor = json.load(file)
    if audio_case == "boundary":
        sampling_rate = preprocessor["sampling_rate"]
        audio = np.full(66 * sampling_rate, 0.01, dtype=np.float32)
        audio[62 * sampling_rate : 63 * sampling_rate] = 0.0
        expected_encode_calls = 2
        expected = None
    else:
        chunk_frames = 2 * model_config["thinker_config"]["audio_config"].get("n_window", 100)
        factor = {"short": 0.5, "full": 1.0, "partial": 1.5, "two-full": 2.0}[audio_case]
        samples = int(chunk_frames * preprocessor["hop_length"] * factor)
        reference = np.asarray(get_audio_dataset(False)[0], dtype=np.float32)
        assert reference.size > 0
        repetitions = (samples + reference.size - 1) // reference.size
        audio = np.tile(reference, repetitions)[:samples]
        expected_encode_calls = 1
        expected = cpu_pipe.generate(audio, language="English", max_new_tokens=8, do_sample=False)
    generation = {"language": "English", "max_new_tokens": 8, "do_sample": False}
    results = [genai_pipe.generate(audio, **generation) for _ in range(2)]
    assert results[0].texts == results[1].texts
    assert results[0].languages == results[1].languages
    if expected is not None:
        assert results[0].texts == expected.texts
        assert results[0].languages == expected.languages
    for result in results:
        assert len(result.texts) == 1
        asr_raw_metrics = result.perf_metrics.asr_raw_metrics
        assert len(asr_raw_metrics.encode_inference_durations) == expected_encode_calls
        assert len(asr_raw_metrics.decode_inference_durations) > 0
        assert result.perf_metrics.get_inference_duration().mean > 0
