# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GenAI integration regressions: pinned GGUF -> Optimum reference IR -> pipeline comparisons.

See README.md, "GGUF Frontend Tests", for the steps, CI runtime and model-coverage checklist.
"""

import ctypes
import gc
import json
import os
import shutil
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import openvino as ov
import openvino_genai as genai

from test_gguf_generated import llama_reference_tools
from test_vlm_pipeline import (
    VlmModelInfo,
    run_vlm_pipeline as run_vlm_image_test,
    run_vlm_start_chat_vs_chat_history as run_vlm_chat_test,
)

from utils.comparation import compare_generation_results
from utils.constants import extra_generate_kwargs, get_default_llm_properties
from utils.custom_op import get_extension_lib_path
from utils.hugging_face import (
    OVConvertedModelSchema,
    convert_models,
    generation_config_to_hf,
    run_hugging_face,
)
from utils.network import retry_request
from utils.ov_genai_pipelines import (
    GGUF_PIPELINE_TYPES,
    PipelineType,
    convert_decoded_results_to_generation_result,
    create_ov_pipeline,
)


pytestmark = [
    pytest.mark.skipif(sys.platform == "darwin", reason="CVS-168882: sporadic segmentation fault"),
    pytest.mark.skipif(sys.platform == "win32", reason="CVS-174065"),
]

# Pinned GGUFs (1.8 / 25.5 / 4.9 / 47.2 MB) and their source tokenizer/configuration.
# The HF model is reconstructed from the GGUF weights, not independently downloaded weights.
GGUF_EXTENSION_MODEL = (
    "aladar/llama-2-tiny-random-GGUF",
    "8d5321916486e1d33c46b16990e8da6567785769",
    "llama-2-tiny-random.gguf",
    "yujiepan/llama-2-tiny-random",
    "74bb065d6381fdf9fcde5896cb32267ba2c0ee0a",
)

TINY_GGUF_MODELS = [
    pytest.param(GGUF_EXTENSION_MODEL, id="llama"),
    pytest.param(
        (
            "mradermacher/tiny-random-qwen3-GGUF",
            "058102830a84bb976597f4d8462bafe0ea319358",
            "tiny-random-qwen3.f16.gguf",
            "snake7gun/tiny-random-qwen3",
            "add6406ed2f82988fd5194e343572194f71b6f61",
        ),
        id="qwen3",
    ),
    pytest.param(
        (
            "mradermacher/tiny-random-Phi3ForCausalLM-GGUF",
            "ff12230de1587e824688cc6afef2af22dc2b7807",
            "tiny-random-Phi3ForCausalLM.f16.gguf",
            "Xenova/tiny-random-Phi3ForCausalLM",
            "bac53a980e9402513e0a5f8433656b4f845c1b9b",
        ),
        id="phi3",
    ),
    pytest.param(
        (
            "ggml-org/tinygemma3-GGUF",
            "c287502cd9e278dac8eed805c112cce5d0081e0b",
            "tinygemma3-Q8_0.gguf",
            "ngxson/tinygemma3_cifar",
            "da67f59d2195a0095711f1494f8ee5823cf54591",
        ),
        id="gemma3",
    ),
]


TINY_GGUF_VLM_MODELS = {
    "qwen35": ("qwen3.5", "aa8ac4a953c6081e0afaa34bd95cefd9c852ddbf"),
    "qwen35moe": ("qwen3.5-moe", "3061bafacf5a42fc57e620dc54ab19fae19ad594"),
}


def reference_properties():
    return {**get_default_llm_properties(), "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}


@dataclass
class TinyGGUFModel:
    gguf_path: Path
    reference: OVConvertedModelSchema


@pytest.fixture(scope="module", params=TINY_GGUF_MODELS)
def tiny_gguf_model(request, tmp_path_factory):
    gguf_repo, gguf_revision, filename, hf_repo, hf_revision = request.param
    gguf_dir = retry_request(
        lambda: snapshot_download(
            gguf_repo,
            revision=gguf_revision,
            allow_patterns=[filename],
        )
    )
    hf_dir = retry_request(
        lambda: snapshot_download(
            hf_repo,
            revision=hf_revision,
            allow_patterns=["*.json", "tokenizer.model", "*.txt"],
        )
    )
    # Use the source configuration/tokenizer: HF's GGUF loader omits some configuration
    # fields (e.g. Qwen3 head_dim) and does not reconstruct all SentencePiece tokenizers.
    config = AutoConfig.from_pretrained(hf_dir).get_text_config()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_model, loading_info = AutoModelForCausalLM.from_pretrained(
        gguf_dir,
        gguf_file=filename,
        config=config,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    assert not loading_info["missing_keys"], loading_info
    assert not loading_info["unexpected_keys"], loading_info
    assert not loading_info["mismatched_keys"], loading_info
    hf_model.eval()
    hf_model.generation_config.pad_token_id = hf_tokenizer.pad_token_id
    reference_dir = tmp_path_factory.mktemp("gguf_optimum_reference")
    hf_model.save_pretrained(reference_dir)
    hf_tokenizer.save_pretrained(reference_dir)
    del hf_model
    gc.collect()

    opt_model = OVModelForCausalLM.from_pretrained(
        reference_dir,
        export=True,
        load_in_8bit=False,
        ov_config=reference_properties(),
    )
    convert_models(opt_model, hf_tokenizer, reference_dir)
    yield TinyGGUFModel(
        Path(gguf_dir) / filename,
        OVConvertedModelSchema(hf_repo, opt_model, hf_tokenizer, reference_dir),
    )
    del opt_model
    gc.collect()


@pytest.fixture(scope="module", params=(*GGUF_PIPELINE_TYPES, PipelineType.AUTO))
def pipeline_type(request):
    return request.param


@pytest.fixture(scope="module")
def gguf_pipeline(tiny_gguf_model, pipeline_type):
    return create_ov_pipeline(
        tiny_gguf_model.gguf_path,
        pipeline_type=pipeline_type,
        ov_config=reference_properties(),
        gguf_reader="FRONTEND",
    )


def _raise_on_gguf_conversion(_):
    raise RuntimeError("GGUF frontend extension was invoked")


@pytest.mark.parametrize(
    "build_pipeline",
    [
        pytest.param(
            lambda path, properties: genai.LLMPipeline(path, "CPU", **properties),
            id="llm-pipeline-without-tokenizer",
        ),
        pytest.param(
            lambda path, properties: genai.LLMPipeline(path, genai.Tokenizer(path), "CPU", **properties),
            id="llm-pipeline-with-tokenizer",
        ),
        pytest.param(
            lambda path, properties: genai.ContinuousBatchingPipeline(path, genai.SchedulerConfig(), "CPU", properties),
            id="continuous-batching-without-tokenizer",
        ),
        pytest.param(
            lambda path, properties: genai.ContinuousBatchingPipeline(
                path, genai.Tokenizer(path), genai.SchedulerConfig(), "CPU", **properties
            ),
            id="continuous-batching-with-tokenizer",
        ),
    ],
)
@pytest.mark.parametrize(
    "make_extension",
    [
        pytest.param(
            lambda: [get_extension_lib_path("openvino_gguf_test_extension")],
            id="library-extension",
        ),
        pytest.param(
            lambda: [ov.frontend.ConversionExtension("GGML_OP_MUL_MAT", _raise_on_gguf_conversion)],
            id="extension-object",
        ),
    ],
)
def test_gguf_frontend_extensions(gguf_extension_model, build_pipeline, make_extension):
    properties = {"GGUF_READER": "FRONTEND", "extensions": make_extension()}
    with pytest.raises(RuntimeError, match="GGUF frontend extension was invoked"):
        build_pipeline(str(gguf_extension_model), properties)


def test_gguf_frontend_extension_filename_in_working_directory(gguf_extension_model, monkeypatch):
    library = get_extension_lib_path("openvino_gguf_test_extension").resolve()
    monkeypatch.chdir(library.parent)
    with pytest.raises(RuntimeError, match="GGUF frontend extension was invoked"):
        genai.LLMPipeline(str(gguf_extension_model), "CPU", GGUF_READER="FRONTEND", extensions=[Path(library.name)])


def test_gguf_frontend_extension_filename_in_library_search_path(gguf_extension_model, tmp_path):
    library = get_extension_lib_path("openvino_gguf_test_extension").resolve()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(library.parent), env.get("LD_LIBRARY_PATH", "")])
    script = """
import sys
import openvino_genai as genai

try:
    genai.LLMPipeline(sys.argv[1], "CPU", GGUF_READER="FRONTEND", extensions=[sys.argv[2]])
except RuntimeError as error:
    assert "GGUF frontend extension was invoked" in str(error), error
else:
    raise AssertionError("GGUF frontend extension was not invoked")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(gguf_extension_model), library.name],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "build_pipeline",
    [
        pytest.param(
            lambda path, properties: genai.LLMPipeline(path, "CPU", **properties, ATTENTION_BACKEND="SDPA"),
            id="llm-pipeline",
        ),
        pytest.param(
            lambda path, properties: genai.ContinuousBatchingPipeline(path, genai.SchedulerConfig(), "CPU", properties),
            id="continuous-batching-pipeline",
        ),
    ],
)
def test_gguf_frontend_extension_library_outlives_conversion(gguf_extension_model, build_pipeline):
    library = get_extension_lib_path("openvino_gguf_generation_extension").resolve()
    pipeline = build_pipeline(str(gguf_extension_model), {"GGUF_READER": "FRONTEND", "extensions": [library]})
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libdl.dlopen.restype = ctypes.c_void_p
    libdl.dlclose.argtypes = [ctypes.c_void_p]
    handle = libdl.dlopen(os.fsencode(library), os.RTLD_NOLOAD | os.RTLD_LAZY)
    assert handle, "GGUF frontend extension library was unloaded while the pipeline is alive"
    assert libdl.dlclose(handle) == 0
    assert pipeline.get_tokenizer()


@pytest.mark.parametrize(
    "build_pipeline, generate",
    [
        pytest.param(
            lambda path, properties: genai.LLMPipeline(path, "CPU", **properties, ATTENTION_BACKEND="SDPA"),
            lambda pipeline, config: pipeline.generate(
                pipeline.get_tokenizer().encode("Hello").input_ids, config
            ).tokens[0],
            id="llm-pipeline-sdpa",
        ),
        pytest.param(
            lambda path, properties: genai.ContinuousBatchingPipeline(path, genai.SchedulerConfig(), "CPU", properties),
            lambda pipeline, config: pipeline.generate([pipeline.get_tokenizer().encode("Hello").input_ids], [config])[
                0
            ].m_generation_ids[0],
            id="continuous-batching-pipeline-pa",
        ),
    ],
)
def test_gguf_frontend_extensions_generation(gguf_extension_model, build_pipeline, generate):
    config = genai.GenerationConfig(max_new_tokens=3, ignore_eos=True, do_sample=False, apply_chat_template=False)
    outputs = []
    for extensions in ([], [get_extension_lib_path("openvino_gguf_generation_extension")]):
        properties = {**reference_properties(), "GGUF_READER": "FRONTEND", "extensions": extensions}
        pipeline = build_pipeline(str(gguf_extension_model), properties)
        outputs.append(generate(pipeline, config))
    assert len(outputs[0]) == config.max_new_tokens
    assert outputs[1] == outputs[0]


@pytest.fixture(scope="module")
def gguf_extension_model():
    repo, revision, filename, *_ = GGUF_EXTENSION_MODEL
    snapshot = retry_request(lambda: snapshot_download(repo, revision=revision, allow_patterns=[filename]))
    return Path(snapshot) / filename


@pytest.mark.parametrize(
    "prompts",
    [
        ["Why is the Sun yellow?"],
        ["Hello", "table is made of", "Difference between Jupiter and Mars is that"],
    ],
)
def test_tiny_gguf_string_inputs(tiny_gguf_model, gguf_pipeline, pipeline_type, prompts):
    if pipeline_type == PipelineType.STATEFUL and len(prompts) > 1:
        pytest.skip("Native GGUF SDPA attention masks support batch one; PA covers batched generation")
    reference = tiny_gguf_model.reference
    generation_config = genai.GenerationConfig(
        max_new_tokens=16,
        min_new_tokens=16,
        apply_chat_template=False,
    )
    for prompt in prompts:
        assert gguf_pipeline.get_tokenizer().encode(prompt).input_ids.data.tolist() == [
            reference.hf_tokenizer.encode(prompt)
        ]
    expected = run_hugging_face(reference.opt_model, reference.hf_tokenizer, prompts, generation_config)
    actual = gguf_pipeline.generate(prompts, generation_config)
    actual = convert_decoded_results_to_generation_result(
        actual,
        len(prompts),
        generation_config.num_return_sequences,
        generation_config.is_beam_search(),
    )
    # Supply a config per prompt so the shared comparison checks every batch element.
    compare_generation_results(prompts, expected, actual, [generation_config] * len(prompts))


@pytest.mark.parametrize("text", [" hello", "  hello"])
def test_tiny_gguf_detokenization(tiny_gguf_model, gguf_pipeline, text):
    tokenizer = tiny_gguf_model.reference.hf_tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    assert gguf_pipeline.get_tokenizer().decode(tokens) == tokenizer.decode(tokens, skip_special_tokens=True)


def test_tiny_gguf_standalone_tokenizer(tiny_gguf_model):
    tokenizer = genai.GGUFTokenizer(tiny_gguf_model.gguf_path, GGUF_READER="FRONTEND")
    assert isinstance(tokenizer, genai.Tokenizer)
    reference = tiny_gguf_model.reference.hf_tokenizer
    prompt = "Why is the Sun yellow?"
    tokens = reference.encode(prompt)
    assert tokenizer.encode(prompt).input_ids.data.tolist() == [tokens]
    assert tokenizer.decode(tokens) == reference.decode(tokens, skip_special_tokens=True)


@pytest.mark.parametrize(
    "prompts",
    [
        ["Why is the Sun yellow?"],
        ["Hello", "table is made of", "Difference between Jupiter and Mars is that"],
    ],
)
def test_tiny_gguf_beam_search(tiny_gguf_model, gguf_pipeline, pipeline_type, prompts):
    if pipeline_type == PipelineType.STATEFUL:
        pytest.skip("Native GGUF SDPA attention masks support batch one; PA covers beam expansion")
    reference = tiny_gguf_model.reference
    inputs = reference.hf_tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
    config = genai.GenerationConfig(max_new_tokens=16, min_new_tokens=16, num_beams=3, num_return_sequences=2)
    hf_config = generation_config_to_hf(reference.opt_model.generation_config, config)
    expected = reference.opt_model.generate(
        **inputs,
        generation_config=hf_config,
        **extra_generate_kwargs(hf_config),
    )
    actual = gguf_pipeline.generate(
        genai.TokenizedInputs(
            ov.Tensor(inputs.input_ids.numpy()),
            ov.Tensor(inputs.attention_mask.numpy()),
        ),
        config,
    )
    # Random fixtures can generate reserved tokens whose display policy differs across tokenizers.
    # Compare token IDs and beam scores directly, including every returned sequence in the batch.
    assert actual.tokens == expected.sequences[:, inputs.input_ids.shape[1] :].tolist()
    np.testing.assert_allclose(actual.scores, expected.sequences_scores.numpy(), atol=0.02, rtol=0)


@pytest.mark.parametrize("with_attention_mask", [False, True])
def test_tiny_gguf_encoded_inputs(tiny_gguf_model, gguf_pipeline, with_attention_mask):
    reference = tiny_gguf_model.reference
    input_ids = np.array([[1, 4, 42]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    generation_config = genai.GenerationConfig(max_new_tokens=16, min_new_tokens=16)
    hf_config = generation_config_to_hf(reference.opt_model.generation_config, generation_config)
    hf_inputs = {"input_ids": torch.from_numpy(input_ids)}
    ov_inputs = ov.Tensor(input_ids)
    if with_attention_mask:
        hf_inputs["attention_mask"] = torch.from_numpy(attention_mask)
        ov_inputs = genai.TokenizedInputs(ov_inputs, ov.Tensor(attention_mask))
    expected = (
        reference.opt_model.generate(
            **hf_inputs,
            generation_config=hf_config,
            **extra_generate_kwargs(hf_config),
        )
        .sequences[:, input_ids.shape[1] :]
        .tolist()
    )
    assert gguf_pipeline.generate(ov_inputs, generation_config).tokens == expected


def test_tiny_gguf_streaming(tiny_gguf_model, gguf_pipeline):
    prompt = "Why is the Sun yellow?"
    generation_config = genai.GenerationConfig(max_new_tokens=16, min_new_tokens=16, apply_chat_template=False)
    reference = tiny_gguf_model.reference
    expected = run_hugging_face(reference.opt_model, reference.hf_tokenizer, [prompt], generation_config)
    chunks = []
    actual = gguf_pipeline.generate(prompt, generation_config, streamer=lambda text: chunks.append(text))
    assert actual == expected[0].m_generation_ids[0]
    assert "".join(chunks) == actual


def test_tiny_gguf_saved_ir(tiny_gguf_model, pipeline_type, tmp_path):
    # Serialization writes beside the GGUF; never modify the shared Hub cache.
    gguf_path = tmp_path / tiny_gguf_model.gguf_path.name
    shutil.copyfile(tiny_gguf_model.gguf_path, gguf_path)
    pipeline = create_ov_pipeline(
        gguf_path,
        pipeline_type=pipeline_type,
        ov_config=reference_properties(),
        gguf_reader="FRONTEND",
        enable_save_ov_model=True,
    )
    prompt = "table is made of"
    config = genai.GenerationConfig(max_new_tokens=16, apply_chat_template=False, ignore_eos=True)
    expected = pipeline.generate(prompt, config)
    del pipeline
    reloaded = create_ov_pipeline(tmp_path, pipeline_type=pipeline_type, ov_config=reference_properties())
    assert reloaded.generate(prompt, config) == expected


@pytest.fixture(scope="module")
def tiny_gguf_vlm_model(request, llama_reference_tools, tmp_path_factory):
    from test_gguf_generated import run_reference_command

    family = getattr(request, "param", "qwen35")
    name, revision = TINY_GGUF_VLM_MODELS[family]
    source = Path(
        retry_request(
            lambda: snapshot_download("optimum-intel-internal-testing/tiny-random-" + name, revision=revision)
        )
    )
    converter = llama_reference_tools[0].parent.parent / "_deps" / "llama_cpp-src" / "convert_hf_to_gguf.py"
    directory = Path.home() / "gguf_models" / "genai_vlm_tests" / family
    directory.mkdir(parents=True, exist_ok=True)
    fixture_source = tmp_path_factory.mktemp(family)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    # The production mRoPE default exceeds the tiny model's rotary width.
    config["text_config"]["rope_parameters"]["mrope_section"] = [1, 1, 0]
    for path in source.iterdir():
        if path.name != "config.json":
            shutil.copy2(path, fixture_source / path.name)
    (fixture_source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    language, projector = directory / "model.gguf", directory / "mmproj.gguf"
    converter_env = os.environ.copy()
    converter_env["PYTHONPATH"] = os.pathsep.join([str(converter.parent / "gguf-py"), str(converter.parent)])
    for output, options in [(language, ["--no-mtp"]), (projector, ["--mmproj"])]:
        command = [
            sys.executable,
            str(converter),
            str(fixture_source),
            "--outfile",
            str(output),
            "--outtype",
            "f32",
            *options,
        ]
        run_reference_command(command, timeout=300, env=converter_env)
    return language, projector


@pytest.fixture
def gguf_vlm_pipelines(tiny_gguf_vlm_model):
    language, projector = tiny_gguf_vlm_model
    name, _ = TINY_GGUF_VLM_MODELS[language.parent.name]
    model_id = "optimum-intel-internal-testing/tiny-random-" + name
    pipelines = []
    for backend in ["SDPA", "PA"]:
        pipeline = genai.VLMPipeline(
            language, "CPU", mmproj_path=str(projector), ATTENTION_BACKEND=backend, **reference_properties()
        )
        pipelines.append(VlmModelInfo(model_id, backend, lambda idx: "", lambda idx: "", 24, pipeline, False))
    return pipelines


@pytest.mark.parametrize("tiny_gguf_vlm_model", list(TINY_GGUF_VLM_MODELS), indirect=True)
@pytest.mark.parametrize("media", ["image", "video"])
@pytest.mark.parametrize("backend", ["SDPA", "PA"])
def test_gguf_vlm_projector_layouts(tiny_gguf_vlm_model, backend, media):
    language, projector = tiny_gguf_vlm_model
    pipeline = genai.VLMPipeline(
        language, "CPU", mmproj_path=str(projector), ATTENTION_BACKEND=backend, **reference_properties()
    )
    image = np.random.default_rng(0).integers(0, 256, (1, 48, 64, 3), dtype=np.uint8)
    inputs = {"images": [ov.Tensor(image)]}
    if media == "video":
        metadata = genai.VideoMetadata()
        metadata.fps = 2.0
        inputs = {"videos": [ov.Tensor(np.repeat(image, 4, axis=0))], "videos_metadata": [metadata]}
    options = {"max_new_tokens": 2, "do_sample": False, "ignore_eos": True}
    result = pipeline.generate("Describe.", **inputs, **options)
    assert result.perf_metrics.get_num_generated_tokens() == 2
    assert result.perf_metrics.get_num_input_tokens() > 0
    assert pipeline.generate("Describe.", **inputs, **options).texts == result.texts


def test_gguf_vlm_attention_backends_and_images(gguf_vlm_pipelines):
    image = ov.Tensor(np.random.default_rng(0).integers(0, 256, (1, 48, 64, 3), dtype=np.uint8))
    config = genai.GenerationConfig(max_new_tokens=8, do_sample=False, ignore_eos=True)
    for images in [[image], [image, image]]:
        results = [run_vlm_image_test(info, images, generation_config=config) for info in gguf_vlm_pipelines]
        assert results[0].texts == results[1].texts
        assert [run_vlm_image_test(info, images, generation_config=config).texts for info in gguf_vlm_pipelines] == [
            r.texts for r in results
        ]


@pytest.mark.parametrize("reader", ["LEGACY", "INVALID"])
def test_gguf_vlm_rejects_incompatible_reader(tiny_gguf_vlm_model, reader):
    language, projector = tiny_gguf_vlm_model
    with pytest.raises(RuntimeError, match="GGUF.*reader|GGUF_READER"):
        genai.VLMPipeline(language, "CPU", mmproj_path=str(projector), GGUF_READER=reader)


def test_gguf_vlm_attention_backends_and_chat(gguf_vlm_pipelines):
    image = ov.Tensor(np.random.default_rng(0).integers(0, 256, (1, 48, 64, 3), dtype=np.uint8))
    config = genai.GenerationConfig(max_new_tokens=8, do_sample=False, ignore_eos=True)
    answers = [run_vlm_chat_test(info, [[image], [image]], generation_config=config) for info in gguf_vlm_pipelines]
    assert answers[0] == answers[1]


def test_gguf_vlm_requires_projector(tiny_gguf_vlm_model):
    language, _ = tiny_gguf_vlm_model
    with pytest.raises(RuntimeError, match="mmproj_path"):
        genai.VLMPipeline(language, "CPU")


@pytest.mark.parametrize("option", ["prompt_lookup", "adapters_sdpa", "adapters_pa"])
def test_gguf_vlm_language_properties(tiny_gguf_vlm_model, option):
    language, projector = tiny_gguf_vlm_model
    properties = reference_properties()
    properties["ATTENTION_BACKEND"] = "SDPA" if option == "adapters_sdpa" else "PA"
    baseline = genai.VLMPipeline(language, "CPU", mmproj_path=str(projector), **properties)
    properties["adapters"] = genai.AdapterConfig()
    if option == "prompt_lookup":
        properties["prompt_lookup"] = True
    pipe = genai.VLMPipeline(language, "CPU", mmproj_path=str(projector), **properties)
    defaults = pipe.get_generation_config()
    assert defaults.eos_token_id == baseline.get_generation_config().eos_token_id
    assert defaults.stop_token_ids == baseline.get_generation_config().stop_token_ids
    assert defaults.adapters is not None
    image = ov.Tensor(np.random.default_rng(0).integers(0, 256, (1, 48, 64, 3), dtype=np.uint8))
    config = {"max_new_tokens": 8, "do_sample": False, "ignore_eos": True}
    if option == "prompt_lookup":
        config["num_assistant_tokens"] = 3
        config["max_ngram_size"] = 3
    assert (
        pipe.generate("Describe the image.", images=[image], **config).texts
        == baseline.generate("Describe the image.", images=[image], **config).texts
    )


@pytest.mark.parametrize("prompt_lookup", [False, True])
@pytest.mark.parametrize("top_level", ["absent", "empty", "active"])
def test_gguf_vlm_language_adapter_overlay(tiny_gguf_vlm_model, tmp_path, prompt_lookup, top_level):
    from safetensors.numpy import save_file

    language, projector = tiny_gguf_vlm_model
    frontend = ov.frontend.FrontEndManager().load_by_framework("gguf")
    model = frontend.convert(frontend.load(str(language)))
    head = next(
        op for op in model.get_ops() if op.get_type_name() == "MatMul" and "result_output" in op.get_friendly_name()
    )
    vocab_size, hidden_size = head.input(1).get_shape()
    weights = np.zeros((vocab_size, 1), dtype=np.float32)
    weights[1000] = 1000
    weights[1001] = -1000
    adapter_path = tmp_path / "language_adapter.safetensors"
    save_file(
        {
            "base_model.model.logits.lora_A.weight": np.ones((1, hidden_size), dtype=np.float32),
            "base_model.model.logits.lora_B.weight": weights,
        },
        str(adapter_path),
    )
    adapter = genai.AdapterConfig(genai.Adapter(str(adapter_path)), 1.0, genai.AdapterConfig.Mode.MODE_DYNAMIC)
    empty_adapter = genai.AdapterConfig()
    properties = {
        **reference_properties(),
        "ATTENTION_BACKEND": "PA" if prompt_lookup else "SDPA",
        "prompt_lookup": prompt_lookup,
        "mmproj_path": str(projector),
    }
    baseline = genai.VLMPipeline(language, "CPU", **properties)
    active = genai.VLMPipeline(language, "CPU", adapters=adapter, **properties)
    role_adapter = empty_adapter if top_level == "active" else adapter
    if top_level != "absent":
        properties["adapters"] = adapter if top_level == "active" else empty_adapter
    properties["MODEL_PROPERTIES"] = {"language_model": {"adapters": role_adapter}}
    pipe = genai.VLMPipeline(language, "CPU", **properties)
    defaults = pipe.get_generation_config()
    assert defaults.adapters is not None
    assert len(defaults.adapters.get_adapters()) == (0 if top_level == "active" else 1)

    image = ov.Tensor(np.zeros((1, 48, 64, 3), dtype=np.uint8))
    config = {"max_new_tokens": 4, "do_sample": False, "ignore_eos": True}
    if prompt_lookup:
        config.update(num_assistant_tokens=3, max_ngram_size=3)
    without_adapter = baseline.generate("Describe.", images=[image], **config).texts
    with_adapter = active.generate("Describe.", images=[image], **config).texts
    assert with_adapter != without_adapter
    assert pipe.generate("Describe.", images=[image], **config).texts == (
        without_adapter if top_level == "active" else with_adapter
    )
    if top_level != "active":
        assert pipe.generate("Describe.", images=[image], adapters=empty_adapter, **config).texts == without_adapter
        assert pipe.generate("Describe.", images=[image], **config).texts == with_adapter
