# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GenAI integration regressions: pinned GGUF -> Optimum reference IR -> pipeline comparisons.

See README.md, "GGUF Frontend Tests", for the steps, CI runtime and model-coverage checklist.
"""

import gc
import json
import os
import shutil
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

from utils.comparation import compare_generation_results
from utils.constants import extra_generate_kwargs, get_default_llm_properties
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
TINY_GGUF_MODELS = [
    pytest.param(
        (
            "aladar/llama-2-tiny-random-GGUF",
            "8d5321916486e1d33c46b16990e8da6567785769",
            "llama-2-tiny-random.gguf",
            "yujiepan/llama-2-tiny-random",
            "74bb065d6381fdf9fcde5896cb32267ba2c0ee0a",
        ),
        id="llama",
    ),
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
    "llava": ("llava", "47d46b90eb00e2190d510b59ccd0b649e590d5be"),
    "internvl2": ("internvl2", "e86ef3972133e0a0a77c337c3bde74ad9726f2c2"),
    "gemma3": ("gemma3", "0cc4c5a822bbc68aeb312fd789046463ca301d14"),
    "gemma4": ("gemma4", "422d12880b72b92b1547ac7ce2e1a4865f6bec6c"),
    "gemma4-unified": ("gemma4-unified-it", "becd0eb532fcbccb96576e32eab0dd304e4e4c47"),
    "qwen35": ("qwen3.5", "aa8ac4a953c6081e0afaa34bd95cefd9c852ddbf"),
    "qwen35moe": ("qwen3.5-moe", "3061bafacf5a42fc57e620dc54ab19fae19ad594"),
    "muse-glimmer": ("muse-glimmer", "cb70af191bbbf704a47b7d24dd5287d97dd60103"),
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

    family = getattr(request, "param", "gemma3")
    name, revision = TINY_GGUF_VLM_MODELS[family]
    repo = "optimum-intel-internal-testing/tiny-random-" + name
    source = Path(retry_request(lambda: snapshot_download(repo, revision=revision)))
    build = llama_reference_tools[0].parent.parent
    converter = build / "_deps" / "llama_cpp-src" / "convert_hf_to_gguf.py"
    directory = Path.home() / "gguf_models" / "genai_vlm_tests" / family
    directory.mkdir(parents=True, exist_ok=True)
    if family.startswith("qwen35") or family == "gemma4-unified":
        fixture_source = tmp_path_factory.mktemp(family)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if family.startswith("qwen35"):
            # The production mRoPE default exceeds the tiny model's rotary width.
            config["text_config"]["rope_parameters"]["mrope_section"] = [1, 1, 0]
        else:
            config["text_config"]["expert_intermediate_size"] = None
        for path in source.iterdir():
            if path.name != "config.json":
                shutil.copy2(path, fixture_source / path.name)
        (fixture_source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        source = fixture_source
    language, projector = directory / "model.gguf", directory / "mmproj.gguf"
    language_options = ["--no-mtp"] if family.startswith("qwen35") else []
    converter_env = os.environ.copy()
    converter_env["PYTHONPATH"] = os.pathsep.join([str(converter.parent / "gguf-py"), str(converter.parent)])
    for output, options in [(language, language_options), (projector, ["--mmproj"])]:
        if family == "llava" and output == projector:
            from safetensors.torch import load_file, save_file
            from transformers import CLIPVisionConfig

            fixture_source = tmp_path_factory.mktemp("llava_vision")
            config = json.loads((source / "config.json").read_text(encoding="utf-8"))
            vision_config = CLIPVisionConfig(**config["vision_config"]).to_dict()
            vision_config["vision_feature_layer"] = config["vision_feature_layer"]
            (fixture_source / "config.json").write_text(json.dumps(vision_config), encoding="utf-8")
            weights = load_file(source / "model.safetensors")
            vision_prefix = "vision_tower."
            save_file(
                {
                    name.removeprefix(vision_prefix): weight
                    for name, weight in weights.items()
                    if name.startswith(vision_prefix)
                },
                fixture_source / "model.safetensors",
            )
            projector_weights = fixture_source / "projector.bin"
            torch.save(
                {name: weight for name, weight in weights.items() if name.startswith("multi_modal_projector.")},
                projector_weights,
            )
            del weights
            legacy_converter = converter.parent / "tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py"
            run_reference_command(
                [
                    sys.executable,
                    str(legacy_converter),
                    "--model-dir",
                    str(fixture_source),
                    "--clip-model-is-vision",
                    "--llava-projector",
                    str(projector_weights),
                    "--output-dir",
                    str(directory),
                    "--use-f32",
                ],
                timeout=300,
                env=converter_env,
            )
            (directory / "mmproj-model-f32.gguf").replace(projector)
            continue
        command = [sys.executable, str(converter), str(source), "--outfile", str(output), "--outtype", "f32", *options]
        if family == "internvl2" and output == projector:
            # Tiny InternVL has two vision blocks, but its MLP uses stage 3.
            wrapper = tmp_path_factory.mktemp("internvl_converter") / "convert.py"
            wrapper.write_text(
                "import gguf\n"
                "import runpy\n"
                "import sys\n\n"
                "original = gguf.get_tensor_name_map\n"
                "def tensor_name_map(arch, n_blocks):\n"
                "    if arch == gguf.MODEL_ARCH.MMPROJ:\n"
                "        n_blocks = max(n_blocks, 4)\n"
                "    return original(arch, n_blocks)\n"
                "gguf.get_tensor_name_map = tensor_name_map\n"
                "converter = sys.argv.pop(1)\n"
                "sys.argv[0] = converter\n"
                "runpy.run_path(converter, run_name='__main__')\n",
                encoding="utf-8",
            )
            command.insert(1, str(wrapper))
        run_reference_command(command, timeout=300, env=converter_env)
        if family == "llava" and output == language:
            from gguf import GGUFReader, TokenType

            tokenizer = AutoTokenizer.from_pretrained(source)
            assert tokenizer.unk_token_id is not None
            reader = GGUFReader(language, mode="r+")
            assert reader.fields["tokenizer.ggml.unknown_token_id"].contents() == tokenizer.unk_token_id
            assert reader.fields["tokenizer.ggml.tokens"].contents(tokenizer.unk_token_id) == tokenizer.unk_token
            # The pinned converter marks the source UNK as CONTROL.
            types = reader.fields["tokenizer.ggml.token_type"]
            types.parts[types.data[tokenizer.unk_token_id]][0] = int(TokenType.UNKNOWN)
            reader.data.flush()
    return language, projector


@pytest.mark.parametrize(
    "tiny_gguf_vlm_model,media",
    [
        pytest.param(family, media, id=f"{family}-{media}")
        for family in TINY_GGUF_VLM_MODELS
        if family not in {"gemma3", "llava", "internvl2"}
        for media in (["image", "video", "audio"] if family.startswith("gemma4") else ["image", "video"])
        if (family, media) != ("muse-glimmer", "video")
    ]
    + [
        pytest.param(
            "muse-glimmer",
            "video",
            id="muse-glimmer-video",
            marks=pytest.mark.xfail(
                reason="Muse Glimmer video processing does not honor do_sample_frames=False.",
                raises=RuntimeError,
                strict=True,
            ),
        )
    ],
    indirect=["tiny_gguf_vlm_model"],
)
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
    elif media == "audio":
        inputs = {"audios": [ov.Tensor(np.zeros(16000, dtype=np.float32))]}
    options = {"max_new_tokens": 2, "do_sample": False, "ignore_eos": True}
    result = pipeline.generate("Describe.", **inputs, **options)
    assert result.perf_metrics.get_num_generated_tokens() == 2
    assert result.perf_metrics.get_num_input_tokens() > 0
    assert pipeline.generate("Describe.", **inputs, **options).texts == result.texts


def test_gguf_vlm_attention_backends_and_images(tiny_gguf_vlm_model):
    language, projector = tiny_gguf_vlm_model
    properties = reference_properties()
    pipelines = [
        genai.VLMPipeline(language, "CPU", mmproj_path=str(projector), ATTENTION_BACKEND=backend, **properties)
        for backend in ["SDPA", "PA"]
    ]
    image = ov.Tensor(np.random.default_rng(0).integers(0, 256, (1, 48, 64, 3), dtype=np.uint8))
    config = {"max_new_tokens": 8, "do_sample": False, "ignore_eos": True}
    for images in [[image], [image, image]]:
        results = [pipe.generate("Describe the image.", images=images, **config) for pipe in pipelines]
        assert results[0].texts == results[1].texts
        for pipe, result in zip(pipelines, results):
            assert pipe.generate("Describe the image.", images=images, **config).texts == result.texts


@pytest.mark.parametrize("tiny_gguf_vlm_model", ["llava", "internvl2"], indirect=True)
def test_gguf_vlm_clip_attention_backends_and_images(tiny_gguf_vlm_model):
    language, projector = tiny_gguf_vlm_model
    pipelines = [
        genai.VLMPipeline(
            language,
            "CPU",
            mmproj_path=str(projector),
            GGUF_READER="FRONTEND",
            ATTENTION_BACKEND=backend,
            **reference_properties(),
        )
        for backend in ["SDPA", "PA"]
    ]
    pixels = np.random.default_rng(0).integers(0, 256, (1, 24, 24, 3), dtype=np.uint8)
    images = [ov.Tensor(pixels), ov.Tensor(np.flip(pixels, axis=2).copy())]
    options = {"max_new_tokens": 4, "do_sample": False, "ignore_eos": True, "apply_chat_template": False}
    previous_input_count = 0
    for media in [images[:1], images]:
        prompt = "Describe the images."
        if language.parent.name == "llava":
            prompt = "USER: " + "<image>\n" * len(media) + "Describe the images. ASSISTANT:"
        results = [pipe.generate(prompt, images=media, **options) for pipe in pipelines]
        assert results[0].texts == results[1].texts
        input_count = results[0].perf_metrics.get_num_input_tokens()
        assert input_count == results[1].perf_metrics.get_num_input_tokens()
        assert input_count > previous_input_count
        previous_input_count = input_count
        for pipe, result in zip(pipelines, results):
            assert result.perf_metrics.get_num_generated_tokens() == 4
            repeated = pipe.generate(prompt, images=media, **options)
            assert repeated.texts == result.texts
            assert repeated.perf_metrics.get_num_input_tokens() == input_count


@pytest.mark.parametrize("tiny_gguf_vlm_model", ["llava"], indirect=True)
def test_gguf_llava_preformatted_prompt(tiny_gguf_vlm_model):
    language, projector = tiny_gguf_vlm_model
    pipelines = [
        genai.VLMPipeline(
            language,
            "CPU",
            mmproj_path=str(projector),
            GGUF_READER="FRONTEND",
            ATTENTION_BACKEND=backend,
            **reference_properties(),
        )
        for backend in ["SDPA", "PA"]
    ]
    name, revision = TINY_GGUF_VLM_MODELS["llava"]
    source = retry_request(
        lambda: snapshot_download(
            "optimum-intel-internal-testing/tiny-random-" + name,
            revision=revision,
            allow_patterns=["*.json", "tokenizer.model"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    prefix, suffix = "USER: ", "\nDescribe. ASSISTANT:"
    expected_count = (
        len(tokenizer.encode(prefix, add_special_tokens=True))
        + config["image_seq_length"]
        + len(tokenizer.encode(suffix, add_special_tokens=False))
    )
    image = ov.Tensor(np.random.default_rng(0).integers(0, 256, (1, 24, 24, 3), dtype=np.uint8))
    prompt = prefix + "<image>Describe. ASSISTANT:"
    options = {"max_new_tokens": 4, "do_sample": False, "ignore_eos": True, "apply_chat_template": False}
    results = [pipe.generate(prompt, images=[image], **options) for pipe in pipelines]
    assert results[0].texts == results[1].texts
    for pipe, result in zip(pipelines, results):
        assert result.perf_metrics.get_num_input_tokens() == expected_count
        assert result.perf_metrics.get_num_generated_tokens() == 4
        repeated = pipe.generate(prompt, images=[image], **options)
        assert repeated.texts == result.texts
        assert repeated.perf_metrics.get_num_input_tokens() == expected_count


@pytest.mark.parametrize("tiny_gguf_vlm_model", ["internvl2"], indirect=True)
def test_gguf_internvl_missing_eos_metadata(tiny_gguf_vlm_model):
    from gguf import GGUFReader, GGUFWriter

    language, projector = tiny_gguf_vlm_model
    reader = GGUFReader(language)
    without_eos = language.with_name("model-without-eos.gguf")
    writer = GGUFWriter(without_eos, reader.fields["general.architecture"].contents(), endianess=reader.endianess)
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name in {"general.architecture", "tokenizer.ggml.eos_token_id"}:
            continue
        writer.add_key_value(name, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert "tokenizer.ggml.eos_token_id" not in GGUFReader(without_eos).fields
    pipeline = genai.VLMPipeline(
        without_eos,
        "CPU",
        mmproj_path=str(projector),
        GGUF_READER="FRONTEND",
        ATTENTION_BACKEND="SDPA",
        **reference_properties(),
    )
    tokenizer = pipeline.get_tokenizer()
    eos_id = tokenizer.get_eos_token_id()
    assert eos_id in pipeline.get_generation_config().stop_token_ids
    assert tokenizer.get_eos_token() == reader.fields["tokenizer.ggml.tokens"].contents(eos_id)
    assert {151643, 151645}.issubset(pipeline.get_generation_config().stop_token_ids)
    image = ov.Tensor(np.random.default_rng(0).integers(0, 256, (1, 24, 24, 3), dtype=np.uint8))
    result = pipeline.generate("Describe.", images=[image], max_new_tokens=2, do_sample=False, ignore_eos=True)
    assert result.perf_metrics.get_num_input_tokens() > 49
    assert result.perf_metrics.get_num_generated_tokens() == 2


@pytest.mark.parametrize("reader", ["LEGACY", "INVALID"])
def test_gguf_vlm_rejects_incompatible_reader(tiny_gguf_vlm_model, reader):
    language, projector = tiny_gguf_vlm_model
    with pytest.raises(RuntimeError, match="GGUF.*reader|GGUF_READER"):
        genai.VLMPipeline(language, "CPU", mmproj_path=str(projector), GGUF_READER=reader)


@pytest.mark.parametrize("chat_api", ["start_chat", "ChatHistory"])
def test_gguf_vlm_attention_backends_and_chat(tiny_gguf_vlm_model, chat_api):
    language, projector = tiny_gguf_vlm_model
    pipelines = [
        genai.VLMPipeline(
            language, "CPU", mmproj_path=str(projector), ATTENTION_BACKEND=backend, **reference_properties()
        )
        for backend in ["SDPA", "PA"]
    ]
    image = ov.Tensor(np.random.default_rng(0).integers(0, 256, (1, 48, 64, 3), dtype=np.uint8))
    config = {"max_new_tokens": 8, "do_sample": False, "ignore_eos": True}
    histories = [genai.ChatHistory(), genai.ChatHistory()]
    if chat_api == "start_chat":
        for pipe in pipelines:
            pipe.start_chat()
    try:
        for prompt in ["Describe the image.", "What else can you see?"]:
            if chat_api == "start_chat":
                results = [pipe.generate(prompt, images=[image], **config) for pipe in pipelines]
            else:
                for history in histories:
                    history.append({"role": "user", "content": prompt})
                results = [
                    pipe.generate(history, images=[image], **config) for pipe, history in zip(pipelines, histories)
                ]
            assert results[0].perf_metrics.get_num_input_tokens() == results[1].perf_metrics.get_num_input_tokens()
            assert results[0].texts == results[1].texts
            if chat_api == "ChatHistory":
                for history, result in zip(histories, results):
                    history.append({"role": "assistant", "content": result.texts[0]})
    finally:
        if chat_api == "start_chat":
            for pipe in pipelines:
                pipe.finish_chat()


def test_gguf_vlm_requires_projector_and_cpu(tiny_gguf_vlm_model):
    language, projector = tiny_gguf_vlm_model
    with pytest.raises(RuntimeError, match="mmproj_path"):
        genai.VLMPipeline(language, "CPU")
    with pytest.raises(RuntimeError, match="CPU"):
        genai.VLMPipeline(language, "GPU", mmproj_path=str(projector))


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
