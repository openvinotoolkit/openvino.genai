# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapters applied to the vision sub-models of a VLM pipeline."""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import openvino
import pytest
from openvino_genai import (
    Adapter,
    AdapterConfig,
    ContinuousBatchingPipeline,
    GenerationConfig,
    SchedulerConfig,
    Tokenizer,
    VLMPipeline,
)

from test_vlm_pipeline import (
    ATTENTION_BACKEND,
    PROMPTS,
    _get_ov_model,
    cat_image,
    cat_tensor,
)

# Qwen3.5 is multimodal on its own, there is no separate VL line for it.
VISION_LORA_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3.5"

pytestmark = [
    pytest.mark.vlm,
    pytest.mark.parametrize("model_id", [VISION_LORA_MODEL_ID], scope="module"),
]


def _build_vision_lora(models_path: Path, adapter_path: Path, rank: int = 2) -> int:
    """Creates a LoRA adapter targeting every linear layer of the vision merger sub-model.

    Layer names follow the PyTorch frontend convention `__module.<module path>/aten::linear/MatMul`,
    and `openvino_vision_embeddings_merger_model.xml` is exported from the `model.visual` submodule.
    """
    from safetensors.numpy import save_file

    model = openvino.Core().read_model(Path(models_path) / "openvino_vision_embeddings_merger_model.xml")
    tensors: dict[str, np.ndarray] = {}
    for op in model.get_ops():
        if op.get_type_name() != "MatMul":
            continue
        module_path = re.match(r"__module\.(.+?)/", op.get_friendly_name())
        weight_shape = op.input(1).get_partial_shape()
        if module_path is None or weight_shape.is_dynamic or weight_shape.rank.get_length() != 2:
            continue
        transpose_b = op.get_attributes()["transpose_b"]
        input_dim = weight_shape[1 if transpose_b else 0].get_length()
        output_dim = weight_shape[0 if transpose_b else 1].get_length()
        key = f"base_model.model.model.visual.{module_path.group(1)}"
        tensors[f"{key}.lora_A.weight"] = np.full((rank, input_dim), 0.5, dtype=np.float32)
        tensors[f"{key}.lora_B.weight"] = np.full((output_dim, rank), 0.5, dtype=np.float32)

    save_file(tensors, str(adapter_path))
    return len(tensors) // 2


def _create_vlm_pipeline(models_path, backend, from_memory, **properties) -> VLMPipeline:
    if not from_memory:
        return VLMPipeline(str(models_path), "CPU", ATTENTION_BACKEND=backend, **properties)
    models_path = Path(models_path)
    models = {
        path.stem.removeprefix("openvino_").removesuffix("_model"): (
            path.read_text(encoding="utf-8"),
            openvino.Tensor(np.fromfile(path.with_suffix(".bin"), dtype=np.uint8)),
        )
        for path in models_path.glob("openvino_*_model.xml")
    }
    return VLMPipeline(
        models, Tokenizer(models_path), models_path, "CPU", ATTENTION_BACKEND=backend, **properties
    )


@pytest.fixture(scope="module")
def vision_lora(model_id: str, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Adapter]:
    pytest.importorskip("transformers", minversion="5.2.0")
    models_path = _get_ov_model(model_id)
    adapter_path = tmp_path_factory.mktemp("vision_lora") / "adapter.safetensors"
    assert _build_vision_lora(models_path, adapter_path) > 0, "No vision layer to adapt was found"
    return models_path, Adapter(str(adapter_path))


@pytest.fixture(scope="module", params=ATTENTION_BACKEND)
def vision_lora_reference(
    request: pytest.FixtureRequest, model_id: str, cat_tensor: openvino.Tensor
) -> tuple[str, str]:
    models_path = _get_ov_model(model_id)
    backend = request.param
    config = GenerationConfig(max_new_tokens=10, do_sample=False)
    reference = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend).generate(
        PROMPTS[0], images=[cat_tensor], generation_config=config
    )
    return backend, reference.texts[0]


@pytest.mark.parametrize(
    "mode", [AdapterConfig.Mode.MODE_STATIC, AdapterConfig.Mode.MODE_DYNAMIC, AdapterConfig.Mode.MODE_FUSE]
)
@pytest.mark.parametrize("from_memory", [False, True], ids=["path", "models-map"])
def test_vlm_vision_lora(mode, from_memory, vision_lora, vision_lora_reference, cat_tensor: openvino.Tensor) -> None:
    models_path, adapter = vision_lora
    backend, reference = vision_lora_reference
    config = GenerationConfig(max_new_tokens=10, do_sample=False)

    adapted = _create_vlm_pipeline(
        models_path, backend, from_memory, adapters=AdapterConfig(adapter, 1.0, mode)
    ).generate(PROMPTS[0], images=[cat_tensor], generation_config=config)

    assert adapted.texts[0] != reference, "Vision LoRA adapter did not affect the generated text"

    if mode == AdapterConfig.Mode.MODE_STATIC:
        alpha_1 = VLMPipeline(
            models_path,
            "CPU",
            ATTENTION_BACKEND=backend,
            adapters=AdapterConfig(adapter, 1.0, AdapterConfig.Mode.MODE_DYNAMIC),
        ).generate(PROMPTS[0], images=[cat_tensor], generation_config=config)
        assert adapted.texts[0] == alpha_1.texts[0], (
            "STATIC and DYNAMIC adapters with the same alpha must produce identical text"
        )


def test_vlm_vision_lora_switched_per_generate(vision_lora, vision_lora_reference, cat_tensor: openvino.Tensor) -> None:
    models_path, adapter = vision_lora
    backend, without_adapters = vision_lora_reference
    config = GenerationConfig(max_new_tokens=10, do_sample=False)

    def dynamic(alpha: float) -> AdapterConfig:
        return AdapterConfig(adapter, alpha, AdapterConfig.Mode.MODE_DYNAMIC)

    def generate(pipe: VLMPipeline, **kwargs) -> str:
        return pipe.generate(PROMPTS[0], images=[cat_tensor], generation_config=config, **kwargs).texts[0]

    # Pipelines configured once and never switched, to compare the switched outputs against.
    alpha_1 = generate(VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend, adapters=dynamic(1.0)))
    alpha_2 = generate(VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend, adapters=dynamic(2.0)))
    assert len({without_adapters, alpha_1, alpha_2}) == 3, "Equal references would make the checks below meaningless"

    pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend, adapters=dynamic(1.0))

    assert generate(pipe) == alpha_1, "An omitted adapters argument must keep the configuration of the constructor"
    assert generate(pipe, adapters=dynamic(2.0)) == alpha_2, "Changing only the alpha must upload the new alpha"
    assert generate(pipe, adapters=AdapterConfig()) == without_adapters, "Dropping the adapter must undo it"


@pytest.mark.parametrize("with_tokenizer", [False, True])
def test_vlm_vision_lora_cb_constructors(with_tokenizer, vision_lora, cat_tensor: openvino.Tensor) -> None:
    models_path, adapter = vision_lora
    adapters = AdapterConfig(adapter, 1.0, AdapterConfig.Mode.MODE_DYNAMIC)
    config = GenerationConfig(max_new_tokens=10, do_sample=False)
    reference = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND="PA", adapters=adapters).generate(
        PROMPTS[0], images=[cat_tensor], generation_config=config
    )

    if with_tokenizer:
        pipe = ContinuousBatchingPipeline(
            models_path, Tokenizer(models_path), SchedulerConfig(), "CPU", adapters=adapters
        )
    else:
        pipe = ContinuousBatchingPipeline(models_path, SchedulerConfig(), "CPU", {"adapters": adapters})

    config.adapters = adapters
    adapted = pipe.generate([PROMPTS[0]], images=[[cat_tensor]], generation_config=[config])
    assert adapted[0].texts[0] == reference.texts[0]

    config.adapters = AdapterConfig()
    without_adapters = pipe.generate([PROMPTS[0]], images=[[cat_tensor]], generation_config=[config])
    reference_without_adapters = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND="PA").generate(
        PROMPTS[0], images=[cat_tensor], generation_config=GenerationConfig(max_new_tokens=10, do_sample=False)
    )
    assert without_adapters[0].texts[0] == reference_without_adapters.texts[0]


@pytest.fixture(scope="module")
def unsupported_vision_lora_model(vision_lora, tmp_path_factory: pytest.TempPathFactory) -> Path:
    models_path, _ = vision_lora
    models_path = Path(models_path)
    target = tmp_path_factory.mktemp("unsupported_vision_lora")
    for source in models_path.iterdir():
        if source.is_file() and (source.suffix == ".json" or source.stem in {
            "openvino_language_model", "openvino_text_embeddings_model", "openvino_tokenizer", "openvino_detokenizer"
        }):
            shutil.copy2(source, target / source.name)
    config_path = target / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "llava"
    config_path.write_text(json.dumps(config))
    pixel_values = openvino.opset13.parameter([1, 3, 14, 14], dtype=np.float32, name="pixel_values")
    openvino.save_model(
        openvino.Model([pixel_values], [pixel_values]), target / "openvino_vision_embeddings_model.xml"
    )
    return target


@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
@pytest.mark.parametrize("from_memory", [False, True], ids=["path", "models-map"])
@pytest.mark.parametrize("target_modules, unsupported_visual_lora", [
    pytest.param(["model.visual.blocks.0.attn.q_proj"], True, id="qwen-vision"),
    pytest.param(["vision_tower.vision_model.encoder.layers.0.self_attn.q_proj"], False, id="other-vision"),
    pytest.param(
        ["model.layers.0.self_attn.q_proj", "model.visual.blocks.0.attn.q_proj"], True, id="mixed"
    ),
    pytest.param(["model.layers.0.self_attn.q_proj"], False, id="lm-only"),
    pytest.param(["model.layers.0.mlp1"], False, id="lm-mlp1"),
    pytest.param(["model.visual_extra.blocks.0.attn.q_proj"], False, id="similar-prefix"),
    pytest.param(["other.visual.blocks.0.attn.q_proj"], False, id="different-prefix"),
    pytest.param([], False, id="empty"),
    pytest.param(None, False, id="omitted"),
])
def test_vlm_unsupported_visual_lora_constructor(
    backend, from_memory, target_modules, unsupported_visual_lora, unsupported_vision_lora_model: Path, tmp_path: Path
) -> None:
    from safetensors.numpy import save_file

    properties = {}
    if target_modules:
        tensors = {
            f"base_model.model.{module}.lora_{matrix}.weight": np.ones((2, 2), dtype=np.float32)
            for module in target_modules
            for matrix in ("A", "B")
        }
        adapter_path = tmp_path / "adapter.safetensors"
        save_file(tensors, str(adapter_path))
        properties["adapters"] = AdapterConfig(Adapter(str(adapter_path)))
    elif target_modules is not None:
        properties["adapters"] = AdapterConfig()

    if unsupported_visual_lora:
        with pytest.raises(RuntimeError, match="Visual LoRA adapters are not supported"):
            _create_vlm_pipeline(unsupported_vision_lora_model, backend, from_memory, **properties)
    else:
        _create_vlm_pipeline(unsupported_vision_lora_model, backend, from_memory, **properties)


@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
@pytest.mark.parametrize("from_memory", [False, True], ids=["path", "models-map"])
def test_vlm_unsupported_visual_lora_generate(backend, from_memory, unsupported_vision_lora_model, vision_lora) -> None:
    _, adapter = vision_lora
    pipe = _create_vlm_pipeline(unsupported_vision_lora_model, backend, from_memory)
    with pytest.raises(RuntimeError, match="Visual LoRA adapters are not supported"):
        pipe.generate(PROMPTS[0], max_new_tokens=1, adapters=AdapterConfig(adapter))


@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
@pytest.mark.parametrize("from_memory", [False, True], ids=["path", "models-map"])
def test_vlm_visual_lora_requires_constructor_config(backend, from_memory, vision_lora) -> None:
    models_path, adapter = vision_lora
    pipe = _create_vlm_pipeline(models_path, backend, from_memory)
    with pytest.raises(RuntimeError, match="Visual LoRA was not enabled when this pipeline was created"):
        pipe.generate(PROMPTS[0], max_new_tokens=1, adapters=AdapterConfig(adapter))
