# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapters applied to the vision sub-models of a VLM pipeline."""

import re
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
# An architecture whose inputs embedder does not support visual LoRA.
UNSUPPORTED_VISION_LORA_MODEL_ID = "optimum-intel-internal-testing/tiny-random-mistral3"
# Architectures whose inputs embedders compile extra sub-models with the pipeline properties.
EXTRA_SUBMODELS_MODEL_IDS = [
    UNSUPPORTED_VISION_LORA_MODEL_ID,
    "optimum-intel-internal-testing/tiny-random-gemma4",
    "optimum-intel-internal-testing/tiny-random-qwen3-omni",
]

pytestmark = pytest.mark.vlm


@pytest.fixture(scope="module")
def model_id() -> str:
    return VISION_LORA_MODEL_ID


def _build_lora(model_path: Path, adapter_path: Path, key_prefix: str, rank: int = 2) -> int:
    """Creates a LoRA adapter targeting every linear layer of the given sub-model.

    Layer names follow the PyTorch frontend convention `__module.<module path>/aten::linear/MatMul`.
    """
    from safetensors.numpy import save_file

    model = openvino.Core().read_model(model_path)
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
        key = f"{key_prefix}{module_path.group(1)}"
        tensors[f"{key}.lora_A.weight"] = np.full((rank, input_dim), 0.5, dtype=np.float32)
        tensors[f"{key}.lora_B.weight"] = np.full((output_dim, rank), 0.5, dtype=np.float32)

    save_file(tensors, str(adapter_path))
    return len(tensors) // 2


def _build_vision_lora(models_path: Path, adapter_path: Path) -> int:
    # `openvino_vision_embeddings_merger_model.xml` is exported from the `model.visual` submodule.
    return _build_lora(
        Path(models_path) / "openvino_vision_embeddings_merger_model.xml", adapter_path, "base_model.model.model.visual."
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
def test_vlm_vision_lora(mode, vision_lora, vision_lora_reference, cat_tensor: openvino.Tensor) -> None:
    models_path, adapter = vision_lora
    backend, reference = vision_lora_reference
    config = GenerationConfig(max_new_tokens=10, do_sample=False)

    adapted = VLMPipeline(
        models_path, "CPU", ATTENTION_BACKEND=backend, adapters=AdapterConfig(adapter, 1.0, mode)
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
def unsupported_vision_lora_model() -> Path:
    pytest.importorskip("transformers", minversion="5.2.0")
    return _get_ov_model(UNSUPPORTED_VISION_LORA_MODEL_ID)


def _create_unsupported_visual_lora_pipeline(
    models_path: Path, backend: str, target_modules, unsupported_visual_lora: bool, tmp_path: Path
) -> None:
    from safetensors.numpy import save_file

    properties = {}
    if target_modules:
        # Rank 2 LoRA for the 64x64 projections of tiny-random-mistral3, so language model layers accept it.
        shapes = {"A": (2, 64), "B": (64, 2)}
        tensors = {
            f"base_model.model.{module}.lora_{matrix}.weight": np.ones(shapes[matrix], dtype=np.float32)
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
            VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend, **properties)
    else:
        VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend, **properties)


# The adapter check does not depend on the attention backend, so one backend covers the names.
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
    target_modules, unsupported_visual_lora, unsupported_vision_lora_model: Path, tmp_path: Path
) -> None:
    _create_unsupported_visual_lora_pipeline(
        unsupported_vision_lora_model, "SDPA", target_modules, unsupported_visual_lora, tmp_path
    )


@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
@pytest.mark.parametrize("target_modules, unsupported_visual_lora", [
    pytest.param(["model.visual.blocks.0.attn.q_proj"], True, id="qwen-vision"),
    pytest.param(["model.layers.0.self_attn.q_proj"], False, id="lm-only"),
])
def test_vlm_unsupported_visual_lora_constructor_backends(
    backend, target_modules, unsupported_visual_lora, unsupported_vision_lora_model: Path, tmp_path: Path
) -> None:
    _create_unsupported_visual_lora_pipeline(
        unsupported_vision_lora_model, backend, target_modules, unsupported_visual_lora, tmp_path
    )


@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
def test_vlm_unsupported_visual_lora_generate(backend, unsupported_vision_lora_model, vision_lora) -> None:
    _, adapter = vision_lora
    pipe = VLMPipeline(unsupported_vision_lora_model, "CPU", ATTENTION_BACKEND=backend)
    with pytest.raises(RuntimeError, match="Visual LoRA adapters are not supported"):
        pipe.generate(PROMPTS[0], max_new_tokens=1, adapters=AdapterConfig(adapter))


@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
def test_vlm_visual_lora_requires_constructor_config(backend, vision_lora) -> None:
    models_path, adapter = vision_lora
    pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    with pytest.raises(RuntimeError, match="Visual LoRA was not enabled when this pipeline was created"):
        pipe.generate(PROMPTS[0], max_new_tokens=1, adapters=AdapterConfig(adapter))


@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
@pytest.mark.parametrize("extra_submodels_model_id", EXTRA_SUBMODELS_MODEL_IDS)
def test_vlm_language_model_lora_with_extra_submodels(extra_submodels_model_id, backend, tmp_path: Path) -> None:
    pytest.importorskip("transformers", minversion="5.2.0")
    models_path = Path(_get_ov_model(extra_submodels_model_id))
    adapter_path = tmp_path / "adapter.safetensors"
    assert _build_lora(models_path / "openvino_language_model.xml", adapter_path, "base_model.model.") > 0
    config = GenerationConfig(max_new_tokens=5, do_sample=False)

    reference = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend).generate(
        PROMPTS[0], generation_config=config
    )
    adapted = VLMPipeline(
        models_path, "CPU", ATTENTION_BACKEND=backend, adapters=AdapterConfig(Adapter(str(adapter_path)))
    ).generate(PROMPTS[0], generation_config=config)

    assert adapted.texts[0] != reference.texts[0], "Language model LoRA adapter did not affect the generated text"
