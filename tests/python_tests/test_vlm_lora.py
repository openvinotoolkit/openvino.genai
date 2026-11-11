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
    ChatHistory,
    ContinuousBatchingPipeline,
    GenerationConfig,
    SchedulerConfig,
    Tokenizer,
    VLMPipeline,
)

from conftest import (
    ATTENTION_BACKEND,
    PROMPTS,
    _get_ov_model,
)

pytestmark = pytest.mark.vlm

# Qwen3.5 is multimodal on its own, there is no separate VL line for it.
VISION_LORA_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3.5"


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


@pytest.fixture(scope="module")
def vision_lora(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Adapter]:
    pytest.importorskip("transformers", minversion="5.2.0")
    models_path = _get_ov_model(VISION_LORA_MODEL_ID)
    adapter_path = tmp_path_factory.mktemp("vision_lora") / "adapter.safetensors"
    assert _build_vision_lora(models_path, adapter_path) > 0, "No vision layer to adapt was found"
    return models_path, Adapter(str(adapter_path))


@pytest.fixture(scope="module", params=ATTENTION_BACKEND)
def vision_lora_reference(request: pytest.FixtureRequest, vision_lora, cat_tensor: openvino.Tensor) -> tuple[str, str]:
    models_path, _ = vision_lora
    backend = request.param
    config = GenerationConfig(max_new_tokens=10, do_sample=False)
    reference = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend).generate(
        PROMPTS[0], images=[cat_tensor], generation_config=config
    )
    return backend, reference.texts[0]


@pytest.mark.parametrize("mode", [AdapterConfig.Mode.MODE_STATIC, AdapterConfig.Mode.MODE_DYNAMIC, AdapterConfig.Mode.MODE_FUSE])
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


@pytest.mark.parametrize("use_chat_history", [False, True])
def test_vlm_vision_lora_switched_per_generate(use_chat_history, vision_lora, vision_lora_reference, cat_tensor: openvino.Tensor) -> None:
    models_path, adapter = vision_lora
    backend, without_adapters = vision_lora_reference
    config = GenerationConfig(max_new_tokens=10, do_sample=False)

    def dynamic(alpha: float) -> AdapterConfig:
        return AdapterConfig(adapter, alpha, AdapterConfig.Mode.MODE_DYNAMIC)

    def generate(pipe: VLMPipeline, **kwargs) -> str:
        prompt = ChatHistory([{"role": "user", "content": PROMPTS[0]}]) if use_chat_history else PROMPTS[0]
        return pipe.generate(prompt, images=[cat_tensor], generation_config=config, **kwargs).texts[0]

    # Pipelines configured once and never switched, to compare the switched outputs against.
    if use_chat_history:
        without_adapters = generate(VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend))
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
        pipe = ContinuousBatchingPipeline(models_path, Tokenizer(models_path), SchedulerConfig(), "CPU", adapters=adapters)
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
