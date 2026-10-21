from dataclasses import dataclass

import openvino as ov
import pytest

from openvino_genai import ChatHistory, GenerationConfig, VLMProcessor, VLMPipeline, VideoMetadata
from test_vlm_pipeline import _get_ov_model, cat_image, PROMPTS


MODEL_IDS = [
    "optimum-intel-internal-testing/tiny-random-qwen2vl",
]

ATTENTION_BACKENDS = ["SDPA", "PA"]

MAX_NEW_TOKENS = 10
DEVICE = "CPU"


@pytest.fixture(scope="module")
def cat_tensor(cat_image) -> ov.Tensor:
    return ov.Tensor(cat_image)


@dataclass(frozen=True)
class VLMObjects:
    ref_pipeline: VLMPipeline
    processor_pipeline: VLMPipeline
    processor: VLMProcessor


@dataclass(frozen=True)
class MultimodalInputs:
    prompt: str
    images: list[ov.Tensor]
    videos: list[ov.Tensor]
    videos_metadata: list[VideoMetadata]


@pytest.fixture(scope="module")
def vlm_objects(request: pytest.FixtureRequest) -> VLMObjects:
    model_id, attention_backend = request.param
    models_path = _get_ov_model(model_id)
    properties = {"ATTENTION_BACKEND": attention_backend}
    ref_pipeline = VLMPipeline(models_path, DEVICE, **properties)
    processor = VLMProcessor(models_path, DEVICE)
    processor_pipeline = VLMPipeline(models_path, processor, DEVICE, **properties)
    return VLMObjects(ref_pipeline=ref_pipeline, processor_pipeline=processor_pipeline, processor=processor)


@pytest.fixture(scope="module")
def multimodal_inputs(request: pytest.FixtureRequest) -> MultimodalInputs:
    prompt, images_fixtures, videos_fixtures, videos_metadata = request.param
    images = [request.getfixturevalue(image_fixture) for image_fixture in images_fixtures]
    videos = [request.getfixturevalue(video_fixture) for video_fixture in videos_fixtures]
    return MultimodalInputs(prompt=prompt, images=images, videos=videos, videos_metadata=videos_metadata)


def _generation_config(pipeline, apply_chat_template) -> GenerationConfig:
    config: GenerationConfig = pipeline.get_generation_config()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample = False
    config.apply_chat_template = apply_chat_template
    config.set_eos_token_id(pipeline.get_tokenizer().get_eos_token_id())
    return config


parametrize_all_models = pytest.mark.parametrize(
    "vlm_objects",
    [(model_id, attention_backend) for model_id in MODEL_IDS for attention_backend in ATTENTION_BACKENDS],
    ids=lambda p: f"{p[0]}/{p[1]}",
    indirect=["vlm_objects"],
)


parametrize_all_inputs = pytest.mark.parametrize(
    "multimodal_inputs",
    [
        pytest.param((PROMPTS[0], [], [], []), id="text_only"),
        pytest.param((PROMPTS[0], ["cat_tensor"], [], []), id="text+image"),
    ],
    indirect=["multimodal_inputs"],
)


@parametrize_all_models
@parametrize_all_inputs
def test_vlm_processor_string_prompt(vlm_objects: VLMObjects, multimodal_inputs: MultimodalInputs):
    ref_pipeline, processor, processor_pipeline = (
        vlm_objects.ref_pipeline,
        vlm_objects.processor,
        vlm_objects.processor_pipeline,
    )
    config = _generation_config(ref_pipeline, apply_chat_template=False)

    ref_result = ref_pipeline.generate(
        multimodal_inputs.prompt,
        images=multimodal_inputs.images,
        videos=multimodal_inputs.videos,
        videos_metadata=multimodal_inputs.videos_metadata,
        generation_config=config,
    )

    processed_inputs = processor.process(
        multimodal_inputs.prompt,
        images=multimodal_inputs.images,
        videos=multimodal_inputs.videos,
        videos_metadata=multimodal_inputs.videos_metadata,
    )

    processor_result = processor_pipeline.generate(processed_inputs, config)

    assert processor_result.texts == ref_result.texts


@parametrize_all_models
@parametrize_all_inputs
def test_vlm_processor_chat_history(vlm_objects: VLMObjects, multimodal_inputs: MultimodalInputs):
    ref_pipeline, processor, processor_pipeline = (
        vlm_objects.ref_pipeline,
        vlm_objects.processor,
        vlm_objects.processor_pipeline,
    )
    config = _generation_config(ref_pipeline, apply_chat_template=True)

    ref_history = ChatHistory()
    processor_history = ChatHistory()

    prompts = PROMPTS[:2]

    for i, prompt in enumerate(prompts):
        ref_history.append({"role": "user", "content": prompt})
        processor_history.append({"role": "user", "content": prompt})

        ref_result = ref_pipeline.generate(
            ref_history,
            images=multimodal_inputs.images,
            videos=multimodal_inputs.videos,
            videos_metadata=multimodal_inputs.videos_metadata,
            generation_config=config,
        )

        ref_history.append({"role": "assistant", "content": ref_result.texts[0]})

        processed_inputs = processor.process(
            processor_history,
            images=multimodal_inputs.images,
            videos=multimodal_inputs.videos,
            videos_metadata=multimodal_inputs.videos_metadata,
        )

        processor_result = processor_pipeline.generate(processed_inputs, config)

        processor_history.append({"role": "assistant", "content": processor_result.texts[0]})

        assert processor_result.texts == ref_result.texts, f"Processor result mismatch at prompt index {i}"


@parametrize_all_models
def test_non_processor_vlm_pipeline_with_processed_inputs_raises(vlm_objects: VLMObjects):
    ref_pipeline, processor = vlm_objects.ref_pipeline, vlm_objects.processor
    config = _generation_config(ref_pipeline, apply_chat_template=False)

    processed_inputs = processor.process(PROMPTS[0])
    with pytest.raises(RuntimeError):
        ref_pipeline.generate(processed_inputs, config)
