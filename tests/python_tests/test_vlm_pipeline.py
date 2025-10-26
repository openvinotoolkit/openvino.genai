# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Fixture hierarchy
synthetic_video ──────── synthetic_video_32x32_tensor
cat_image ───────────┬── cat_tensor
    │                ├── cat_image_384x384
    │                ├── cat_image_336x336
    │                └── cat_image_32x32
    │
    ├── iteration_images
    │       ├── cat_tensor
    │       ├── car_tensor
    │       └── handwritten_tensor
    │
    ├── image_sequence
    │       └── cat_tensor
    │
    └── conversation_requests
            ├── cat_tensor
            ├── car_tensor
            └── handwritten_tensor
car_tensor
handwritten_tensor
ov_pipe_model
ov_continious_batching_pipe
"""

import collections
from dataclasses import dataclass
from typing import Callable, Generator
import openvino_tokenizers
import openvino
import PIL
import pytest
import platform
import requests
import sys
import os
import numpy as np
import transformers
from optimum.intel.openvino import OVModelForVisualCausalLM
from openvino_genai import (
    VLMPipeline,
    GenerationConfig,
    SchedulerConfig,
    ContinuousBatchingPipeline,
    GenerationStatus,
    StreamingStatus,
    GenerationFinishReason,
)

from utils.network import retry_request
from utils.generation_config import (
    get_beam_search,
    get_multinomial_all_parameters,
    get_greedy,
)
from utils.constants import get_ov_cache_models_dir

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VlmModelInfo:
    model_id: str
    ov_backend: str
    image_tag: Callable[[int], str]
    resolution: int
    pipeline: VLMPipeline


PROMPTS: list[str] = [
    "What is on the image?",
    "What is special about this image?",
]


VIDEO_MODEL_IDS = [
    "katuni4ka/tiny-random-llava-next-video",
]


MODEL_IDS: list[str] = [
    "katuni4ka/tiny-random-minicpmv-2_6",
    "katuni4ka/tiny-random-phi3-vision",
    "katuni4ka/tiny-random-phi-4-multimodal",
    "katuni4ka/tiny-random-llava",
    "katuni4ka/tiny-random-llava-next",
    "katuni4ka/tiny-random-internvl2",
    "katuni4ka/tiny-random-qwen2vl",
    "katuni4ka/tiny-random-qwen2.5-vl",
    "katuni4ka/tiny-random-gemma3",
    "qnguyen3/nanoLLaVA",
    *VIDEO_MODEL_IDS,
]


ADD_REQUEST_MODEL_IDS = [
    MODEL_IDS[0],
    *VIDEO_MODEL_IDS
]


TAG_GENERATOR_BY_MODEL: dict[str, Callable[[int], str]] = {
    "katuni4ka/tiny-random-llava": lambda idx: "<image>",
    "katuni4ka/tiny-random-llava-next": lambda idx: "<image>",
    "katuni4ka/tiny-random-qwen2vl": lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
    "katuni4ka/tiny-random-qwen2.5-vl": lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
    "katuni4ka/tiny-random-gemma3": lambda idx: "<start_of_image>",
    "katuni4ka/tiny-random-internvl2": lambda idx: "<image>\n",
    "katuni4ka/tiny-random-minicpmv-2_6": lambda idx: "<image>./</image>\n",
    "katuni4ka/tiny-random-phi3-vision": lambda idx: f"<|image_{idx + 1}|>\n",
    "katuni4ka/tiny-random-llava-next-video": lambda idx: "<image>\n",
    "qnguyen3/nanoLLaVA": lambda idx: "<image>\n",
}


RESOLUTION_BY_MODEL: dict[str, int | None] = {
    "katuni4ka/tiny-random-gemma3": 32,
    "qnguyen3/nanoLLaVA": 384,
    "katuni4ka/tiny-random-llava-next-video": 336,
}


RESOLUTION_BY_VIDEO_MODEL: dict[str, int | None] = {
    "katuni4ka/tiny-random-llava-next-video": 32,
}


DEFAULT_RESOLUTION = 336


ATTENTION_BACKEND: list[str] = ["PA", "SDPA"]


DEFAULT_MAX_NEW_TOKENS = 30
DEFAULT_SCORE_EPSILON = 0.001
IMAGE_TOKENS_NUM = 54
MAX_RETRIES = 10
RETRY_BASE_DELAY_SEC = 0.1
RETRY_MAX_DELAY_SEC = 2.0

TEST_IMAGE_URLS = {
    'cat': 'https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11',
    'car': 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg',
    'handwritten': 'https://github.com/user-attachments/assets/8c9ae017-7837-4abc-ae92-c1054c9ec350'
}


def _setup_generation_config(
    pipeline: VLMPipeline, 
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, 
    ignore_eos: bool = False,
    set_eos_token: bool = True,
    do_sample: bool = True,
) -> GenerationConfig:
    generation_config = pipeline.get_generation_config()
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample
    
    if set_eos_token:
        generation_config.set_eos_token_id(pipeline.get_tokenizer().get_eos_token_id())
    
    if ignore_eos:
        generation_config.ignore_eos = True        
    
    return generation_config


def _get_ov_model(model_id: str) -> str:
    ov_cache_models_dir = get_ov_cache_models_dir()
    dir_name = str(model_id).replace(os.sep, "_")
    model_dir = ov_cache_models_dir / dir_name
    if (model_dir / "openvino_language_model.xml").exists():
        return model_dir
    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    processor = retry_request(
        lambda: transformers.AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            **align_with_optimum_cli,
        )
    )
    model = retry_request(
        lambda: OVModelForVisualCausalLM.from_pretrained(
            model_id,
            compile=False,
            device="CPU",
            export=True,
            load_in_8bit=False,
            trust_remote_code=True,
        )
    )
    if model.config.model_type == "llava-qwen2":
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # For tiny-random-internvl2 processor is actually tokenizer
    elif isinstance(processor, transformers.Qwen2TokenizerFast):
        tokenizer = processor
        processor = transformers.AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
    else:
        tokenizer = processor.tokenizer
        if tokenizer.chat_template is None:
            tokenizer.chat_template = processor.chat_template
    tokenizer.save_pretrained(model_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
        tokenizer, with_detokenizer=True
    )
    openvino.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, model_dir / "openvino_detokenizer.xml")

    if tokenizer.chat_template is not None and model.config.model_type == "phi3_v":
        # It seems that tiny-random-phi3-vision is saved incorrectly. That line works this around.
        processor.chat_template = tokenizer.chat_template
    processor.audio_tokenizer = None
    processor.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    return model_dir


# On macOS, transformers<4.52 is required, but this causes gemma3 to fail
GEMMA3_MACOS_XFAIL_REASON = "gemma3 not supported on macOS with older transformers"


@pytest.fixture(scope="module")
def ov_pipe_model(request: pytest.FixtureRequest) -> VlmModelInfo:
    ov_model, ov_backend = request.param
    
    if sys.platform == "darwin" and "gemma3" in ov_model:
        pytest.xfail(GEMMA3_MACOS_XFAIL_REASON)

    models_path = _get_ov_model(ov_model)
    
    pipeline = VLMPipeline(
        models_path, 
        "CPU", 
        ATTENTION_BACKEND=ov_backend
    )
    return VlmModelInfo(
        ov_model, 
        ov_backend, 
        TAG_GENERATOR_BY_MODEL.get(ov_model, lambda idx: ""), 
        RESOLUTION_BY_MODEL.get(ov_model, DEFAULT_RESOLUTION), 
        pipeline
    )


parametrize_all_models = pytest.mark.parametrize(
    "ov_pipe_model",
    [(m, b) for m in MODEL_IDS for b in ATTENTION_BACKEND],
    ids=lambda p: f"{p[0]}/{p[1]}",
    indirect=["ov_pipe_model"],
)


parametrize_all_models_with_video = pytest.mark.parametrize(
    "ov_pipe_model",
    [(m, b) for m in VIDEO_MODEL_IDS for b in ATTENTION_BACKEND],
    ids=lambda p: f"{p[0]}/{p[1]}",
    indirect=["ov_pipe_model"],
)


parametrize_one_model_sdpa = pytest.mark.parametrize(
    "ov_pipe_model",
    [(MODEL_IDS[0], "SDPA")],
    ids=lambda p: f"{p[0]}/{p[1]}",
    indirect=["ov_pipe_model"],
)


parametrize_one_model_pa = pytest.mark.parametrize(
    "ov_pipe_model",
    [(MODEL_IDS[0], "PA")],
    ids=lambda p: f"{p[0]}/{p[1]}",
    indirect=["ov_pipe_model"],
)


parametrize_one_model_backends = pytest.mark.parametrize(
    "ov_pipe_model",
    [(MODEL_IDS[0], b) for b in ATTENTION_BACKEND],
    ids=lambda p: f"{p[0]}/{p[1]}",
    indirect=["ov_pipe_model"],
)
    
@pytest.fixture(scope="module")
def ov_continious_batching_pipe() -> ContinuousBatchingPipeline:
    models_path = _get_ov_model(MODEL_IDS[0])
    return ContinuousBatchingPipeline(models_path, SchedulerConfig(), "CPU")
    
@pytest.fixture(scope="module")
def ov_continious_batching_pipe_gemma() -> ContinuousBatchingPipeline:
    models_path = _get_ov_model(MODEL_IDS[8])
    return ContinuousBatchingPipeline(models_path, SchedulerConfig(), "CPU")


def download_image(link: str) -> PIL.Image:
    return PIL.Image.open(requests.get(link, stream=True).raw).convert("RGB")


def from_cache_or_download(pytestconfig: pytest.Config, link: str, file_name: str):
    def implementation():
        try:
            image_path = pytestconfig.cache.mkdir("images") / file_name
        except AttributeError:
            # Cache is disabled with -p no:cacheprovider
            return download_image(link)
        if image_path.exists():
            image: PIL.Image = PIL.Image.open(image_path)
        else:
            image = download_image(link)
            image.save(image_path)
        return image
    return retry(implementation, PIL.UnidentifiedImageError)


@pytest.fixture(scope="module")
def cat_image(pytestconfig: pytest.Config):
    return from_cache_or_download(pytestconfig, TEST_IMAGE_URLS['cat'], "cat.jpg")

def resize_video(video, shape):
    video_resized = []
    for frame in video:
        pil_image = PIL.Image.fromarray(frame)
        resized = pil_image.resize(shape)
        video_resized.append(np.array(resized))
    return np.array(video_resized)

@pytest.fixture(scope="module")
def synthetic_video(pytestconfig):
    # TODO: use real video
    car_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    image = from_cache_or_download(pytestconfig, car_url, "car.jpg")

    # make 10 frames
    total_frames = 10
    frames = []
    frames.append(np.array(image))
    shift = 3
    for i in range(1, total_frames):
        new_frame = np.zeros(np.array(image).shape, np.array(image).dtype)

        width, height = image.size
        for x in range(0, width):
            for y in range(0, height):
                # shift previous frame
                new_frame[y, x] = frames[i-1][y, (x - shift + width) % width]
        frames.append(new_frame)

    return frames

@pytest.fixture(scope="module")
def synthetic_video_32x32(synthetic_video):
    return resize_video(synthetic_video, (32, 32))


@pytest.fixture(scope="module")
def cat_image_384x384(cat_image):
    return cat_image.resize((384, 384))


@pytest.fixture(scope="module")
def cat_image_336x336(cat_image):
    return cat_image.resize((336, 336))


@pytest.fixture(scope="module")
def cat_image_32x32(cat_image):
    return cat_image.resize((32, 32))


@pytest.fixture(scope="module")
def cat_tensor(cat_image) -> openvino.Tensor:
    return openvino.Tensor(cat_image)


@pytest.fixture(scope="module")
def car_tensor(pytestconfig: pytest.Config) -> openvino.Tensor:
    return openvino.Tensor(from_cache_or_download(pytestconfig, TEST_IMAGE_URLS['car'], "car.jpg"))


@pytest.fixture(scope="module")
def synthetic_video_32x32_tensor(synthetic_video_32x32):
    return openvino.Tensor(synthetic_video_32x32)


@pytest.fixture(scope="module")
def handwritten_tensor(pytestconfig: pytest.Config) -> openvino.Tensor:
    return openvino.Tensor(from_cache_or_download(pytestconfig, TEST_IMAGE_URLS['handwritten'], "handwritten.png"))


@pytest.fixture(scope="function", params=[
    pytest.param([], id="no_images"),
    pytest.param(["cat_tensor"], id="single_image"),
    pytest.param(["cat_tensor", "handwritten_tensor", "car_tensor"], id="multiple_images"),
])
def test_images(request: pytest.FixtureRequest):
    return [request.getfixturevalue(image) for image in request.param]


@pytest.mark.precommit
@parametrize_all_models
def test_vlm_pipeline(ov_pipe_model: VlmModelInfo, test_images: list[openvino.Tensor]):
    ov_pipe = ov_pipe_model.pipeline
    result_from_streamer = []
    
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

    generation_config = _setup_generation_config(ov_pipe)

    res = ov_pipe.generate(
        PROMPTS[0],
        images=test_images,
        generation_config=generation_config,
        streamer=streamer,
    )
    assert res.texts[0] == "".join(result_from_streamer)


@pytest.mark.precommit
@pytest.mark.parametrize(
    "config", 
    [
        pytest.param(get_greedy(), id="greedy"),
        pytest.param(get_beam_search(), id="beam_search"),
    ]
)
@parametrize_one_model_pa
def test_vlm_continuous_batching_generate_vs_add_request(
    ov_pipe_model: VlmModelInfo,
    ov_continious_batching_pipe: ContinuousBatchingPipeline,
    config: GenerationConfig, 
    request: pytest.FixtureRequest,
    cat_tensor: openvino.Tensor
):    
    ov_pipe = ov_pipe_model.pipeline
    generation_config = config
    generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    image_links_list = [[], [cat_tensor]]
    
    if ov_pipe_model.model_id in VIDEO_MODEL_IDS:
        synthetic_video_32x32_tensor = request.getfixturevalue("synthetic_video_32x32_tensor")
        images_list = [[], [cat_tensor], [cat_tensor]]
        videos_list = [[synthetic_video_32x32_tensor], [synthetic_video_32x32_tensor], []]
    else:
        images_list = [[], [cat_tensor]]
        videos_list = [[], []]

    res_generate = []
    for idx, images in enumerate(images_list):
        videos = videos_list[idx]
        res_generate.append(
            ov_pipe.generate(
                PROMPTS[0], 
                images=images, 
                videos=videos, 
                generation_config=generation_config,
            )
        )

    tokenizer = ov_continious_batching_pipe.get_tokenizer()

    for idx, images in enumerate(images_list):
        videos = videos_list[idx]
        handle = ov_continious_batching_pipe.add_request(
            idx, PROMPTS[0], 
            images=images, 
            videos=videos, 
            generation_config=generation_config,
        )
        while handle.get_status() != GenerationStatus.FINISHED:
            ov_continious_batching_pipe.step()
        outputs = handle.read_all()
        for out_idx, output in enumerate(outputs):
            text = tokenizer.decode(output.generated_ids)
            assert text == res_generate[idx].texts[out_idx]
            assert abs(output.score - res_generate[idx].scores[out_idx]) < DEFAULT_SCORE_EPSILON
            assert (
                output.finish_reason == GenerationFinishReason.STOP
                or output.finish_reason == GenerationFinishReason.LENGTH
            )


@pytest.mark.precommit
@pytest.mark.parametrize(
    "config", 
    [
        pytest.param(get_greedy(), id="greedy"),
        pytest.param(get_beam_search(), id="beam_search"),
    ]
)
def test_vlm_continuous_batching_generate_vs_add_request_for_gemma(
    ov_continious_batching_pipe_gemma: ContinuousBatchingPipeline,
    config: GenerationConfig, 
    cat_tensor: openvino.Tensor,
):
    ov_cb_pipe = ov_continious_batching_pipe_gemma
    image_links_list = [[], [cat_tensor]]
    tokenizer = ov_cb_pipe.get_tokenizer()

    for idx, images in enumerate(image_links_list):
        handle = ov_cb_pipe.add_request(
            idx, PROMPTS[0], images, config
        )
        while handle.get_status() != GenerationStatus.FINISHED:
            ov_cb_pipe.step()
        outputs = handle.read_all()
        for output in outputs:
            text = tokenizer.decode(output.generated_ids)
            assert len(output.generated_ids) > 0, f"Should generate at least one token"
            assert text.strip() != "", f"Decoded text should not be empty"
            assert (
                output.finish_reason == GenerationFinishReason.STOP
                or output.finish_reason == GenerationFinishReason.LENGTH
            )


@pytest.mark.parametrize(
    "config", 
    [
        pytest.param(get_greedy(), id="greedy"),
        pytest.param(get_beam_search(), id="beam_search"),
    ]
)
@parametrize_one_model_sdpa
def test_vlm_continuous_batching_vs_stateful(
    ov_pipe_model: VlmModelInfo,
    ov_continious_batching_pipe: ContinuousBatchingPipeline,
    config: GenerationConfig, 
    cat_tensor: openvino.Tensor,
):
    ov_pipe = ov_pipe_model.pipeline
    generation_config = config
    generation_config.max_new_tokens = 25
    image_links_list = [[], [cat_tensor]]

    res_cb = []
    for images in image_links_list:
        res_cb.append(
            ov_continious_batching_pipe.generate(
                [PROMPTS[0]], images=[images], generation_config=[generation_config]
            )
        )

    for idx, images in enumerate(image_links_list):
        res_stateful = ov_pipe.generate(
            PROMPTS[0], images=images, generation_config=generation_config
        )
        for out_idx, text in enumerate(res_stateful.texts):
            assert text == res_cb[idx][0].texts[out_idx]
            assert (
                abs(res_stateful.scores[out_idx] - res_cb[idx][0].scores[out_idx]) < DEFAULT_SCORE_EPSILON
            )


@pytest.fixture(scope="module", params=[
    pytest.param([[], []], id="generation with text input only"),
    pytest.param(
        [[], ["cat_tensor", "car_tensor", "handwritten_tensor"], []],
        id="combination of generations with text input and text + image input, empty image first"
    ),
    pytest.param(
        [["cat_tensor", "car_tensor", "handwritten_tensor"], ["cat_tensor"]],
        id="generation with text + image input"
    ),
    pytest.param(
        [["cat_tensor", "car_tensor", "handwritten_tensor"], [], ["cat_tensor"]],
        id="combination of generations with text input and text + image input, image input first"
    ),
])
def iteration_images(request) -> list[list[PIL.Image]]:
    return [[request.getfixturevalue(image) for image in bundle] for bundle in request.param]


@pytest.fixture(scope="module", params=[
    pytest.param(
        [[[], [], []], [[], [ "synthetic_video_32x32_tensor"], []]],
        id="Video on second iteration"
    ),
    pytest.param(
        [[["cat_tensor"], [], []], [["synthetic_video_32x32_tensor"], [], ["synthetic_video_32x32_tensor"]]],
        id="Image + video on first iteration, image on third iteration"
    ),
    pytest.param(
        [[["cat_tensor", "car_tensor", "handwritten_tensor"], []], [["synthetic_video_32x32_tensor"], ["synthetic_video_32x32_tensor"]]],
        id="3 images + video on first iteration, video on second iteration"
    ),
])
def iteration_images_and_videos(request):
    params = []
    for param in request.param:
        params.append([[request.getfixturevalue(image) for image in bundle] for bundle in param])
    return params


@pytest.mark.precommit
@parametrize_all_models
@pytest.mark.parametrize("system_message", ["", "You are a helpful assistant."])
def test_vlm_pipeline_chat(
    ov_pipe_model: VlmModelInfo, 
    system_message: str, 
    iteration_images: list[list[PIL.Image]],
):
    ov_pipe = ov_pipe_model.pipeline
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False
    
    generation_config = _setup_generation_config(ov_pipe)

    ov_pipe.start_chat(system_message)

    images = iteration_images[0]

    result_from_streamer = []
    res = ov_pipe.generate(
        PROMPTS[0],
        images=images,
        generation_config=generation_config,
        streamer=streamer,
    )
    assert res.texts[0] == "".join(result_from_streamer)

    for image_set in iteration_images[1:]:
        result_from_streamer = []
        res = ov_pipe.generate(
            PROMPTS[1],
            images=image_set,
            generation_config=generation_config,
            streamer=streamer,
        )
        assert res.texts[0] == "".join(result_from_streamer)

    ov_pipe.finish_chat()

@pytest.mark.precommit
@parametrize_all_models_with_video
@pytest.mark.parametrize("system_message", ["", "You are a helpful assistant."])
def test_vlm_pipeline_chat_with_video(
    ov_pipe_model: VlmModelInfo, 
    system_message: str, 
    iteration_images_and_videos,
):
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

    ov_pipe = ov_pipe_model.pipeline
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    ov_pipe.start_chat(system_message)
    iteration_images = iteration_images_and_videos[0]
    iteration_videos = iteration_images_and_videos[1]

    images = iteration_images[0]
    videos = iteration_videos[0]

    result_from_streamer = []
    res = ov_pipe.generate(
        PROMPTS[0],
        images=images,
        videos=videos,
        generation_config=generation_config,
        streamer=streamer,
    )
    assert res.texts[0] == "".join(result_from_streamer)

    for idx, image_set in enumerate(iteration_images[1:]):
        result_from_streamer = []
        videos = iteration_videos[idx]
        res = ov_pipe.generate(
            PROMPTS[1],
            images=image_set,
            videos=videos,
            generation_config=generation_config,
            streamer=streamer,
        )
        assert res.texts[0] == "".join(result_from_streamer)

    ov_pipe.finish_chat()


@pytest.mark.precommit
@parametrize_one_model_backends
def test_vlm_get_tokenizer(ov_pipe_model: VlmModelInfo):
    ov_pipe = ov_pipe_model.pipeline
    tokenizer = ov_pipe.get_tokenizer()
    tokenizer.encode("")


@pytest.mark.precommit
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(get_beam_search(), id="beam_search"),
        pytest.param(get_multinomial_all_parameters(), id="multinomial_all_parameters"),
    ],
)
@parametrize_one_model_backends
def test_sampling(
    ov_pipe_model: VlmModelInfo,
    config, 
    cat_tensor: openvino.Tensor,
):
    ov_pipe = ov_pipe_model.pipeline
    ov_pipe.generate(PROMPTS[0], image=cat_tensor, generation_config=config)


@pytest.mark.precommit
@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
def test_perf_metrics(
    backend: str, 
    cat_tensor: openvino.Tensor, 
):
    import numpy as np
    from time import perf_counter_ns

    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    # Using non-cached model to get more accurate load time
    model_path = _get_ov_model("katuni4ka/tiny-random-minicpmv-2_6")
    start_time = perf_counter_ns()
    pipe = VLMPipeline(model_path, "CPU", ATTENTION_BACKEND=backend)
    start_generate = perf_counter_ns()
    load_time = (start_generate - start_time) / 1_000_000.0
    
    result = pipe.generate(
        PROMPTS[0],
        images=[cat_tensor],
        generation_config=GenerationConfig(max_new_tokens=max_new_tokens),
    )
    generate_time = (perf_counter_ns() - start_generate) / 1_000_000.0

    perf_metrics = result.perf_metrics

    assert perf_metrics is not None

    assert 0 < perf_metrics.get_load_time() < load_time
    num_tokens = perf_metrics.get_num_generated_tokens()
    assert 0 < num_tokens <= max_new_tokens
    assert 0 < perf_metrics.get_num_input_tokens() < len(PROMPTS[0]) + IMAGE_TOKENS_NUM
    assert 0 < perf_metrics.get_ttft().mean < generate_time
    assert 0 < perf_metrics.get_tpot().mean < generate_time / num_tokens
    assert 0 < perf_metrics.get_ipot().mean < generate_time / num_tokens
    assert (num_tokens - 1) / (
        (generate_time - perf_metrics.get_ttft().mean) / 1000.0
    ) < perf_metrics.get_throughput().mean

    assert 0 < perf_metrics.get_inference_duration().mean < generate_time
    assert 0 < perf_metrics.get_generate_duration().mean < generate_time
    assert 0 < perf_metrics.get_tokenization_duration().mean < generate_time
    assert 0 < perf_metrics.get_detokenization_duration().mean < generate_time
    assert 0 < perf_metrics.get_prepare_embeddings_duration().mean < generate_time

    squared_generate_time = generate_time * generate_time
    assert 0 <= perf_metrics.get_ttft().std < squared_generate_time
    assert 0 <= perf_metrics.get_tpot().std < squared_generate_time
    assert 0 <= perf_metrics.get_ipot().std < squared_generate_time
    assert 0 <= perf_metrics.get_throughput().std < squared_generate_time
    assert 0 <= perf_metrics.get_inference_duration().std < squared_generate_time
    assert 0 <= perf_metrics.get_generate_duration().std < squared_generate_time
    assert 0 <= perf_metrics.get_tokenization_duration().std < squared_generate_time
    assert 0 <= perf_metrics.get_detokenization_duration().std < squared_generate_time
    assert (
        0 <= perf_metrics.get_prepare_embeddings_duration().std < squared_generate_time
    )

    # assert that calculating statistics manually from the raw counters we get the same results as from PerfMetrics
    vlm_raw_metrics = perf_metrics.vlm_raw_metrics

    raw_dur = np.array(vlm_raw_metrics.prepare_embeddings_durations) / 1000.0
    mean_dur, std_dur = perf_metrics.get_prepare_embeddings_duration()
    assert np.allclose(mean_dur, np.mean(raw_dur))
    assert np.allclose(std_dur, np.std(raw_dur))


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_npu_no_exception(model_id, backend, cat_tensor, handwritten_tensor, car_tensor):
    unsupported_models = {
        "katuni4ka/tiny-random-internvl2",
        "katuni4ka/tiny-random-gemma3",
    }

    if model_id in unsupported_models:
        pytest.skip(f"{model_id} is not supported")

    models_path = _get_ov_model(model_id)
    properties = {
        "DEVICE_PROPERTIES": {
            "NPU": {"NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE", "MAX_PROMPT_LEN": 2048}
        }
    }

    ov_pipe = VLMPipeline(models_path, "NPU", ATTENTION_BACKEND=backend, config=properties)

    generation_config = _setup_generation_config(ov_pipe)

    for image in cat_tensor, handwritten_tensor, car_tensor:
        ov_pipe.generate(
            PROMPTS[0], images=[image], generation_config=generation_config
        )


@pytest.fixture(scope="module", params=[
    pytest.param(["cat_tensor"], id="cat_tensor - one image"), 
    pytest.param([], id="empty"),
])
def image_sequence(request):
    return [request.getfixturevalue(image) for image in request.param]


@pytest.mark.precommit
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_npu_no_image():
    models_path = _get_ov_model(MODEL_IDS[0])
    properties = {
        "DEVICE_PROPERTIES": {
            "NPU": {"NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE", "MAX_PROMPT_LEN": 2048}
        }
    }

    ov_pipe = VLMPipeline(models_path, "NPU", config=properties)

    generation_config = _setup_generation_config(ov_pipe)

    ov_pipe.generate(
        PROMPTS[0], generation_config=generation_config
    )


@pytest.mark.precommit
@parametrize_all_models
def test_vlm_pipeline_chat_streamer_cancel_second_generate(
    request: pytest.FixtureRequest,
    ov_pipe_model: VlmModelInfo, 
    image_sequence: list[openvino.Tensor]
):
    ov_pipe = ov_pipe_model.pipeline
    callback_questions = [
        "Explain in details 1+1=",
        "Why is the Sun yellow?",
        "What is the previous answer?",
    ]

    current_iter = 0
    num_iters = 1

    def streamer(subword):
        nonlocal current_iter
        current_iter += 1
        return (
            StreamingStatus.CANCEL
            if current_iter == num_iters
            else StreamingStatus.RUNNING
        )

    generation_config = _setup_generation_config(ov_pipe, ignore_eos=True, do_sample=False)

    images_and_videos = {"images": image_sequence}
    if ov_pipe_model.model_id in VIDEO_MODEL_IDS:
        video = request.getfixturevalue("synthetic_video_32x32_tensor")
        images_and_videos["videos"] = video

    results_with_cancel = ""
    ov_pipe.start_chat()
    results_with_cancel += ov_pipe.generate(
        callback_questions[0], **images_and_videos, generation_config=generation_config
    ).texts[0]
    # doesn't add to results_with_cancel as it should be complitely removed from the history
    ov_pipe.generate(
        callback_questions[1],
        images=image_sequence,
        generation_config=generation_config,
        streamer=streamer,
    )
    results_with_cancel += ov_pipe.generate(
        callback_questions[2], **images_and_videos, generation_config=generation_config
    ).texts[0]
    ov_pipe.finish_chat()

    results = ""
    ov_pipe.start_chat()
    results += ov_pipe.generate(
        callback_questions[0], **images_and_videos, generation_config=generation_config
    ).texts[0]
    results += ov_pipe.generate(
        callback_questions[2], **images_and_videos, generation_config=generation_config
    ).texts[0]
    ov_pipe.finish_chat()

    assert results_with_cancel == results

    results = ""
    ov_pipe.start_chat()
    results += ov_pipe.generate(
        callback_questions[0], **images_and_videos, generation_config=generation_config
    ).texts[0]
    results += ov_pipe.generate(
        callback_questions[2], **images_and_videos, generation_config=generation_config
    ).texts[0]
    ov_pipe.finish_chat()

    assert results_with_cancel == results


@pytest.mark.precommit
@parametrize_one_model_backends
def test_start_chat_clears_history(
    ov_pipe_model: VlmModelInfo,
    image_sequence: list[openvino.Tensor], 
):
    ov_pipe = ov_pipe_model.pipeline
    callback_questions = [
        "Why is the Sun yellow?"
    ]
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    ov_pipe.start_chat()
    results_first_generate = ov_pipe.generate(
        callback_questions[0], images=image_sequence, generation_config=generation_config
    ).texts[0]

    ov_pipe.start_chat()
    results_second_generate = ov_pipe.generate(
        callback_questions[0], images=image_sequence, generation_config=generation_config
    ).texts[0]
    ov_pipe.finish_chat()

    assert results_first_generate == results_second_generate


@pytest.mark.precommit
def test_start_chat_clears_history_cb_api(
    ov_continious_batching_pipe: ContinuousBatchingPipeline, 
    image_sequence: list[openvino.Tensor]
):
    callback_questions = [
        "Why is the Sun yellow?"
    ]
    generation_config = GenerationConfig(max_new_tokens=DEFAULT_MAX_NEW_TOKENS)

    results_first_generate = ""
    ov_continious_batching_pipe.start_chat("You are helpful assistant.")
    results_first_generate = ov_continious_batching_pipe.generate(
        [callback_questions[0]], images=[image_sequence], generation_config=[generation_config]
    )[0].texts[0]

    results_second_generate = ""
    ov_continious_batching_pipe.start_chat("You are helpful assistant.")
    results_second_generate += ov_continious_batching_pipe.generate(
        [callback_questions[0]], images=[image_sequence], generation_config=[generation_config]
    )[0].texts[0]
    ov_continious_batching_pipe.finish_chat()

    assert results_first_generate == results_second_generate


@pytest.mark.precommit
@parametrize_all_models
def test_vlm_pipeline_chat_streamer_cancel_first_generate(
    request: pytest.FixtureRequest,
    ov_pipe_model: VlmModelInfo, 
    image_sequence: list[openvino.Tensor], 
):    
    if "phi" in ov_pipe_model.model_id and ov_pipe_model.ov_backend == "SDPA":
        pytest.skip("SDPA is failing for phi models on VLM model reusing")
    
    ov_pipe = ov_pipe_model.pipeline
    callback_questions = [
        "Why is the Sun yellow?",
        "1+1=",
    ]

    current_iter = 0
    num_iters = 3
    streamer_generation_result = ""

    def streamer(subword):
        nonlocal current_iter
        nonlocal streamer_generation_result

        current_iter += 1
        streamer_generation_result += subword
        return (
            StreamingStatus.CANCEL
            if current_iter == num_iters
            else StreamingStatus.RUNNING
        )

    generation_config = _setup_generation_config(
        ov_pipe, ignore_eos=True, do_sample=False
    )

    images_and_videos = {"images": image_sequence}
    if ov_pipe_model.model_id in VIDEO_MODEL_IDS:
        video = request.getfixturevalue("synthetic_video_32x32_tensor")
        images_and_videos["videos"] = video

    ov_pipe.start_chat()
    ov_pipe.generate(
        callback_questions[0],
        **images_and_videos,
        generation_config=generation_config,
        streamer=streamer,
    )
    res_first = streamer_generation_result
    current_iter = 0
    streamer_generation_result = ""
    ov_pipe.generate(
        callback_questions[0],
        **images_and_videos,
        generation_config=generation_config,
        streamer=streamer,
    )
    ov_pipe.finish_chat()
    res_second = streamer_generation_result

    assert res_first == res_second


def retry(func, exception_type=AssertionError):
    __tracebackhide__ = True
    max_retries = 20
    for idx in range(max_retries):
        try:
            return func()
        except exception_type:
            if idx == max_retries - 1:
                raise


def generate(vlm: VLMPipeline, requests):
    generation_config = _setup_generation_config(vlm, set_eos_token=False)
    vlm.set_generation_config(generation_config)
    vlm.start_chat()
    answers = [vlm.generate(prompt, images=images, do_sample=False) for (prompt, images) in requests]
    vlm.finish_chat()
    return answers


@pytest.fixture(scope="module")
def conversation_requests(
    cat_tensor: openvino.Tensor, 
    car_tensor: openvino.Tensor, 
    handwritten_tensor: openvino.Tensor
) -> list[tuple[str, list[openvino.Tensor]]]:
    return [
        ("Describe", [cat_tensor]),
        ("How many images are there?", [car_tensor, handwritten_tensor]),
    ]


TAG_INSERTED_BY_TEMPLATE = [
    ("katuni4ka/tiny-random-llava", "PA"),
    ("katuni4ka/tiny-random-llava-next", "PA"),
    ("katuni4ka/tiny-random-qwen2vl", "PA"),
    ("katuni4ka/tiny-random-qwen2.5-vl", "PA"),
    ("katuni4ka/tiny-random-gemma3", "SDPA"),
    ("qnguyen3/nanoLLaVA", "PA"),
    ("katuni4ka/tiny-random-llava-next-video", "PA"),
]


IMAGE_ID_IGNORANT_MODELS_TO_TAG = TAG_INSERTED_BY_TEMPLATE + [
    ("katuni4ka/tiny-random-internvl2", "PA"),
]


MODELS_TO_TAG = IMAGE_ID_IGNORANT_MODELS_TO_TAG + [
    ("katuni4ka/tiny-random-minicpmv-2_6", "PA"),
    ("katuni4ka/tiny-random-phi3-vision", "PA"),
]


def model_and_tag_parametrize(
    items: tuple[str, str, Callable[[int], str]] | None = None
) -> Callable[[Callable], Generator]:
    if items is None:
        items = MODELS_TO_TAG

    return pytest.mark.parametrize(
        "ov_pipe_model",
        items,
        indirect=["ov_pipe_model"],
        ids=[f"{item[0]}/{item[1]}" for item in items]
    )


@pytest.mark.precommit
@model_and_tag_parametrize(TAG_INSERTED_BY_TEMPLATE)
def test_model_tags_representation(ov_pipe_model: VlmModelInfo, cat_tensor: openvino.Tensor):
    ov_pipe = ov_pipe_model.pipeline
    model_id = ov_pipe_model.model_id
    
    generation_config = _setup_generation_config(ov_pipe, set_eos_token=False)
    ov_pipe.set_generation_config(generation_config)
    prompt = "Describe"

    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    if model_id == "qnguyen3/nanoLLaVA":
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        messages = [
            {"role": "user", "content": f'<image>\n{prompt}'}
        ]
        templated_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        processor = retry_request(
            lambda: transformers.AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                **align_with_optimum_cli,
            )
        )
        templated_prompt = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        automatic_tags = ov_pipe.generate(prompt, images=[cat_tensor], do_sample=False)
        reference_tags = ov_pipe.generate(
            templated_prompt, images=[cat_tensor], apply_chat_template=False, do_sample=False
        )
        assert automatic_tags.texts == reference_tags.texts
        assert automatic_tags.scores == reference_tags.scores

    retry(workaround_inconsistent_inference)


@pytest.mark.precommit
@model_and_tag_parametrize()
def test_model_tags_prepend_native(
    ov_pipe_model: VlmModelInfo, 
    conversation_requests: list[tuple[str, list[openvino.Tensor]]]
):
    ov_pipe = ov_pipe_model.pipeline
    tag = ov_pipe_model.image_tag
    
    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        answers = generate(ov_pipe, conversation_requests)

        ov_pipe.start_chat()
        native_tag0 = ov_pipe.generate(
            tag(0) + conversation_requests[0][0], images=conversation_requests[0][1], do_sample=False
        )
        assert native_tag0.texts == answers[0].texts
        assert native_tag0.scores == answers[0].scores
        native_tags1 = ov_pipe.generate(
            tag(1) + tag(2) + conversation_requests[1][0], images=conversation_requests[1][1], do_sample=False
        )
        assert native_tags1.texts == answers[1].texts
        assert native_tags1.scores == answers[1].scores
        ov_pipe.finish_chat()

    retry(workaround_inconsistent_inference)


@pytest.mark.precommit
@model_and_tag_parametrize()
def test_model_tags_prepend_universal(
    ov_pipe_model: VlmModelInfo, 
    conversation_requests: list[tuple[str, list[openvino.Tensor]]]
):
    ov_pipe = ov_pipe_model.pipeline
    
    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        answers = generate(ov_pipe, conversation_requests)

        ov_pipe.start_chat()
        universal_tag0 = ov_pipe.generate(
            "<ov_genai_image_0>" + conversation_requests[0][0], images=conversation_requests[0][1], do_sample=False
        )
        assert universal_tag0.texts == answers[0].texts
        assert universal_tag0.scores == answers[0].scores
        universal_tags1 = ov_pipe.generate(
            "<ov_genai_image_1><ov_genai_image_2>" + conversation_requests[1][0],
            images=conversation_requests[1][1],
            do_sample=False
        )
        assert universal_tags1.texts == answers[1].texts
        assert universal_tags1.scores == answers[1].scores
        ov_pipe.finish_chat()

    retry(workaround_inconsistent_inference)

@pytest.fixture(scope="module")
def cat_image_384x384(cat_image):
    return cat_image.resize((384, 384))

@pytest.mark.precommit
@model_and_tag_parametrize()
def test_model_tags_append(
    ov_pipe_model: VlmModelInfo, 
    conversation_requests: list[tuple[str, list[openvino.Tensor]]]
):
    ov_pipe = ov_pipe_model.pipeline
    tag = ov_pipe_model.image_tag
    
    generation_config = _setup_generation_config(ov_pipe, set_eos_token=False)
    ov_pipe.set_generation_config(generation_config)

    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        ov_pipe.start_chat()
        native_tag0 = ov_pipe.generate(
            conversation_requests[0][0] + tag(0), images=conversation_requests[0][1], do_sample=False
        )
        native_tags1 = ov_pipe.generate(
            conversation_requests[1][0] + tag(1) + tag(2), images=conversation_requests[1][1], do_sample=False
        )
        ov_pipe.finish_chat()

        ov_pipe.start_chat()
        universal_tag0 = ov_pipe.generate(
            conversation_requests[0][0] + "<ov_genai_image_0>", images=conversation_requests[0][1], do_sample=False
        )
        assert universal_tag0.texts == native_tag0.texts
        assert universal_tag0.scores == native_tag0.scores
        universal_tags1 = ov_pipe.generate(
            conversation_requests[1][0] + "<ov_genai_image_1><ov_genai_image_2>",
            images=conversation_requests[1][1],
            do_sample=False
        )
        assert universal_tags1.texts == native_tags1.texts
        assert universal_tags1.scores == native_tags1.scores
        ov_pipe.finish_chat()

    retry(workaround_inconsistent_inference)


@pytest.mark.precommit
@model_and_tag_parametrize(IMAGE_ID_IGNORANT_MODELS_TO_TAG)
def test_model_tags_same_reference(ov_pipe_model: VlmModelInfo, cat_tensor: openvino.Tensor):
    ov_pipe = ov_pipe_model.pipeline
    
    generation_config = _setup_generation_config(ov_pipe, max_new_tokens=2, set_eos_token=False)
    ov_pipe.set_generation_config(generation_config)
    
    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        one_image = ov_pipe.generate("<ov_genai_image_0>" * 2, images=[cat_tensor], do_sample=False)
        two_images = ov_pipe.generate(
            "<ov_genai_image_0><ov_genai_image_1>", images=[cat_tensor, cat_tensor], do_sample=False
        )
        assert one_image.texts == two_images.texts
        assert one_image.scores == two_images.scores

    retry(workaround_inconsistent_inference)


@pytest.mark.precommit
@model_and_tag_parametrize()
def test_model_tags_older(ov_pipe_model: VlmModelInfo, car_tensor: openvino.Tensor):
    ov_pipe = ov_pipe_model.pipeline
    
    generation_config = _setup_generation_config(ov_pipe, set_eos_token=False)
    ov_pipe.set_generation_config(generation_config)
    ov_pipe.start_chat()
    ov_pipe.generate("", images=[car_tensor])
    with pytest.raises(RuntimeError):
        ov_pipe.generate("<ov_genai_image_0>", images=[car_tensor])
    ov_pipe.finish_chat()
        
        
@pytest.mark.precommit
@model_and_tag_parametrize()
def test_model_tags_missing_universal(ov_pipe_model: VlmModelInfo):
    ov_pipe = ov_pipe_model.pipeline
    
    with pytest.raises(RuntimeError):
        ov_pipe.generate("<ov_genai_image_0>")
        
        
@pytest.mark.precommit
@model_and_tag_parametrize()
def test_model_tags_missing_native(ov_pipe_model: VlmModelInfo):
    ov_pipe = ov_pipe_model.pipeline
    image_tag = ov_pipe_model.image_tag
    
    with pytest.raises(RuntimeError):
        ov_pipe.generate(image_tag(0))
            

@pytest.mark.precommit
@pytest.mark.parametrize(
    "ov_pipe_model",
    [
        pytest.param(("katuni4ka/tiny-random-qwen2vl","SDPA")),
        pytest.param(("katuni4ka/tiny-random-qwen2vl", "PA")),
        pytest.param(("katuni4ka/tiny-random-qwen2.5-vl", "SDPA")),
        pytest.param(("katuni4ka/tiny-random-qwen2.5-vl", "PA"), marks=pytest.mark.xfail(reason="CVS-167316")),
        (
            pytest.param(("katuni4ka/tiny-random-gemma3", "SDPA"), marks=pytest.mark.xfail(reason=GEMMA3_MACOS_XFAIL_REASON)) 
            if sys.platform == "darwin" 
            else pytest.param(("katuni4ka/tiny-random-gemma3",  "SDPA"))
        ),
        pytest.param(("katuni4ka/tiny-random-gemma3", "PA"), marks=pytest.mark.xfail(reason="CVS-171180")),
        pytest.param(("qnguyen3/nanoLLaVA", "SDPA")),
        pytest.param(("qnguyen3/nanoLLaVA", "PA")),
        pytest.param(("katuni4ka/tiny-random-llava-next-video", "SDPA")),
        pytest.param(("katuni4ka/tiny-random-llava-next-video", "PA")),
    ],
    ids=lambda p: f"{p[0]}/{p[1]}",
    indirect=["ov_pipe_model"],
)
def test_vlm_pipeline_match_optimum_preresized(request, ov_pipe_model: VlmModelInfo):
    class NanollavaProcessorWrapper:
        def __init__(self, processor, config, model_dtype):
            self.processor = processor
            self.config = config
            self.model_dtype = model_dtype

        def __call__(self, images, return_tensors):
            return {"pixel_values": self.processor(images, self.config).to(dtype=self.model_dtype)}

    def get_nanollava_processor():
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='auto',
            trust_remote_code=True)
        return NanollavaProcessorWrapper(hf_model.process_images, hf_model.config, hf_model.dtype)
    
    ov_pipe = ov_pipe_model.pipeline
    model_id = ov_pipe_model.model_id
    resolution = ov_pipe_model.resolution
    

    prompt = "Describe this image."
    resized_image = None
    resized_video = None
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text"},
            ],
        }
    ]
    is_video_model = ov_pipe_model.model_id in VIDEO_MODEL_IDS
    if is_video_model:
        prompt = "Describe this video."
        resized_video = request.getfixturevalue("synthetic_video_32x32")
        conversation[0]["content"] = [{"type": "video"}] + conversation[0]["content"]

    resized_image = request.getfixturevalue(f"cat_image_{resolution}x{resolution}")
    if is_video_model:
        prompt = "Describe this image and video."
    else:
        prompt = "Describe this image."
    conversation[0]["content"] = [{"type": "image"}] + conversation[0]["content"]

    conversation[0]["content"][-1]["text"] = prompt

    model_path = _get_ov_model(model_id)

    # Run the model with optimum-intel
    model = OVModelForVisualCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = None
    if model.config.model_type == "llava-qwen2":
        processor = get_nanollava_processor()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING
        preprocess_inputs = MODEL_TYPE_TO_CLS_MAPPING[model.config.model_type].preprocess_inputs
        inputs = preprocess_inputs(prompt, resized_image, processor, tokenizer, config=model.config)
    else:
        processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Gemma3 input_ids has two bos tokens when running with optimum: one in chat template + "add_bos_token" is set to True in tokenizer_config.json
        if model.config.model_type == "gemma3":
            processor.tokenizer.add_bos_token = False
        params = {}
        if resized_image is not None:
            params["images"] = [resized_image]
        if resized_video is not None:
            params["videos"] = [resized_video]
        templated_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[templated_prompt], **params, padding=True, return_tensors="pt")

    max_new_tokens = 100

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, output_ids)]

    if model.config.model_type == "llava-qwen2":
        assert tokenizer is not None, "Tokenizer should be set for llava-qwen2 models."
        optimum_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    else:
        optimum_output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        optimum_text = optimum_output[0]

    params = {}
    if resized_image is not None:
        params["images"] = [openvino.Tensor(resized_image)]
    if resized_video is not None:
        params["videos"] = [openvino.Tensor(resized_video)]

    genai_output = ov_pipe.generate(prompt, **params, max_new_tokens=max_new_tokens, do_sample=False)
    genai_text = genai_output.texts[0]

    assert optimum_text == genai_text
