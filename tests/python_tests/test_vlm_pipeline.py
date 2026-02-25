# Copyright (C) 2018-2026 Intel Corporation
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
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
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
from optimum.utils.import_utils import is_transformers_version
from huggingface_hub import snapshot_download
from openvino_genai import (
    VLMPipeline,
    GenerationConfig,
    SchedulerConfig,
    ContinuousBatchingPipeline,
    GenerationStatus,
    StreamingStatus,
    GenerationFinishReason,
    ChatHistory,
)

from utils.network import retry_request
from utils.generation_config import (
    get_beam_search,
    get_multinomial_all_parameters,
    get_greedy,
)
from utils.constants import get_ov_cache_converted_models_dir
from utils.atomic_download import AtomicDownloadManager

import logging
logger = logging.getLogger(__name__)


class VisionType(Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"


@dataclass(frozen=True)
class VlmModelInfo:
    model_id: str
    ov_backend: str
    image_tag: Callable[[int], str]
    video_tag: Callable[[int], str]
    resolution: int
    pipeline: VLMPipeline

    def get_vision_tag(self, vision_type: VisionType) -> Callable[[int], str]:
        return self.image_tag if vision_type == VisionType.IMAGE else self.video_tag


PROMPTS: list[str] = [
    "What is in the image?",
    "What is special about this image?",
    "Describe the image"
]


VIDEO_MODEL_IDS = [
    "optimum-intel-internal-testing/tiny-random-llava-next-video",
    "optimum-intel-internal-testing/tiny-random-qwen2vl",
    "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
]


MODEL_IDS: list[str] = [
    "optimum-intel-internal-testing/tiny-random-minicpmv-2_6",
    "optimum-intel-internal-testing/tiny-random-phi3-vision",
    "optimum-intel-internal-testing/tiny-random-phi-4-multimodal",
    "optimum-intel-internal-testing/tiny-random-llava",
    "optimum-intel-internal-testing/tiny-random-llava-next",
    "optimum-intel-internal-testing/tiny-random-internvl2",
    "optimum-intel-internal-testing/tiny-random-gemma3",
    "qnguyen3/nanoLLaVA",
    "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6",
    *VIDEO_MODEL_IDS,
]


ADD_REQUEST_MODEL_IDS = [
    MODEL_IDS[0],
    *VIDEO_MODEL_IDS
]


IMAGE_TAG_GENERATOR_BY_MODEL: dict[str, Callable[[int], str]] = {
    "optimum-intel-internal-testing/tiny-random-llava": lambda idx: "<image>",
    "optimum-intel-internal-testing/tiny-random-llava-next": lambda idx: "<image>",
    "optimum-intel-internal-testing/tiny-random-qwen2vl": lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
    "optimum-intel-internal-testing/tiny-random-qwen2.5-vl": lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
    "optimum-intel-internal-testing/tiny-random-gemma3": lambda idx: "<start_of_image>",
    "optimum-intel-internal-testing/tiny-random-internvl2": lambda idx: "<image>\n",
    "optimum-intel-internal-testing/tiny-random-minicpmv-2_6": lambda idx: "<image>./</image>\n",
    "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6": lambda idx: "<image>./</image>\n",
    "optimum-intel-internal-testing/tiny-random-phi3-vision": lambda idx: f"<|image_{idx + 1}|>\n",
    "optimum-intel-internal-testing/tiny-random-llava-next-video": lambda idx: "<image>\n",
    "qnguyen3/nanoLLaVA": lambda idx: "<image>\n",
}

VIDEO_TAG_GENERATOR_BY_MODEL: dict[str, Callable[[int], str]] = {
    "optimum-intel-internal-testing/tiny-random-llava-next-video": lambda idx: "<video>",
    "optimum-intel-internal-testing/tiny-random-qwen2vl": lambda idx: "<|vision_start|><|video_pad|><|vision_end|>",
    "optimum-intel-internal-testing/tiny-random-qwen2.5-vl": lambda idx: "<|vision_start|><|video_pad|><|vision_end|>",
}


RESOLUTION_BY_MODEL: dict[str, int | None] = {
    "optimum-intel-internal-testing/tiny-random-gemma3": 32,
    "qnguyen3/nanoLLaVA": 384,
    "optimum-intel-internal-testing/tiny-random-llava-next-video": 336,
    "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6": 448,
    "optimum-intel-internal-testing/tiny-random-qwen2vl": 336,
    "optimum-intel-internal-testing/tiny-random-qwen2.5-vl": 336,
}


RESOLUTION_BY_VIDEO_MODEL: dict[str, int | None] = {
    "optimum-intel-internal-testing/tiny-random-llava-next-video": 32,
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


NPU_UNSUPPORTED_MODELS = {
    "optimum-intel-internal-testing/tiny-random-internvl2",
}

DEFAULT_NPUW_PROPERTIES = {
    "DEVICE_PROPERTIES": {"NPU": {"NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE", "MAX_PROMPT_LEN": 4096}}
}

NPU_SUPPORTED_MODELS = [id for id in MODEL_IDS if id not in NPU_UNSUPPORTED_MODELS and id not in VIDEO_MODEL_IDS]

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
    if model_id in {"optimum-intel-internal-testing/tiny-random-phi-4-multimodal", "qnguyen3/nanoLLaVA"}:
        pytest.skip("ValueError: The current version of Transformers does not allow for the export of the model. Maximum required is 4.53.3, got: 4.55.4")
    if "optimum-intel-internal-testing/tiny-random-phi3-vision" == model_id:
        pytest.xfail("AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'. Ticket CVS-175110")
    if "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6" == model_id and is_transformers_version(
        ">", "4.51.3"
    ):
        pytest.skip(
            "ValueError: The current version of Transformers does not allow for the export of the model. Maximum supported version is 4.51.3"
        )

    ov_cache_converted_dir = get_ov_cache_converted_models_dir()
    dir_name = str(model_id).replace(os.sep, "_")
    model_dir = ov_cache_converted_dir / dir_name

    manager = AtomicDownloadManager(model_dir)

    if manager.is_complete() or (model_dir / "openvino_language_model.xml").exists():
        return model_dir

    def convert_to_temp(temp_dir: Path) -> None:
        model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
        align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
        processor = retry_request(
            lambda: transformers.AutoProcessor.from_pretrained(
                model_cached,
                trust_remote_code=True,
                **align_with_optimum_cli,
            )
        )
        model = retry_request(
            lambda: OVModelForVisualCausalLM.from_pretrained(
                model_cached,
                compile=False,
                device="CPU",
                export=True,
                load_in_8bit=False,
                trust_remote_code=model_id in {
                    "optimum-intel-internal-testing/tiny-random-minicpmv-2_6",
                    "optimum-intel-internal-testing/tiny-random-internvl2",
                    "optimum-intel-internal-testing/tiny-random-phi3-vision",
                    "optimum-intel-internal-testing/tiny-random-phi-4-multimodal",
                    "qnguyen3/nanoLLaVA",
                    "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6",
                },
            )
        )
        if model.config.model_type == "llava-qwen2":
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True)
        # For tiny-random-internvl2 processor is actually tokenizer
        elif isinstance(processor, transformers.Qwen2TokenizerFast):
            tokenizer = processor
            processor = transformers.AutoImageProcessor.from_pretrained(model_cached, trust_remote_code=True)
        else:
            tokenizer = processor.tokenizer
            if tokenizer.chat_template is None:
                tokenizer.chat_template = processor.chat_template
        tokenizer.save_pretrained(temp_dir)
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
            tokenizer, with_detokenizer=True
        )
        openvino.save_model(ov_tokenizer, temp_dir / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, temp_dir / "openvino_detokenizer.xml")

        if tokenizer.chat_template is not None and model.config.model_type == "phi3_v":
            # It seems that tiny-random-phi3-vision is saved incorrectly. That line works this around.
            processor.chat_template = tokenizer.chat_template
        processor.audio_tokenizer = None
        processor.save_pretrained(temp_dir)
        model.save_pretrained(temp_dir)

    manager.execute(convert_to_temp)
    return model_dir


# On macOS, transformers<4.52 is required, but this causes gemma3 to fail
GEMMA3_MACOS_XFAIL_REASON = "gemma3 not supported on macOS with older transformers"


@pytest.fixture(scope="module")
def ov_pipe_model(request: pytest.FixtureRequest) -> VlmModelInfo:
    if not (2 <= len(request.param) <= 3):
        raise ValueError("expected request.param must be a tuple of length 2 or 3")
    ov_model, ov_backend = request.param[:2]
    preprocess_method = request.param[2] if len(request.param) == 3 else None

    if sys.platform == "darwin" and "gemma3" in ov_model:
        pytest.xfail(GEMMA3_MACOS_XFAIL_REASON)

    models_path = _get_ov_model(ov_model)

    vision_preprocess_env_set = False
    key = "VISION_PREPROCESS"
    if preprocess_method == "CPP":
        # If environment is already set, don't override it.
        if key not in os.environ:
            os.environ[key] = "CPP"
            vision_preprocess_env_set = True

    try:
        pipeline = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=ov_backend)
    finally:
        if vision_preprocess_env_set:
            os.environ.pop(key, None)
    return VlmModelInfo(
        ov_model,
        ov_backend,
        IMAGE_TAG_GENERATOR_BY_MODEL.get(ov_model, lambda idx: ""),
        VIDEO_TAG_GENERATOR_BY_MODEL.get(ov_model, lambda idx: ""),
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


def _dict_to_sorted_tuple(d):
    if isinstance(d, dict):
        return tuple([(key, _dict_to_sorted_tuple(value)) for key, value in sorted(d.items())])

    return d


def _sorted_tuple_to_dict(t):
    if isinstance(t, tuple):
        return {key: _sorted_tuple_to_dict(value) for key, value in t}

    return t


@pytest.fixture(scope="module")
def ov_npu_pipe_model(request: pytest.FixtureRequest) -> VlmModelInfo:
    ov_model, config = request.param

    if sys.platform == "darwin" and "gemma3" in ov_model:
        pytest.xfail(GEMMA3_MACOS_XFAIL_REASON)

    models_path = _get_ov_model(ov_model)

    pipeline = VLMPipeline(models_path, "NPU", config=_sorted_tuple_to_dict(config))
    return VlmModelInfo(
        ov_model,
        "SDPA",
        IMAGE_TAG_GENERATOR_BY_MODEL.get(ov_model, lambda idx: ""),
        VIDEO_TAG_GENERATOR_BY_MODEL.get(ov_model, lambda idx: ""),
        RESOLUTION_BY_MODEL.get(ov_model, DEFAULT_RESOLUTION),
        pipeline,
    )


def _parametrize_npu_models(models: str | list[str], config: dict | None = None, config_name: str | None = None):
    if isinstance(models, str):
        models = [models]

    assert (config is None and config_name is None) or (config is not None and config_name is not None)
    if config is not None:
        config = _dict_to_sorted_tuple(config)

    params = [(model, config) for model in models]

    return pytest.mark.parametrize(
        "ov_npu_pipe_model",
        params,
        ids=lambda p: f"{p[0]}/{config_name}",
        indirect=["ov_npu_pipe_model"],
    )


parametrize_all_models_npu = _parametrize_npu_models(
    NPU_SUPPORTED_MODELS, DEFAULT_NPUW_PROPERTIES, "DEFAULT_NPUW_PROPERTIES"
)
parametrize_one_model_npu = _parametrize_npu_models(
    NPU_SUPPORTED_MODELS[0], DEFAULT_NPUW_PROPERTIES, "DEFAULT_NPUW_PROPERTIES"
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
def cat_image_448x448(cat_image):
    return cat_image.resize((448, 448))


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


@parametrize_one_model_sdpa
def test_vlm_readonly_image_tensor(ov_pipe_model: VlmModelInfo, cat_image_32x32):
    ov_pipe = ov_pipe_model.pipeline
    generation_config = _setup_generation_config(ov_pipe, max_new_tokens=5)

    image_array = np.array(cat_image_32x32, dtype=np.uint8)
    image_array.flags.writeable = False

    readonly_image_tensor = openvino.Tensor(image_array)
    ov_pipe.generate(
        PROMPTS[0],
        images=[readonly_image_tensor],
        generation_config=generation_config,
    )


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
            idx,
            PROMPTS[0],
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


@parametrize_one_model_sdpa
def test_vlm_continuous_batching_vs_stateful_chat_history(
    ov_pipe_model: VlmModelInfo,
    ov_continious_batching_pipe: ContinuousBatchingPipeline,
    cat_tensor: openvino.Tensor,
    car_tensor: openvino.Tensor,
):
    ov_pipe = ov_pipe_model.pipeline
    generation_config = get_greedy()
    image_links_list = [[cat_tensor], [car_tensor]]

    histories_batch = 2
    histories_cb = []
    histories_stateful = []

    for i in range(histories_batch):
        histories_cb.append(ChatHistory())
        histories_stateful.append(ChatHistory())

    # Continuous batching generation
    results_cb = []
    for images in image_links_list:
        for i in range(histories_batch):
            histories_cb[i].append({"role": "user", "content": PROMPTS[i]})
        results = ov_continious_batching_pipe.generate(
            histories_cb,
            images=[images for _ in range(histories_batch)],
            generation_config=[generation_config for _ in range(histories_batch)],
        )
        for i in range(histories_batch):
            histories_cb[i].append({"role": "assistant", "content": results[i].texts[0]})
        results_cb.append(results)

    # Stateful generation + comparison
    for i in range(histories_batch):
        for q_i, images in enumerate(image_links_list):
            histories_stateful[i].append({"role": "user", "content": PROMPTS[i]})
            result_stateful = ov_pipe.generate(
                histories_stateful[i], images=images, generation_config=generation_config
            )
            histories_stateful[i].append({"role": "assistant", "content": result_stateful.texts[0]})
            for out_idx, text in enumerate(result_stateful.texts):
                assert text == results_cb[q_i][i].texts[out_idx]
                assert abs(result_stateful.scores[out_idx] - results_cb[q_i][i].scores[out_idx]) < DEFAULT_SCORE_EPSILON


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
        [[["cat_tensor", "car_tensor", "handwritten_tensor"], []], [["synthetic_video_32x32_tensor", "synthetic_video_32x32_tensor"], ["synthetic_video_32x32_tensor"]]],
        id="3 images + 2 videos on first iteration, video on second iteration"
    ),
])
def iteration_images_and_videos(request):
    params = []
    for param in request.param:
        params.append([[request.getfixturevalue(image) for image in bundle] for bundle in param])
    return params


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


@parametrize_all_models
def test_vlm_pipeline_start_chat_vs_chat_history(
    ov_pipe_model: VlmModelInfo,
    iteration_images: list[list[PIL.Image]],
):
    ov_pipe = ov_pipe_model.pipeline

    generation_config = _setup_generation_config(ov_pipe, do_sample=False)

    prompts_with_images = [
        (PROMPTS[0], iteration_images[0]),
        *[(PROMPTS[1], image_set) for image_set in iteration_images[1:]],
    ]

    # Collect chat_history results
    answers_chat_history = []
    history = ChatHistory()
    for prompt, images in prompts_with_images:
        history.append({"role": "user", "content": prompt})
        messages_before = history.get_messages()
        res = ov_pipe.generate(
            history,
            images=images,
            generation_config=generation_config,
        )
        messages_after = history.get_messages()
        assert messages_before == messages_after, "ChatHistory messages should not be mutated after generate."
        answer = res.texts[0]
        history.append({"role": "assistant", "content": answer})
        answers_chat_history.append(answer)

    # Collect start_chat results
    answers_start_chat = []
    ov_pipe.start_chat()
    for prompt, images in prompts_with_images:
        res = ov_pipe.generate(
            prompt,
            images=images,
            generation_config=generation_config,
        )
        answers_start_chat.append(res.texts[0])
    ov_pipe.finish_chat()

    for i, (answer_start_chat, answer_chat_history) in enumerate(zip(answers_start_chat, answers_chat_history)):
        assert answer_start_chat == answer_chat_history, (
            f"Answer {i} does not match!\n"
            f"answer_start_chat: {answer_start_chat}\n"
            f"answer_chat_history: {answer_chat_history}"
        )


@pytest.mark.parametrize(
    "ov_pipe_model",
    [
        pytest.param(
            ("optimum-intel-internal-testing/tiny-random-qwen2.5-vl", "SDPA"),
            id="qwen2.5-vl/SDPA",
        ),
    ],
    indirect=["ov_pipe_model"],
)
def test_vlm_pipeline_chat_history_multipart_content(
    ov_pipe_model: VlmModelInfo,
    iteration_images: list[list[PIL.Image]],
):
    ov_pipe = ov_pipe_model.pipeline

    generation_config = _setup_generation_config(ov_pipe, do_sample=False)

    prompts_with_images = [
        (PROMPTS[0], iteration_images[0]),
        *[(PROMPTS[1], image_set) for image_set in iteration_images[1:]],
    ]

    # Collect chat_history results (baseline)
    answers_chat_history = []
    history = ChatHistory()
    for prompt, images in prompts_with_images:
        history.append({"role": "user", "content": prompt})
        res = ov_pipe.generate(
            history,
            images=images,
            generation_config=generation_config,
        )
        answer = res.texts[0]
        history.append({"role": "assistant", "content": answer})
        answers_chat_history.append(answer)

    # Collect chat_history with multipart content (OpenAI-like) results
    answers_chat_history_multipart_content = []
    history_multipart_content = ChatHistory()
    for prompt, images in prompts_with_images:
        history_multipart_content.append(
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt},
                ],
            }
        )
        res = ov_pipe.generate(
            history_multipart_content,
            images=images,
            generation_config=generation_config,
        )
        answer = res.texts[0]
        history_multipart_content.append({"role": "assistant", "content": answer})
        answers_chat_history_multipart_content.append(answer)

    for i, (answer_1, answer_2) in enumerate(zip(answers_chat_history, answers_chat_history_multipart_content)):
        assert answer_1 == answer_2, (
            f"Answer {i} does not match!\n"
            f"answers_chat_history: {answer_1}\n"
            f"answers_chat_history_multipart_content: {answer_2}"
        )


@pytest.fixture(scope="module", params=[
    pytest.param([[], []], id="generation with text input only"),
    pytest.param(
        [[], ["cat_tensor"], ["car_tensor"], ["handwritten_tensor"], []],
        id="combination of generations with text input and text + image input, empty image first"
    ),
    pytest.param(
        [["cat_tensor"], ["car_tensor"], ["handwritten_tensor"]],
        id="generation with text + image input"
    ),
    pytest.param(
        [["cat_tensor"], ["car_tensor"], [], ["handwritten_tensor"]],
        id="combination of generations with text input and text + image input, image input first"
    ),
])
def iteration_images_npu(request):
    return [[request.getfixturevalue(image) for image in bundle] for bundle in request.param]


@parametrize_all_models_npu
@pytest.mark.parametrize("system_message", ["", "You are a helpful assistant."])
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_pipeline_chat_npu(ov_npu_pipe_model: VlmModelInfo, system_message, iteration_images_npu):
    def run_chat(ov_pipe, system_message, iteration_images):
        result_from_streamer = []
        def streamer(word: str) -> bool:
            result_from_streamer.append(word)
            return False

        generation_config = ov_pipe.get_generation_config()
        generation_config.max_new_tokens = 30
        generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

        ov_pipe.start_chat(system_message)

        for i, image_set in enumerate(iteration_images):
            result_from_streamer = []
            res = ov_pipe.generate(
                PROMPTS[i % len(PROMPTS)],
                images=image_set,
                generation_config=generation_config,
                streamer=streamer,
            )
            assert res.texts[0] == "".join(result_from_streamer)

        ov_pipe.finish_chat()

    npu_pipe = ov_npu_pipe_model.pipeline

    run_chat(npu_pipe, system_message, iteration_images_npu)


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


@parametrize_one_model_backends
def test_vlm_get_tokenizer(ov_pipe_model: VlmModelInfo):
    ov_pipe = ov_pipe_model.pipeline
    tokenizer = ov_pipe.get_tokenizer()
    tokenizer.encode("")


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


@pytest.mark.parametrize("backend", ATTENTION_BACKEND)
def test_perf_metrics(
    backend: str,
    cat_tensor: openvino.Tensor,
):
    import numpy as np
    from time import perf_counter_ns

    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    # Using non-cached model to get more accurate load time
    model_path = _get_ov_model("optimum-intel-internal-testing/tiny-random-minicpmv-2_6")
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


@parametrize_all_models_npu
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_npu_no_exception(ov_npu_pipe_model: VlmModelInfo, cat_tensor):
    ov_pipe = ov_npu_pipe_model.pipeline

    generation_config = _setup_generation_config(ov_pipe)

    ov_pipe.generate(PROMPTS[0], images=[cat_tensor], generation_config=generation_config)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(["cat_tensor"], id="cat_tensor - one image"),
        pytest.param([], id="empty"),
    ],
)
def image_sequence(request):
    return [request.getfixturevalue(image) for image in request.param]


@parametrize_one_model_npu
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_npu_no_image(ov_npu_pipe_model: VlmModelInfo):
    ov_pipe = ov_npu_pipe_model.pipeline

    generation_config = _setup_generation_config(ov_pipe)

    ov_pipe.generate(
        PROMPTS[0], generation_config=generation_config
    )


@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_npu_auto_config(cat_tensor):
    models_path = _get_ov_model(NPU_SUPPORTED_MODELS[0])
    properties = {
        "DEVICE_PROPERTIES": {
            "NPU": {"NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE", "MAX_PROMPT_LEN": 2048},
            "AUTO": {openvino.properties.device.priorities: "CPU,GPU"},
        }
    }

    ov_pipe = VLMPipeline(models_path, "NPU", config=properties)

    generation_config = _setup_generation_config(ov_pipe)

    ov_pipe.generate(PROMPTS[0], images=[cat_tensor], generation_config=generation_config)


@parametrize_one_model_npu
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_npu_multiple_images(
    ov_npu_pipe_model: VlmModelInfo, cat_tensor: openvino.Tensor, handwritten_tensor: openvino.Tensor
):
    ov_pipe = ov_npu_pipe_model.pipeline

    generation_config = _setup_generation_config(ov_pipe)

    ov_pipe.generate(PROMPTS[0], images=[cat_tensor, handwritten_tensor], generation_config=generation_config)


@parametrize_all_models
def test_vlm_pipeline_chat_streamer_cancel_second_generate(
    request: pytest.FixtureRequest, ov_pipe_model: VlmModelInfo, image_sequence: list[openvino.Tensor]
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
    # doesn't add to results_with_cancel as it should be completely removed from the history
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


def test_start_chat_clears_history_cb_api(
    ov_continious_batching_pipe: ContinuousBatchingPipeline, image_sequence: list[openvino.Tensor]
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


def generate(
    vlm: VLMPipeline, requests: list[tuple[str, list[openvino.Tensor]]], vision_type: VisionType = VisionType.IMAGE
):
    generation_config = _setup_generation_config(vlm, set_eos_token=False)
    vlm.set_generation_config(generation_config)
    vlm.start_chat()
    answers = [
        vlm.generate(prompt, **get_vision_inputs_kwargs(visions, vision_type), do_sample=False)
        for (prompt, visions) in requests
    ]
    vlm.finish_chat()
    return answers


@pytest.fixture(scope="module")
def conversation_requests(
    cat_tensor: openvino.Tensor, car_tensor: openvino.Tensor, handwritten_tensor: openvino.Tensor
) -> list[tuple[str, list[openvino.Tensor]]]:
    return [
        ("Describe", [cat_tensor]),
        ("How many images are there?", [car_tensor, handwritten_tensor]),
    ]


@pytest.fixture(scope="module")
def conversation_video_requests(
    synthetic_video_32x32_tensor: openvino.Tensor,
) -> list[tuple[str, list[openvino.Tensor]]]:
    return [
        ("Describe", [synthetic_video_32x32_tensor]),
        ("How many images are there?", [synthetic_video_32x32_tensor, synthetic_video_32x32_tensor]),
    ]


TAG_INSERTED_BY_TEMPLATE = [
    ("optimum-intel-internal-testing/tiny-random-llava", "PA"),
    ("optimum-intel-internal-testing/tiny-random-llava-next", "PA"),
    ("optimum-intel-internal-testing/tiny-random-qwen2vl", "PA"),
    ("optimum-intel-internal-testing/tiny-random-qwen2.5-vl", "PA"),
    ("optimum-intel-internal-testing/tiny-random-gemma3", "SDPA"),
    ("qnguyen3/nanoLLaVA", "PA"),
    ("optimum-intel-internal-testing/tiny-random-llava-next-video", "PA"),
]


IMAGE_ID_IGNORANT_MODELS_TO_TAG = TAG_INSERTED_BY_TEMPLATE + [
    ("optimum-intel-internal-testing/tiny-random-internvl2", "PA"),
]


MODELS_TO_TAG = IMAGE_ID_IGNORANT_MODELS_TO_TAG + [
    ("optimum-intel-internal-testing/tiny-random-minicpmv-2_6", "PA"),
    ("optimum-intel-internal-testing/tiny-random-phi3-vision", "PA"),
]


def get_vision_inputs_kwargs(visions: list[openvino.Tensor], vision_type: VisionType) -> dict:
    if vision_type == VisionType.IMAGE:
        return {"images": visions}
    else:
        return {"videos": visions}


def get_universal_tag(vision_type: VisionType, index: int) -> str:
    if vision_type == VisionType.IMAGE:
        return f"<ov_genai_image_{index}>"
    else:
        return f"<ov_genai_video_{index}>"


def parametrize_model_with_vision_type(
    items: list[tuple[str, str]] | None = None,
    xfail: dict[tuple[str, str, VisionType], str] | None = None,  # (model, attn_backend, VisionType) -> reason,
) -> Callable[[Callable], Generator]:
    if items is None:
        items = MODELS_TO_TAG

    xfail = xfail or {}

    # params: items (model and backend) + vision_type
    params: list[tuple[tuple[str, str], VisionType]] = []
    ids = []
    for item in items:

        def append_param(item, vision_type):
            model_id, attn_backend = item[0], item[1]
            ids.append(f"{model_id}/{attn_backend}/{vision_type.value}")
            reason = xfail.get((model_id, attn_backend, vision_type))
            if reason:
                params.append(pytest.param(item, vision_type, marks=pytest.mark.xfail(reason=reason)))
            else:
                params.append((item, vision_type))

        append_param(item, VisionType.IMAGE)
        if item[0] in VIDEO_MODEL_IDS:
            append_param(item, VisionType.VIDEO)

    return pytest.mark.parametrize(
        "ov_pipe_model,vision_type",
        params,
        indirect=["ov_pipe_model"],
        ids=ids,
    )


@parametrize_model_with_vision_type(
    TAG_INSERTED_BY_TEMPLATE,
    xfail={("optimum-intel-internal-testing/tiny-random-llava", "PA", VisionType.IMAGE): "CVS-179090"},
)
def test_model_tags_representation(
    ov_pipe_model: VlmModelInfo,
    vision_type: VisionType,
    request: pytest.FixtureRequest,
):
    ov_pipe = ov_pipe_model.pipeline
    model_id = ov_pipe_model.model_id

    generation_config = _setup_generation_config(ov_pipe, set_eos_token=False)
    ov_pipe.set_generation_config(generation_config)
    prompt = "Describe"

    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
    if model_id == "qnguyen3/nanoLLaVA":
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True)
        messages = [{"role": "user", "content": f"{ov_pipe_model.get_vision_tag(vision_type)(0)}{prompt}"}]
        templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        processor = retry_request(
            lambda: transformers.AutoProcessor.from_pretrained(
                model_cached,
                trust_remote_code=True,
                **align_with_optimum_cli,
            )
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image" if vision_type == VisionType.IMAGE else "video"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        templated_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    input_tensor: openvino.Tensor = request.getfixturevalue(
        "cat_tensor" if vision_type == VisionType.IMAGE else "synthetic_video_32x32_tensor"
    )
    vision_inputs_kwargs = get_vision_inputs_kwargs([input_tensor], vision_type)

    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        automatic_tags = ov_pipe.generate(prompt, **vision_inputs_kwargs, do_sample=False)
        reference_tags = ov_pipe.generate(
            templated_prompt, **vision_inputs_kwargs, apply_chat_template=False, do_sample=False
        )
        assert automatic_tags.texts == reference_tags.texts
        assert automatic_tags.scores == reference_tags.scores

    retry(workaround_inconsistent_inference)


@parametrize_model_with_vision_type()
def test_model_tags_prepend_native(
    ov_pipe_model: VlmModelInfo,
    vision_type: VisionType,
    request: pytest.FixtureRequest,
):
    ov_pipe = ov_pipe_model.pipeline
    vision_tag = ov_pipe_model.get_vision_tag(vision_type)

    conversation_requests = request.getfixturevalue(
        "conversation_requests" if vision_type == VisionType.IMAGE else "conversation_video_requests"
    )

    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        answers = generate(ov_pipe, conversation_requests, vision_type)

        ov_pipe.start_chat()
        native_tag0 = ov_pipe.generate(
            vision_tag(0) + conversation_requests[0][0],
            **get_vision_inputs_kwargs(conversation_requests[0][1], vision_type),
            do_sample=False,
        )
        assert native_tag0.texts == answers[0].texts
        assert native_tag0.scores == answers[0].scores
        native_tags1 = ov_pipe.generate(
            vision_tag(1) + vision_tag(2) + conversation_requests[1][0],
            **get_vision_inputs_kwargs(conversation_requests[1][1], vision_type),
            do_sample=False,
        )
        assert native_tags1.texts == answers[1].texts
        assert native_tags1.scores == answers[1].scores
        ov_pipe.finish_chat()

    retry(workaround_inconsistent_inference)


@parametrize_model_with_vision_type()
def test_model_tags_prepend_universal(
    ov_pipe_model: VlmModelInfo,
    vision_type: VisionType,
    request: pytest.FixtureRequest,
):
    ov_pipe = ov_pipe_model.pipeline

    conversation_requests = request.getfixturevalue(
        "conversation_requests" if vision_type == VisionType.IMAGE else "conversation_video_requests"
    )

    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        answers = generate(ov_pipe, conversation_requests, vision_type)

        ov_pipe.start_chat()
        universal_tag0 = ov_pipe.generate(
            get_universal_tag(vision_type, 0) + conversation_requests[0][0],
            **get_vision_inputs_kwargs(conversation_requests[0][1], vision_type),
            do_sample=False,
        )
        assert universal_tag0.texts == answers[0].texts
        assert universal_tag0.scores == answers[0].scores
        universal_tags1 = ov_pipe.generate(
            get_universal_tag(vision_type, 1) + get_universal_tag(vision_type, 2) + conversation_requests[1][0],
            **get_vision_inputs_kwargs(conversation_requests[1][1], vision_type),
            do_sample=False
        )
        assert universal_tags1.texts == answers[1].texts
        assert universal_tags1.scores == answers[1].scores
        ov_pipe.finish_chat()

    retry(workaround_inconsistent_inference)


@parametrize_model_with_vision_type()
def test_model_tags_append(
    ov_pipe_model: VlmModelInfo,
    vision_type: VisionType,
    request: pytest.FixtureRequest,
):
    ov_pipe = ov_pipe_model.pipeline
    vision_tag = ov_pipe_model.get_vision_tag(vision_type)

    conversation_requests = request.getfixturevalue(
        "conversation_requests" if vision_type == VisionType.IMAGE else "conversation_video_requests"
    )

    generation_config = _setup_generation_config(ov_pipe, set_eos_token=False)
    ov_pipe.set_generation_config(generation_config)

    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        ov_pipe.start_chat()
        native_tag0 = ov_pipe.generate(
            conversation_requests[0][0] + vision_tag(0),
            **get_vision_inputs_kwargs(conversation_requests[0][1], vision_type),
            do_sample=False,
        )
        native_tags1 = ov_pipe.generate(
            conversation_requests[1][0] + vision_tag(1) + vision_tag(2),
            **get_vision_inputs_kwargs(conversation_requests[1][1], vision_type),
            do_sample=False,
        )
        ov_pipe.finish_chat()

        ov_pipe.start_chat()
        universal_tag0 = ov_pipe.generate(
            conversation_requests[0][0] + get_universal_tag(vision_type, 0),
            **get_vision_inputs_kwargs(conversation_requests[0][1], vision_type),
            do_sample=False,
        )
        assert universal_tag0.texts == native_tag0.texts
        assert universal_tag0.scores == native_tag0.scores
        universal_tags1 = ov_pipe.generate(
            conversation_requests[1][0] + get_universal_tag(vision_type, 1) + get_universal_tag(vision_type, 2),
            **get_vision_inputs_kwargs(conversation_requests[1][1], vision_type),
            do_sample=False
        )
        assert universal_tags1.texts == native_tags1.texts
        assert universal_tags1.scores == native_tags1.scores
        ov_pipe.finish_chat()

    retry(workaround_inconsistent_inference)


@parametrize_model_with_vision_type(IMAGE_ID_IGNORANT_MODELS_TO_TAG)
def test_model_tags_same_reference(
    ov_pipe_model: VlmModelInfo,
    vision_type: VisionType,
    request: pytest.FixtureRequest,
):
    ov_pipe = ov_pipe_model.pipeline

    generation_config = _setup_generation_config(ov_pipe, max_new_tokens=2, set_eos_token=False)
    ov_pipe.set_generation_config(generation_config)

    input_tensor: openvino.Tensor = request.getfixturevalue(
        "cat_tensor" if vision_type == VisionType.IMAGE else "synthetic_video_32x32_tensor"
    )

    def workaround_inconsistent_inference():
        __tracebackhide__ = True
        one_input = ov_pipe.generate(
            get_universal_tag(vision_type, 0) * 2,
            **get_vision_inputs_kwargs([input_tensor], vision_type),
            do_sample=False,
        )
        two_inputs = ov_pipe.generate(
            get_universal_tag(vision_type, 0) + get_universal_tag(vision_type, 1),
            **get_vision_inputs_kwargs([input_tensor] * 2, vision_type),
            do_sample=False,
        )
        assert one_input.texts == two_inputs.texts
        assert one_input.scores == two_inputs.scores

    retry(workaround_inconsistent_inference)


@parametrize_model_with_vision_type()
def test_model_tags_older(
    ov_pipe_model: VlmModelInfo,
    vision_type: VisionType,
    request: pytest.FixtureRequest,
):
    ov_pipe = ov_pipe_model.pipeline

    input_tensor: openvino.Tensor = request.getfixturevalue(
        "car_tensor" if vision_type == VisionType.IMAGE else "synthetic_video_32x32_tensor"
    )

    generation_config = _setup_generation_config(ov_pipe, set_eos_token=False)
    ov_pipe.set_generation_config(generation_config)
    ov_pipe.start_chat()
    ov_pipe.generate("", **get_vision_inputs_kwargs([input_tensor], vision_type))
    with pytest.raises(RuntimeError):
        ov_pipe.generate(get_universal_tag(vision_type, 0), **get_vision_inputs_kwargs([input_tensor], vision_type))
    ov_pipe.finish_chat()


@parametrize_model_with_vision_type()
def test_model_tags_missing_universal(ov_pipe_model: VlmModelInfo, vision_type: VisionType):
    ov_pipe = ov_pipe_model.pipeline

    with pytest.raises(RuntimeError):
        ov_pipe.generate(get_universal_tag(vision_type, 0))


@parametrize_model_with_vision_type()
def test_model_tags_missing_native(ov_pipe_model: VlmModelInfo, vision_type: VisionType):
    ov_pipe = ov_pipe_model.pipeline
    vision_tag = ov_pipe_model.get_vision_tag(vision_type)

    with pytest.raises(RuntimeError):
        ov_pipe.generate(vision_tag(0))


def run_compare_genai_optimum(ov_pipe_model: VlmModelInfo, image, video):
    class NanollavaProcessorWrapper:
        def __init__(self, processor, config, model_dtype):
            self.processor = processor
            self.config = config
            self.model_dtype = model_dtype

        def __call__(self, images, return_tensors):
            return {"pixel_values": self.processor(images, self.config).to(dtype=self.model_dtype)}

    def get_nanollava_processor():
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_cached, device_map="auto", trust_remote_code=True
        )
        return NanollavaProcessorWrapper(hf_model.process_images, hf_model.config, hf_model.dtype)

    ov_pipe = ov_pipe_model.pipeline

    model_id = ov_pipe_model.model_id
    model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
    model_path = _get_ov_model(model_id)
    optimum_model = OVModelForVisualCausalLM.from_pretrained(model_path, trust_remote_code=True)

    prompt_parts = []
    if image is not None:
        prompt_parts.append("image")

    if video is not None:
        prompt_parts.append("video")

    if len(prompt_parts) == 1:
        prompt = f"Describe this {prompt_parts[0]}."
    elif len(prompt_parts) == 2:
        prompt = f"Describe this {prompt_parts[0]} and {prompt_parts[1]}."
    else:
        prompt = "Describe."

    # Run the optimum_model with optimum-intel
    tokenizer = None
    if optimum_model.config.model_type == "llava-qwen2":
        processor = get_nanollava_processor()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True)

        from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING

        preprocess_inputs = MODEL_TYPE_TO_CLS_MAPPING[optimum_model.config.model_type].preprocess_inputs
        inputs = preprocess_inputs(prompt, image, processor, tokenizer, config=optimum_model.config)
    else:
        processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Gemma3 input_ids has two bos tokens when running with optimum: one in chat template + "add_bos_token" is set to True in tokenizer_config.json
        if optimum_model.config.model_type == "gemma3":
            processor.tokenizer.add_bos_token = False
        if optimum_model.config.model_type in ["internvl_chat", "minicpmv"]:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True)
        if optimum_model.config.model_type == "minicpmv":
            # optimum 1.27.0 will manually apply chat template if processor.chat_template isn't set.
            # So, make sure we set it here to align with GenAI routines.
            if (
                getattr(processor, "chat_template", None) is None
                and getattr(tokenizer, "chat_template", None) is not None
            ):
                processor.chat_template = tokenizer.chat_template

        inputs = optimum_model.preprocess_inputs(
            text=prompt, image=image, video=video, processor=processor, tokenizer=tokenizer, config=optimum_model.config
        )

    max_new_tokens = 100
    output_ids = optimum_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, output_ids)]

    if optimum_model.config.model_type == "llava-qwen2":
        assert tokenizer is not None, "Tokenizer should be set for llava-qwen2 models."
        optimum_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    else:
        optimum_output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        optimum_text = optimum_output[0]

    params = {}
    if image is not None:
        params["images"] = [openvino.Tensor(image)]
    if video is not None:
        params["videos"] = [openvino.Tensor(video)]

    genai_output = ov_pipe.generate(prompt, **params, max_new_tokens=max_new_tokens, do_sample=False)
    genai_text = genai_output.texts[0]

    assert optimum_text == genai_text


# (Width, Height)
OPTIMUM_VS_GENAI_DEFAULT_IMAGE_RESOLUTIONS = [(100, 77), (999, 666), (1920, 1080)]

# (Width, Height)
OPTIMUM_VS_GENAI_DEFAULT_VIDEO_RESOLUTIONS = [(32, 32), (176, 132), (640, 480)]

# For qwen2-series models, we use smaller image / video resolutions.
# This is because running with larger image and/or video resolutions allocates,
# a ton of memory. And in the case of optimum, there seems to be a big chunk that
# is not freed after test completion. See ticket: CVS-180177
OPTIMUM_VS_GENAI_PER_MODEL_IMAGE_RESOLUTIONS = {
    "optimum-intel-internal-testing/tiny-random-qwen2vl": [(100, 77), (350, 350), (480, 512)],
    "optimum-intel-internal-testing/tiny-random-qwen2.5-vl": [(100, 77), (350, 350), (480, 512)],
}

OPTIMUM_VS_GENAI_PER_MODEL_VIDEO_RESOLUTIONS = {
    "optimum-intel-internal-testing/tiny-random-qwen2vl": [(32, 32), (70, 70)],
    "optimum-intel-internal-testing/tiny-random-qwen2.5-vl": [(32, 32), (70, 70)],
}

# test-id glob pattern -> xfail reason
# test-id's are of the form:
# "<model_id>/<attn_backend>/<preprocessing>/image-<W>x<H>/video-<W>x<H>"
OPTIMUM_VS_GENAI_MODEL_EXPECTED_FAIL_CASES = {
    # gemma3 PA cases
    "*tiny-random-gemma3/PA/*": "CVS-167316",
    # qwen2vl cases that use 70x70 video resolution
    "*tiny-random-qwen2vl/*/video-70x70": "CVS-180070",
    # qwen2.5-vl cases that use 350x350 image, or 70x70 video resolutions
    "*tiny-random-qwen2.5-vl/*/image-350x350*": "CVS-180070",
    "*tiny-random-qwen2.5-vl/*/video-70x70": "CVS-180070",
    # llava-next-video graph pre-processing 'real' resize cases that include video
    "*tiny-random-llava-next-video/*/GRAPH/video*": "CVS-180070",
    "*tiny-random-llava-next-video/*/GRAPH/image*/video*": "CVS-180070",
    # llava-next text-only cases
    "*tiny-random-llava-next/*/CPP/text-only": "CVS-180070",
    # llava cases
    "*tiny-random-llava/*": "CVS-180070",
    # MiniCPM-o-2_6 text-only cases
    "*tiny-random-MiniCPM-o-2_6/*/text-only": "CVS-180070",
    # minicpmv-2_6 cases with images
    "*tiny-random-minicpmv-2_6/*/image*": "CVS-180070",
}

# For these models, we will add both CPP and GRAPH pre-processing tests.
MODELS_THAT_SUPPORT_GRAPH_PREPROCESSING = [
    "optimum-intel-internal-testing/tiny-random-llava-next-video",
    "optimum-intel-internal-testing/tiny-random-phi3-vision",
    "optimum-intel-internal-testing/tiny-random-phi-4-multimodal",
    "optimum-intel-internal-testing/tiny-random-qwen2vl",
    "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
]

# For these models, we will only add GRAPH pre-processing tests.
MODELS_THAT_DO_NOT_SUPPORT_CPP_PREPROCESSING = ["optimum-intel-internal-testing/tiny-random-phi-4-multimodal"]


# Each test will have an id in one of the following formats:
# text-only: <model_id>/<attn_backend>/<preprocessing>/text-only
# text+image: <model_id>/<attn_backend>/<preprocessing>/image-<W>x<H>
# text+image+video: <model_id>/<attn_backend>/<preprocessing>/image-<W>x<H>/video-<W>x<H>
#
# If a model-id is defined in RESOLUTION_BY_MODEL, then there will be pre-resize cases added:
# text+image: <model_id>/<attn_backend>/<preprocessing>/preresized-image
# text+image+video: <model_id>/<attn_backend>/<preprocessing>/preresized-image+video
# id's that match glob patterns in OPTIMUM_VS_GENAI_MODEL_EXPECTED_FAIL_CASES are marked as xfail.
def parametrize_optimum_vs_genai(models: list[str] | None = None) -> Callable[[Callable], Generator]:
    from itertools import product
    from fnmatch import fnmatch

    if models is None:
        models = MODEL_IDS

    params = []

    def append_test_case(
        *,
        model_id: str,
        attn_backend: str,
        preprocessing: str,
        has_image: bool,
        has_video: bool,
        image_resolution: tuple[int, int],
        video_resolution: tuple[int, int],
        test_id: str,
    ):
        xfail_reason = None
        for pattern, reason in OPTIMUM_VS_GENAI_MODEL_EXPECTED_FAIL_CASES.items():
            if fnmatch(test_id, pattern):
                xfail_reason = reason
                break

        marks = pytest.mark.xfail(reason=xfail_reason) if xfail_reason else ()
        params.append(
            pytest.param(
                (model_id, attn_backend, preprocessing),
                has_image,
                has_video,
                image_resolution,
                video_resolution,
                marks=marks,
                id=test_id,
            )
        )

    for model_id in models:
        supported_attn_backends = ATTENTION_BACKEND
        supported_preprocessing = ["CPP", "GRAPH"]

        if model_id not in MODELS_THAT_SUPPORT_GRAPH_PREPROCESSING:
            supported_preprocessing = ["CPP"]

        if model_id in MODELS_THAT_DO_NOT_SUPPORT_CPP_PREPROCESSING:
            supported_preprocessing = ["GRAPH"]

        supported_has_image_inputs = [False, True]
        supported_has_video_inputs = [False]

        if model_id in VIDEO_MODEL_IDS:
            supported_has_video_inputs = [False, True]

        for attn_backend, preprocessing, has_image, has_video in product(
            ATTENTION_BACKEND,
            supported_preprocessing,
            supported_has_image_inputs,
            supported_has_video_inputs,
        ):
            # add pre-resized cases, if model is defined in RESOLUTION_BY_MODEL
            if has_image and model_id in RESOLUTION_BY_MODEL:
                test_id = f"{model_id}/{attn_backend}/{preprocessing}/preresized-image"

                res = RESOLUTION_BY_MODEL[model_id]
                image_resolution = (res, res)

                video_resolution = None
                if has_video:
                    test_id += "+video"
                    # for pre-resize cases, we always use 32x32 video resolution.
                    video_resolution = (32, 32)

                append_test_case(
                    model_id=model_id,
                    attn_backend=attn_backend,
                    preprocessing=preprocessing,
                    has_image=has_image,
                    has_video=has_video,
                    image_resolution=image_resolution,
                    video_resolution=video_resolution,
                    test_id=test_id,
                )

            # 'Real' resolution cases
            image_resolutions = [None]
            if has_image:
                image_resolutions = OPTIMUM_VS_GENAI_PER_MODEL_IMAGE_RESOLUTIONS.get(
                    model_id, OPTIMUM_VS_GENAI_DEFAULT_IMAGE_RESOLUTIONS
                )

            video_resolutions = [None]
            if has_video:
                video_resolutions = OPTIMUM_VS_GENAI_PER_MODEL_VIDEO_RESOLUTIONS.get(
                    model_id, OPTIMUM_VS_GENAI_DEFAULT_VIDEO_RESOLUTIONS
                )

            for image_resolution, video_resolution in product(image_resolutions, video_resolutions):
                test_id = f"{model_id}/{attn_backend}/{preprocessing}"
                if image_resolution:
                    test_id += f"/image-{image_resolution[0]}x{image_resolution[1]}"
                if video_resolution:
                    test_id += f"/video-{video_resolution[0]}x{video_resolution[1]}"
                if not image_resolution and not video_resolution:
                    test_id += f"/text-only"

                append_test_case(
                    model_id=model_id,
                    attn_backend=attn_backend,
                    preprocessing=preprocessing,
                    has_image=has_image,
                    has_video=has_video,
                    image_resolution=image_resolution,
                    video_resolution=video_resolution,
                    test_id=test_id,
                )

    return pytest.mark.parametrize(
        "ov_pipe_model,has_image,has_video,image_input_resolution,video_input_resolution",
        params,
        indirect=["ov_pipe_model"],
    )


@parametrize_optimum_vs_genai()
def test_vlm_pipeline_match_optimum_with_resolutions(
    request: pytest.FixtureRequest,
    ov_pipe_model: VlmModelInfo,
    has_image: bool,
    has_video: bool,
    image_input_resolution: tuple[int, int],
    video_input_resolution: tuple[int, int],
):
    resized_image = None
    resized_video = None
    if has_image:
        resized_image = request.getfixturevalue("cat_image")
        resized_image = resized_image.resize(image_input_resolution)

    if has_video:
        resized_video = request.getfixturevalue("synthetic_video")
        resized_video = resize_video(resized_video, video_input_resolution)

    run_compare_genai_optimum(ov_pipe_model, resized_image, resized_video)


# CDPruner Tests

CDPRUNER_SUPPORTED_MODELS = [
    "optimum-intel-internal-testing/tiny-random-qwen2vl",
    "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
]

parametrize_cdpruner_models = pytest.mark.parametrize(
    "ov_pipe_model",
    [(m, b) for m in CDPRUNER_SUPPORTED_MODELS for b in ATTENTION_BACKEND],
    ids=lambda p: f"{p[0]}/{p[1]}",
    indirect=["ov_pipe_model"],
)


@parametrize_cdpruner_models
@pytest.mark.parametrize("pruning_ratio", [0, 30, 50, 80])
def test_cdpruner_functionality(ov_pipe_model: VlmModelInfo, cat_tensor: openvino.Tensor, pruning_ratio: int):
    """Test CDPruner functionality with different pruning ratios."""
    ov_pipe = ov_pipe_model.pipeline
    generation_config = _setup_generation_config(ov_pipe, max_new_tokens=20, do_sample=False)
    generation_config.pruning_ratio = pruning_ratio

    result = ov_pipe.generate(PROMPTS[0], images=[cat_tensor], generation_config=generation_config)

    # Verify result is non-empty
    assert result.texts[0].strip() != "", f"Result with {pruning_ratio}% pruning should not be empty"

    # Verify perf metrics are available
    assert result.perf_metrics is not None, "Performance metrics should be available"


@parametrize_cdpruner_models
def test_cdpruner_with_multiple_images(
    ov_pipe_model: VlmModelInfo,
    cat_tensor: openvino.Tensor,
    car_tensor: openvino.Tensor,
    handwritten_tensor: openvino.Tensor,
):
    """Test CDPruner with multiple images."""
    ov_pipe = ov_pipe_model.pipeline
    generation_config = _setup_generation_config(ov_pipe, max_new_tokens=25, do_sample=False)

    images = [cat_tensor, car_tensor, handwritten_tensor]

    # Test with 30% pruning
    generation_config.pruning_ratio = 30
    result = ov_pipe.generate("Describe these images.", images=images, generation_config=generation_config)

    assert result.texts[0].strip() != "", "Result with multiple images should not be empty"
    assert result.perf_metrics is not None


@parametrize_cdpruner_models
@pytest.mark.xfail(condition=(sys.platform == "win32"), run=False, reason="Segfault. Ticket - 179274")
def test_cdpruner_chat_mode(ov_pipe_model: VlmModelInfo, cat_tensor: openvino.Tensor, car_tensor: openvino.Tensor):
    """Test CDPruner in chat mode."""
    ov_pipe = ov_pipe_model.pipeline
    generation_config = _setup_generation_config(ov_pipe, max_new_tokens=20, do_sample=False)

    # Enable pruning
    generation_config.pruning_ratio = 25

    # Start chat
    ov_pipe.start_chat("You are a helpful assistant.")

    # First turn with image
    result1 = ov_pipe.generate("What is in this image?", images=[cat_tensor], generation_config=generation_config)
    assert result1.texts[0].strip() != "", "First turn result should not be empty"

    # Second turn with different image
    result2 = ov_pipe.generate("Now describe this one.", images=[car_tensor], generation_config=generation_config)
    assert result2.texts[0].strip() != "", "Second turn result should not be empty"

    # Third turn without image
    result3 = ov_pipe.generate("What did you see in total?", generation_config=generation_config)
    assert result3.texts[0].strip() != "", "Third turn result should not be empty"

    ov_pipe.finish_chat()


@parametrize_cdpruner_models
@pytest.mark.parametrize("relevance_weight", [0.0, 0.2, 0.8, 1.0])
def test_cdpruner_with_relevance_weight(
    ov_pipe_model: VlmModelInfo, cat_tensor: openvino.Tensor, relevance_weight: float
):
    """Test CDPruner with different relevance weights."""
    ov_pipe = ov_pipe_model.pipeline
    generation_config = _setup_generation_config(ov_pipe, max_new_tokens=20, do_sample=False)
    generation_config.pruning_ratio = 30
    generation_config.relevance_weight = relevance_weight
    result = ov_pipe.generate(PROMPTS[0], images=[cat_tensor], generation_config=generation_config)

    assert result.texts[0].strip() != "", f"Result with relevance_weight={relevance_weight} should not be empty"


@parametrize_cdpruner_models
def test_cdpruner_disable_after_enable(ov_pipe_model: VlmModelInfo, cat_tensor: openvino.Tensor):
    """Test disabling CDPruner after enabling it."""
    ov_pipe = ov_pipe_model.pipeline

    # Enable pruning
    config_with_pruning = _setup_generation_config(ov_pipe, max_new_tokens=20, do_sample=False)
    config_with_pruning.pruning_ratio = 40
    result_with_pruning = ov_pipe.generate(PROMPTS[0], images=[cat_tensor], generation_config=config_with_pruning)

    # Disable pruning
    config_no_pruning = _setup_generation_config(ov_pipe, max_new_tokens=20, do_sample=False)
    config_no_pruning.pruning_ratio = 0
    result_without_pruning = ov_pipe.generate(PROMPTS[0], images=[cat_tensor], generation_config=config_no_pruning)

    assert result_with_pruning.texts[0].strip() != "", "Result with pruning should not be empty"
    assert result_without_pruning.texts[0].strip() != "", "Result without pruning should not be empty"


@pytest.fixture(scope="module")
def ov_continuous_batching_pipe_qwen2vl() -> ContinuousBatchingPipeline:
    """Fixture for Qwen2VL continuous batching pipeline."""
    model_path = _get_ov_model(CDPRUNER_SUPPORTED_MODELS[0])
    return ContinuousBatchingPipeline(model_path, SchedulerConfig(), "CPU")


def test_cdpruner_continuous_batching(
    ov_continuous_batching_pipe_qwen2vl: ContinuousBatchingPipeline,
    cat_tensor: openvino.Tensor,
    car_tensor: openvino.Tensor,
):
    """Test CDPruner with continuous batching pipeline."""
    # Enable pruning via GenerationConfig
    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 20
    generation_config.do_sample = False
    generation_config.pruning_ratio = 25

    # Test batch with different images
    results = ov_continuous_batching_pipe_qwen2vl.generate(
        [PROMPTS[0]], images=[[car_tensor]], generation_config=[generation_config]
    )

    assert results[0].texts[0].strip() != "", "Result should not be empty"
