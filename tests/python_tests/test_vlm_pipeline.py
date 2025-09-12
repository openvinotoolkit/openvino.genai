# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Fixture hierarchy
cat_image ─────────┬─── cat_image_336x336
    |              │
cat_tensor         └─── cat_image_32x32
    │
    ├──── iteration_images
    │          │
    │          ├─── cat_tensor
    │          ├─── car_tensor
    │          └─── handwritten_tensor
    │
    ├──── image_sequence
    │
    └──── conversation_requests
                │
                ├─── cat_tensor
                ├─── car_tensor
                └─── handwritten_tensor
car_tensor
handwritten_tensor
model_and_tag
"""

import openvino_tokenizers
import openvino
import PIL
import pytest
import platform
import requests
import sys
import os
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
from utils.constants import get_default_llm_properties, get_ov_cache_models_dir


def get_ov_model(model_id):
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
    # For tiny-random-internvl2 processor is actually tokenizer
    if isinstance(processor, transformers.Qwen2TokenizerFast):
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
    model = retry_request(
        lambda: OVModelForVisualCausalLM.from_pretrained(
            model_id,
            compile=False,
            device="CPU",
            export=True,
            load_in_8bit=False,
            trust_remote_code=True,
            ov_config=get_default_llm_properties(),
        )
    )
    if tokenizer.chat_template is not None and model.config.model_type == "phi3_v":
        processor.chat_template = tokenizer.chat_template  # It seems that tiny-random-phi3-vision is saved incorrectly. That line works this around.
    processor.audio_tokenizer = None
    processor.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    return model_dir


prompts = [
    "What is on the image?",
    "What is special about this image?",
]


def download_image(link):
    return PIL.Image.open(requests.get(link, stream=True).raw).convert("RGB")


def from_cache_or_download(pytestconfig, link, file_name):
    def implementation():
        try:
            image_path = pytestconfig.cache.mkdir("images") / file_name
        except AttributeError:
            # Cache is disabled with -p no:cacheprovider
            return download_image(link)
        if image_path.exists():
            image = PIL.Image.open(image_path)
        else:
            image = download_image(link)
            image.save(image_path)
        return image
    return retry(implementation, PIL.UnidentifiedImageError)


@pytest.fixture(scope="module")
def cat_image(pytestconfig):
    cat_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    return from_cache_or_download(pytestconfig, cat_url, "cat.jpg")


@pytest.fixture(scope="module")
def cat_tensor(cat_image):
    return openvino.Tensor(cat_image)


@pytest.fixture(scope="module")
def car_tensor(pytestconfig):
    car_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    return openvino.Tensor(from_cache_or_download(pytestconfig, car_url, "car.jpg"))


@pytest.fixture(scope="module")
def handwritten_tensor(pytestconfig):
    handwritten_url = "https://github.com/user-attachments/assets/8c9ae017-7837-4abc-ae92-c1054c9ec350"
    return openvino.Tensor(from_cache_or_download(pytestconfig, handwritten_url, "handwritten.png"))


model_ids = [
    "katuni4ka/tiny-random-minicpmv-2_6",
    "katuni4ka/tiny-random-phi3-vision",
    "katuni4ka/tiny-random-phi-4-multimodal",
    "katuni4ka/tiny-random-llava",
    "katuni4ka/tiny-random-llava-next",
    "katuni4ka/tiny-random-internvl2",
    "katuni4ka/tiny-random-qwen2vl",
    "katuni4ka/tiny-random-qwen2.5-vl",
    "katuni4ka/tiny-random-gemma3",
]

# On macOS, transformers<4.52 is required, but this causes gemma3 to fail
GEMMA3_MACOS_XFAIL_REASON = "gemma3 not supported on macOS with older transformers"

if sys.platform == "darwin":
    model_ids = [
        pytest.param(
            model_id,
            marks=pytest.mark.xfail(reason=GEMMA3_MACOS_XFAIL_REASON)
        ) if "gemma3" in model_id else model_id
        for model_id in model_ids
    ]

attention_backend = ["PA", "SDPA"]


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("backend", attention_backend)
def test_vlm_pipeline(model_id, backend, cat_tensor, handwritten_tensor, car_tensor):
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

    models_path = get_ov_model(model_id)
    ov_pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    for images in [], [cat_tensor], [cat_tensor, handwritten_tensor, car_tensor]:
        result_from_streamer = []
        res = ov_pipe.generate(
            prompts[0],
            images=images,
            generation_config=generation_config,
            streamer=streamer,
        )
        assert res.texts[0] == "".join(result_from_streamer)


configs = [
    get_greedy(),
    get_beam_search(),
]


@pytest.mark.precommit
@pytest.mark.parametrize("config", configs)
def test_vlm_continuous_batching_generate_vs_add_request(config, cat_tensor):
    scheduler_config = SchedulerConfig()
    models_path = get_ov_model(model_ids[0])
    ov_pipe = VLMPipeline(
        models_path,
        "CPU",
        scheduler_config=scheduler_config,
        **get_default_llm_properties(),
    )
    generation_config = config
    generation_config.max_new_tokens = 30
    eps = 0.001
    image_links_list = [[], [cat_tensor]]

    res_generate = []
    for images in image_links_list:
        res_generate.append(
            ov_pipe.generate(
                prompts[0], images=images, generation_config=generation_config
            )
        )

    cb_pipe = ContinuousBatchingPipeline(
        models_path,
        scheduler_config=scheduler_config,
        device="CPU",
        properties=get_default_llm_properties(),
    )
    tokenizer = cb_pipe.get_tokenizer()

    for idx, images in enumerate(image_links_list):
        handle = cb_pipe.add_request(idx, prompts[0], images, generation_config)
        while handle.get_status() != GenerationStatus.FINISHED:
            cb_pipe.step()
        outputs = handle.read_all()
        for out_idx, output in enumerate(outputs):
            text = tokenizer.decode(output.generated_ids)
            assert text == res_generate[idx].texts[out_idx]
            assert abs(output.score - res_generate[idx].scores[out_idx]) < eps
            assert (
                output.finish_reason == GenerationFinishReason.STOP
                or output.finish_reason == GenerationFinishReason.LENGTH
            )


@pytest.mark.precommit
@pytest.mark.parametrize("config", configs)
def test_vlm_continuous_batching_vs_stateful(config, cat_tensor):
    scheduler_config = SchedulerConfig()
    models_path = get_ov_model(model_ids[0])
    cb_pipe = ContinuousBatchingPipeline(
        models_path,
        scheduler_config=scheduler_config,
        device="CPU",
        properties=get_default_llm_properties(),
    )
    generation_config = config
    generation_config.max_new_tokens = 25
    eps = 0.001
    image_links_list = [[], [cat_tensor]]

    res_cb = []
    for images in image_links_list:
        res_cb.append(
            cb_pipe.generate(
                [prompts[0]], images=[images], generation_config=[generation_config]
            )
        )

    models_path = get_ov_model(model_ids[0])
    for idx, images in enumerate(image_links_list):
        stateful_pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND="SDPA", **get_default_llm_properties())

        res_stateful = stateful_pipe.generate(
            prompts[0], images=images, generation_config=generation_config
        )
        for out_idx, text in enumerate(res_stateful.texts):
            assert text == res_cb[idx][0].texts[out_idx]
            assert (
                abs(res_stateful.scores[out_idx] - res_cb[idx][0].scores[out_idx]) < eps
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
def iteration_images(request):
    return [[request.getfixturevalue(image) for image in bundle] for bundle in request.param]


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("system_message", ["", "You are a helpful assistant."])
@pytest.mark.parametrize("backend", attention_backend)
def test_vlm_pipeline_chat(model_id, system_message, iteration_images, backend):
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

    models_path = get_ov_model(model_id)
    ov_pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    ov_pipe.start_chat(system_message)

    images = iteration_images[0]

    result_from_streamer = []
    res = ov_pipe.generate(
        prompts[0],
        images=images,
        generation_config=generation_config,
        streamer=streamer,
    )
    assert res.texts[0] == "".join(result_from_streamer)

    for image_set in iteration_images[1:]:
        result_from_streamer = []
        res = ov_pipe.generate(
            prompts[1],
            images=image_set,
            generation_config=generation_config,
            streamer=streamer,
        )
        assert res.texts[0] == "".join(result_from_streamer)

    ov_pipe.finish_chat()


@pytest.mark.precommit
@pytest.mark.parametrize("backend", attention_backend)
def test_vlm_get_tokenizer(backend):
    models_path = get_ov_model("katuni4ka/tiny-random-minicpmv-2_6")
    pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    tokenizer = pipe.get_tokenizer()
    tokenizer.encode("")


@pytest.mark.precommit
@pytest.mark.parametrize(
    "config",
    [
        get_beam_search(),
        get_multinomial_all_parameters(),
    ],
)
@pytest.mark.parametrize("backend", attention_backend)
def test_sampling(config, backend, cat_tensor):
    models_path = get_ov_model("katuni4ka/tiny-random-minicpmv-2_6")
    pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    pipe.generate(prompts[0], image=cat_tensor, generation_config=config)


@pytest.mark.precommit
@pytest.mark.parametrize("backend", attention_backend)
def test_perf_metrics(backend, cat_tensor):
    import numpy as np
    from time import perf_counter_ns

    models_path = get_ov_model("katuni4ka/tiny-random-minicpmv-2_6")

    image_tokens_num = 54  # the number of tokens into which this test image is encoded
    max_new_tokens = 30

    start_time = perf_counter_ns()
    pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    start_generate = perf_counter_ns()
    result = pipe.generate(
        prompts[0],
        images=[cat_tensor],
        generation_config=GenerationConfig(max_new_tokens=max_new_tokens),
    )
    generate_time = (perf_counter_ns() - start_generate) / 1_000_000.0
    load_time = (start_generate - start_time) / 1_000_000.0

    perf_metrics = result.perf_metrics

    assert perf_metrics is not None

    assert 0 < perf_metrics.get_load_time() < load_time
    num_tokens = perf_metrics.get_num_generated_tokens()
    assert 0 < num_tokens <= max_new_tokens
    assert 0 < perf_metrics.get_num_input_tokens() < len(prompts[0]) + image_tokens_num
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
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("backend", attention_backend)
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

    models_path = get_ov_model(model_id)
    properties = {
        "DEVICE_PROPERTIES": {
            "NPU": {"NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE", "MAX_PROMPT_LEN": 2048}
        }
    }

    ov_pipe = VLMPipeline(models_path, "NPU", ATTENTION_BACKEND=backend, config=properties)

    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    for image in cat_tensor, handwritten_tensor, car_tensor:
        ov_pipe.generate(
            prompts[0], images=[image], generation_config=generation_config
        )


@pytest.fixture(scope="module", params=[
    ["cat_tensor"], []
])
def image_sequence(request):
    return [request.getfixturevalue(image) for image in request.param]


@pytest.mark.precommit
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_npu_no_image():
    models_path = get_ov_model(model_ids[0])
    properties = {
        "DEVICE_PROPERTIES": {
            "NPU": {"NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE", "MAX_PROMPT_LEN": 2048}
        }
    }

    ov_pipe = VLMPipeline(models_path, "NPU", config=properties)

    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    ov_pipe.generate(
        prompts[0], generation_config=generation_config
    )


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("backend", attention_backend)
def test_vlm_pipeline_chat_streamer_cancel_second_generate(model_id, image_sequence, backend):
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

    models_path = get_ov_model(model_id)
    ov_pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())
    generation_config.ignore_eos = True

    results_with_cancel = ""
    ov_pipe.start_chat()
    results_with_cancel += ov_pipe.generate(
        callback_questions[0], images=image_sequence, generation_config=generation_config
    ).texts[0]
    # doesn't add to results_with_cancel as it should be complitely removed from the history
    ov_pipe.generate(
        callback_questions[1],
        images=image_sequence,
        generation_config=generation_config,
        streamer=streamer,
    )
    results_with_cancel += ov_pipe.generate(
        callback_questions[2], images=image_sequence, generation_config=generation_config
    ).texts[0]
    ov_pipe.finish_chat()

    results = ""
    ov_pipe.start_chat()
    results += ov_pipe.generate(
        callback_questions[0], images=image_sequence, generation_config=generation_config
    ).texts[0]
    results += ov_pipe.generate(
        callback_questions[2], images=image_sequence, generation_config=generation_config
    ).texts[0]
    ov_pipe.finish_chat()

    assert results_with_cancel == results

    results = ""
    ov_pipe.start_chat()
    results += ov_pipe.generate(
        callback_questions[0], images=image_sequence, generation_config=generation_config
    ).texts[0]
    results += ov_pipe.generate(
        callback_questions[2], images=image_sequence, generation_config=generation_config
    ).texts[0]
    ov_pipe.finish_chat()

    assert results_with_cancel == results


@pytest.mark.precommit
@pytest.mark.parametrize("backend", attention_backend)
def test_start_chat_clears_history(backend, image_sequence):
    callback_questions = [
        "Why is the Sun yellow?"
    ]
    models_path = get_ov_model(model_ids[0])
    ov_pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30

    results_first_generate = ""
    ov_pipe.start_chat()
    results_first_generate += ov_pipe.generate(
        callback_questions[0], images=image_sequence, generation_config=generation_config
    ).texts[0]

    results_second_generate = ""
    ov_pipe.start_chat()
    results_second_generate += ov_pipe.generate(
        callback_questions[0], images=image_sequence, generation_config=generation_config
    ).texts[0]

    assert results_first_generate == results_second_generate

@pytest.mark.precommit
def test_start_chat_clears_history_cb_api(image_sequence):
    callback_questions = [
        "Why is the Sun yellow?"
    ]
    models_path = get_ov_model(model_ids[0])
    ov_pipe = ContinuousBatchingPipeline(models_path, SchedulerConfig(), "CPU")
    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 30

    results_first_generate = ""
    ov_pipe.start_chat("You are helpful assistant.")
    results_first_generate = ov_pipe.generate(
        [callback_questions[0]], images=[image_sequence], generation_config=[generation_config]
    )[0].texts[0]

    results_second_generate = ""
    ov_pipe.start_chat("You are helpful assistant.")
    results_second_generate += ov_pipe.generate(
        [callback_questions[0]], images=[image_sequence], generation_config=[generation_config]
    )[0].texts[0]

    assert results_first_generate == results_second_generate


@pytest.mark.precommit
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("backend", attention_backend)
def test_vlm_pipeline_chat_streamer_cancel_first_generate(model_id, image_sequence, backend):
    callback_questions = [
        "Why is the Sun yellow?",
        "1+1=",
    ]

    current_iter = 0
    num_iters = 3
    streamer_generation_result = ""

    def streamer(subword):
        nonlocal current_iter
        current_iter += 1
        nonlocal streamer_generation_result
        streamer_generation_result += subword
        return (
            StreamingStatus.CANCEL
            if current_iter == num_iters
            else StreamingStatus.RUNNING
        )

    models_path = get_ov_model(model_id)
    ov_pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=backend)
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.ignore_eos = True
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    ov_pipe.start_chat()
    _ = ov_pipe.generate(
        callback_questions[0],
        images=image_sequence,
        generation_config=generation_config,
        streamer=streamer,
    )
    res_first = streamer_generation_result
    current_iter = 0
    streamer_generation_result = ""
    _ = ov_pipe.generate(
        callback_questions[0],
        images=image_sequence,
        generation_config=generation_config,
        streamer=streamer,
    )
    ov_pipe.finish_chat()
    res_second = streamer_generation_result

    assert res_first == res_second


def retry(func, exception_type=AssertionError):
    max_retries = 20
    for idx in range(max_retries):
        try:
            return func()
        except exception_type:
            if idx == max_retries - 1:
                raise


def generate(vlm, requests):
    generation_config = vlm.get_generation_config()
    generation_config.max_new_tokens = 30
    vlm.set_generation_config(generation_config)
    vlm.start_chat()
    answers = [vlm.generate(prompt, images=images) for (prompt, images) in requests]
    vlm.finish_chat()
    return answers


@pytest.fixture(scope="module")
def conversation_requests(cat_tensor, car_tensor, handwritten_tensor):
    return [
        ("Describe", [cat_tensor]),
        ("How many images are there?", [car_tensor, handwritten_tensor]),
    ]


tag_inserted_by_template = [
    ("katuni4ka/tiny-random-llava", lambda idx: "<image>"),
    ("katuni4ka/tiny-random-llava-next", lambda idx: "<image>"),
    ("katuni4ka/tiny-random-qwen2vl", lambda idx: "<|vision_start|><|image_pad|><|vision_end|>"),
    ("katuni4ka/tiny-random-qwen2.5-vl", lambda idx: "<|vision_start|><|image_pad|><|vision_end|>"),
    ("katuni4ka/tiny-random-gemma3", lambda idx: "<start_of_image>"),
]

image_id_ignorant =  tag_inserted_by_template + [
    ("katuni4ka/tiny-random-internvl2", lambda idx: "<image>\n"),
]


models_to_tag = image_id_ignorant + [
    # minicpm tracks image number in expanded tags
    ("katuni4ka/tiny-random-minicpmv-2_6", lambda idx: "(<image>./</image>)\n"),
    (
        "katuni4ka/tiny-random-phi3-vision",
        lambda idx: "<|image_" + str(idx + 1) + "|>\n",
    ),
]


@pytest.fixture(scope="module")
def model_and_tag(request):
    model_id, tag = request.param
    if sys.platform == "darwin" and "gemma3" in model_id:
        pytest.xfail(GEMMA3_MACOS_XFAIL_REASON)
    model = get_ov_model(model_id)
    backend = "PA"
    # TODO Remove when PA will be enabled for gemma3
    if model_id == "katuni4ka/tiny-random-gemma3":
        backend = "SDPA"
    vlm = VLMPipeline(model, "CPU", ATTENTION_BACKEND=backend)
    return vlm, tag


@pytest.mark.precommit
class TestImageTags:
    @pytest.mark.parametrize(
        "model_and_tag, model_id",
        [((model_id, tag), model_id) for model_id, tag in tag_inserted_by_template],
        indirect=["model_and_tag"],
    )
    def test_representation(self, model_and_tag, model_id, cat_tensor):
        vlm, tag = model_and_tag
        generation_config = vlm.get_generation_config()
        generation_config.max_new_tokens = 30
        vlm.set_generation_config(generation_config)
        prompt = "Describe"

        align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
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
            automatic_tags = vlm.generate(prompt, images=[cat_tensor])
            reference_tags = vlm.generate(
                templated_prompt, images=[cat_tensor], apply_chat_template=False
            )
            assert automatic_tags.texts == reference_tags.texts
            assert automatic_tags.scores == reference_tags.scores

        retry(workaround_inconsistent_inference)

    @pytest.mark.parametrize("model_and_tag", models_to_tag, indirect=True)
    def test_prepend_native(self, model_and_tag, conversation_requests):
        vlm, tag = model_and_tag

        def workaround_inconsistent_inference():
            answers = generate(vlm, conversation_requests)

            vlm.start_chat()
            native_tag0 = vlm.generate(
                tag(0) + conversation_requests[0][0], images=conversation_requests[0][1]
            )
            assert native_tag0.texts == answers[0].texts
            assert native_tag0.scores == answers[0].scores
            native_tags1 = vlm.generate(
                tag(1) + tag(2) + conversation_requests[1][0], images=conversation_requests[1][1]
            )
            assert native_tags1.texts == answers[1].texts
            assert native_tags1.scores == answers[1].scores
            vlm.finish_chat()

        retry(workaround_inconsistent_inference)

    @pytest.mark.parametrize("model_and_tag", models_to_tag, indirect=True)
    def test_prepend_universal(self, model_and_tag, conversation_requests):
        vlm, _ = model_and_tag

        def workaround_inconsistent_inference():
            answers = generate(vlm, conversation_requests)

            vlm.start_chat()
            universal_tag0 = vlm.generate(
                "<ov_genai_image_0>" + conversation_requests[0][0], images=conversation_requests[0][1]
            )
            assert universal_tag0.texts == answers[0].texts
            assert universal_tag0.scores == answers[0].scores
            universal_tags1 = vlm.generate(
                "<ov_genai_image_1><ov_genai_image_2>" + conversation_requests[1][0],
                images=conversation_requests[1][1],
            )
            assert universal_tags1.texts == answers[1].texts
            assert universal_tags1.scores == answers[1].scores
            vlm.finish_chat()

        retry(workaround_inconsistent_inference)

    @pytest.mark.parametrize("model_and_tag", models_to_tag, indirect=True)
    def test_append(self, model_and_tag, conversation_requests):
        vlm, tag = model_and_tag
        generation_config = vlm.get_generation_config()
        generation_config.max_new_tokens = 30
        vlm.set_generation_config(generation_config)

        def workaround_inconsistent_inference():
            vlm.start_chat()
            native_tag0 = vlm.generate(
                conversation_requests[0][0] + tag(0), images=conversation_requests[0][1]
            )
            native_tags1 = vlm.generate(
                conversation_requests[1][0] + tag(1) + tag(2), images=conversation_requests[1][1]
            )
            vlm.finish_chat()

            vlm.start_chat()
            universal_tag0 = vlm.generate(
                conversation_requests[0][0] + "<ov_genai_image_0>", images=conversation_requests[0][1]
            )
            assert universal_tag0.texts == native_tag0.texts
            assert universal_tag0.scores == native_tag0.scores
            universal_tags1 = vlm.generate(
                conversation_requests[1][0] + "<ov_genai_image_1><ov_genai_image_2>",
                images=conversation_requests[1][1],
            )
            assert universal_tags1.texts == native_tags1.texts
            assert universal_tags1.scores == native_tags1.scores
            vlm.finish_chat()

        retry(workaround_inconsistent_inference)

    @pytest.mark.parametrize("model_and_tag", image_id_ignorant, indirect=True)
    def test_same_reference(self, model_and_tag, cat_tensor):
        vlm, _ = model_and_tag
        generation_config = vlm.get_generation_config()
        generation_config.max_new_tokens = 30
        vlm.set_generation_config(generation_config)

        def workaround_inconsistent_inference():
            one_image = vlm.generate("<ov_genai_image_0>" * 2, images=[cat_tensor])
            two_images = vlm.generate(
                "<ov_genai_image_0><ov_genai_image_1>", images=[cat_tensor, cat_tensor]
            )
            assert one_image.texts == two_images.texts
            assert one_image.scores == two_images.scores

        retry(workaround_inconsistent_inference)

    @pytest.mark.parametrize("model_and_tag", models_to_tag, indirect=True)
    def test_older(self, model_and_tag, car_tensor):
        vlm, _ = model_and_tag
        generation_config = vlm.get_generation_config()
        generation_config.max_new_tokens = 30
        vlm.set_generation_config(generation_config)
        vlm.start_chat()
        vlm.generate("", images=[car_tensor])
        with pytest.raises(RuntimeError):
            vlm.generate("<ov_genai_image_0>", images=[car_tensor])

    @pytest.mark.parametrize("model_and_tag", models_to_tag, indirect=True)
    def test_missing_universal(self, model_and_tag):
        vlm, _ = model_and_tag
        with pytest.raises(RuntimeError):
            vlm.generate("<ov_genai_image_0>")

    @pytest.mark.parametrize("model_and_tag", models_to_tag, indirect=True)
    def test_missing_native(self, model_and_tag):
        vlm, tag = model_and_tag
        with pytest.raises(RuntimeError):
            vlm.generate(tag(0))


@pytest.fixture(scope="module")
def cat_image_336x336(cat_image):
    return cat_image.resize((336, 336))


@pytest.fixture(scope="module")
def cat_image_32x32(cat_image):
    return cat_image.resize((32, 32))


@pytest.mark.precommit
@pytest.mark.parametrize(
    "model_id, image_name, backend",
    [
        pytest.param("katuni4ka/tiny-random-qwen2vl", "cat_image_336x336", "SDPA"),
        pytest.param("katuni4ka/tiny-random-qwen2vl", "cat_image_336x336", "PA"),
        pytest.param("katuni4ka/tiny-random-qwen2.5-vl", "cat_image_336x336", "SDPA"),
        pytest.param("katuni4ka/tiny-random-qwen2.5-vl", "cat_image_336x336", "PA", marks=pytest.mark.xfail(reason="CVS-167316")),
        pytest.param("katuni4ka/tiny-random-gemma3", "cat_image_32x32", "SDPA", marks=pytest.mark.xfail(reason=GEMMA3_MACOS_XFAIL_REASON)) if sys.platform == "darwin" else pytest.param("katuni4ka/tiny-random-gemma3", "cat_image_32x32", "SDPA"),
        pytest.param("katuni4ka/tiny-random-gemma3", "cat_image_32x32", "PA", marks=pytest.mark.xfail(reason="CVS-171180")),
    ],
)
def test_vlm_pipeline_match_optimum_preresized(request, model_id, image_name, backend):
    resized_image = request.getfixturevalue(image_name)

    prompt = "Describe this image."
    max_new_tokens = 100

    model_path = get_ov_model(model_id)

    # Run the model with optimum-intel
    model = OVModelForVisualCausalLM.from_pretrained(model_path, trust_remote_code=True)
    processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Gemma3 input_ids has two bos tokens when running with optimum: one in chat template + "add_bos_token" is set to True in tokenizer_config.json
    if model.config.model_type == "gemma3":
        processor.tokenizer.add_bos_token = False

    templated_prompt = processor.apply_chat_template(conversation,add_generation_prompt=True)
    inputs = processor(text=[templated_prompt], images=[resized_image], padding=True, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    optimum_output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    optimum_text = optimum_output[0]

    # Run the model with GenAI
    vlm = VLMPipeline(model_path, "CPU", ATTENTION_BACKEND=backend)
    genai_output = vlm.generate(prompt, images=[openvino.Tensor(resized_image)], max_new_tokens=max_new_tokens)
    genai_text = genai_output.texts[0]

    assert optimum_text == genai_text
