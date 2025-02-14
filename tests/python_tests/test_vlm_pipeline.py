# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_tokenizers
import openvino
import pytest
import platform
import sys
import transformers
from optimum.intel.openvino import OVModelForVisualCausalLM
from openvino_genai import VLMPipeline, GenerationConfig, SchedulerConfig, ContinuousBatchingPipeline, GenerationStatus, StreamingStatus

from utils.network import retry_request
from utils.generation_config import get_beam_search, get_multinomial_all_parameters, get_greedy
from utils.constants import get_default_llm_properties

def get_ov_model(model_id, cache):
    model_dir = cache.mkdir(model_id.split('/')[-1])
    if (model_dir / "openvino_language_model.xml").exists():
        return model_dir
    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    processor = retry_request(lambda: transformers.AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        **align_with_optimum_cli,
    ))
    processor.tokenizer.save_pretrained(model_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(processor.tokenizer, with_detokenizer=True)
    openvino.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, model_dir / "openvino_detokenizer.xml")
    model = retry_request(lambda: OVModelForVisualCausalLM.from_pretrained(model_id, compile=False, device="CPU", export=True, load_in_8bit=False, trust_remote_code=True, ov_config=get_default_llm_properties()))
    if processor.tokenizer.chat_template is not None:
        processor.chat_template = processor.tokenizer.chat_template  # It seems that tiny-random-phi3-vision is saved incorrectly. That line works this around.
    processor.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    return model_dir


prompts = [
    "What is on the image?",
    "What is special about this image?",
]

image_links = [
    "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://github.com/user-attachments/assets/8c9ae017-7837-4abc-ae92-c1054c9ec350"
]

image_links_for_testing = [
    [],
    [image_links[0]],
    [image_links[0], image_links[2], image_links[1]]
]

model_ids = [
    "katuni4ka/tiny-random-minicpmv-2_6",
    "katuni4ka/tiny-random-phi3-vision",
    "katuni4ka/tiny-random-llava",
    "katuni4ka/tiny-random-llava-next",
    "katuni4ka/tiny-random-qwen2vl",
]


def get_image_by_link(link):
    from PIL import Image
    import requests
    from openvino import Tensor
    import numpy as np

    image = Image.open(requests.get(link, stream=True).raw)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_data = np.array((np.array(image.getdata()) - 128).astype(np.byte)).reshape(1, image.size[1], image.size[0], 3)
    return Tensor(image_data)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("model_id", model_ids)
def test_vlm_pipeline(model_id, cache):
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

    models_path = get_ov_model(model_id, cache)
    ov_pipe = VLMPipeline(models_path, "CPU")
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    for links in image_links_for_testing:
        images = []
        for link in links:
            images.append(get_image_by_link(link))

        result_from_streamer = []
        res = ov_pipe.generate(prompts[0], images=images, generation_config=generation_config, streamer=streamer)
        assert res.texts[0] == ''.join(result_from_streamer)


configs = [
    get_greedy(),
    get_beam_search(),
]

@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("config", configs)
def test_vlm_continuous_batching_generate_vs_add_request(config, cache):
    scheduler_config = SchedulerConfig()
    models_path = get_ov_model(model_ids[0], cache)
    ov_pipe = VLMPipeline(models_path, "CPU", scheduler_config=scheduler_config, **get_default_llm_properties())
    generation_config = config
    generation_config.max_new_tokens = 30
    eps = 0.001
    image_links_list = [
        [],
        [image_links[0]]
    ]

    res_generate = []
    for links in image_links_list:
        images = []
        for link in links:
            images.append(get_image_by_link(link))

        res_generate.append(ov_pipe.generate(prompts[0], images=images, generation_config=generation_config))

    cb_pipe = ContinuousBatchingPipeline(models_path, scheduler_config=scheduler_config, device="CPU", properties=get_default_llm_properties())
    tokenizer = cb_pipe.get_tokenizer()

    for idx, links in enumerate(image_links_list):
        images = []
        for link in links:
            images.append(get_image_by_link(link))
        handle = cb_pipe.add_request(idx, prompts[0], images, generation_config)
        while handle.get_status() != GenerationStatus.FINISHED:
            cb_pipe.step()
        outputs = handle.read_all()
        for out_idx, output in enumerate(outputs):
            text = tokenizer.decode(output.generated_ids)
            assert text == res_generate[idx].texts[out_idx]
            assert abs(output.score - res_generate[idx].scores[out_idx]) < eps


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("config", configs)
def test_vlm_continuous_batching_vs_stateful(config, cache):
    scheduler_config = SchedulerConfig()
    models_path = get_ov_model(model_ids[0], cache)
    cb_pipe = ContinuousBatchingPipeline(models_path, scheduler_config=scheduler_config, device="CPU", properties=get_default_llm_properties())
    generation_config = config
    generation_config.max_new_tokens = 25
    eps = 0.001
    image_links_list = [
        [],
        [image_links[0]]
    ]

    res_cb = []
    for links in image_links_list:
        images = []
        for link in links:
            images.append(get_image_by_link(link))

        res_cb.append(cb_pipe.generate([prompts[0]], images=[images], generation_config=[generation_config]))

    models_path = get_ov_model(model_ids[0], cache)
    for idx, links in enumerate(image_links_list):
        stateful_pipe = VLMPipeline(models_path, "CPU", **get_default_llm_properties())

        images = []
        for link in links:
            images.append(get_image_by_link(link))

        res_stateful = stateful_pipe.generate(prompts[0], images=images, generation_config=generation_config)
        for out_idx, text in enumerate(res_stateful.texts):
            assert text == res_cb[idx][0].m_generation_ids[out_idx]
            assert abs(res_stateful.scores[out_idx] - res_cb[idx][0].m_scores[out_idx]) < eps



@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("config", configs)
def test_vlm_with_scheduler_vs_default(config, cache):
    scheduler_config = SchedulerConfig()
    models_path = get_ov_model(model_ids[0], cache)
    cb_pipe = VLMPipeline(models_path, "CPU", scheduler_config=scheduler_config, **get_default_llm_properties())
    generation_config = config
    generation_config.max_new_tokens = 25
    eps = 0.001
    image_links_list = [
        [],
        [image_links[0]]
    ]

    res_cb = []
    for links in image_links_list:
        images = []
        for link in links:
            images.append(get_image_by_link(link))

        res_cb.append(cb_pipe.generate(prompts[0], images=images, generation_config=generation_config))

    models_path = get_ov_model(model_ids[0], cache)
    for idx, links in enumerate(image_links_list):
        stateful_pipe = VLMPipeline(models_path, "CPU", **get_default_llm_properties())

        images = []
        for link in links:
            images.append(get_image_by_link(link))

        res_stateful = stateful_pipe.generate(prompts[0], images=images, generation_config=generation_config)
        for out_idx, text in enumerate(res_stateful.texts):
            assert text == res_cb[idx].texts[out_idx]
            assert abs(res_stateful.scores[out_idx] - res_cb[idx].scores[out_idx]) < eps


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("system_message", ["", "You are a helpful assistant."])
def test_vlm_pipeline_chat(model_id, system_message, cache):
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

    models_path = get_ov_model(model_id, cache)
    ov_pipe = VLMPipeline(models_path, "CPU")
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    for links in image_links_for_testing:
        images = []
        for link in links:
            images.append(get_image_by_link(link))

        ov_pipe.start_chat(system_message)

        result_from_streamer = []
        res = ov_pipe.generate(prompts[0], images=images, generation_config=generation_config, streamer=streamer)
        assert res.texts[0] == ''.join(result_from_streamer)

        for prompt in prompts[1:]:
            result_from_streamer = []
            res = ov_pipe.generate(prompt, generation_config=generation_config, streamer=streamer)
            assert res.texts[0] == ''.join(result_from_streamer)

        ov_pipe.finish_chat()


@pytest.mark.precommit
@pytest.mark.nightly
def test_vlm_get_tokenizer(cache):
    models_path = get_ov_model("katuni4ka/tiny-random-minicpmv-2_6", cache)
    pipe = VLMPipeline(models_path, "CPU")
    tokenizer = pipe.get_tokenizer()
    tokenizer.encode("")


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("config", [
    get_beam_search(),
    get_multinomial_all_parameters(),
])
def test_sampling(config, cache):
    models_path = get_ov_model("katuni4ka/tiny-random-minicpmv-2_6", cache)
    image = get_image_by_link(image_links[0])
    pipe = VLMPipeline(models_path, "CPU")
    pipe.generate(prompts[0], image=image, generation_config=config)


@pytest.mark.precommit
@pytest.mark.nightly
def test_perf_metrics(cache):
    import numpy as np
    from time import perf_counter_ns
    models_path = get_ov_model("katuni4ka/tiny-random-minicpmv-2_6", cache)

    images = [get_image_by_link(image_links[0])]
    image_tokens_num = 54 # the number of tokens into which this test image is encoded
    max_new_tokens = 30

    start_time = perf_counter_ns()
    pipe = VLMPipeline(models_path, "CPU")
    start_generate = perf_counter_ns()
    result = pipe.generate(prompts[0], images=images, generation_config=GenerationConfig(max_new_tokens=max_new_tokens))
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
    assert num_tokens / (generate_time / 1000.0) < perf_metrics.get_throughput().mean < num_tokens / ((generate_time - perf_metrics.get_ttft().mean) / 1000.0)
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
    assert 0 <= perf_metrics.get_prepare_embeddings_duration().std < squared_generate_time

    # assert that calculating statistics manually from the raw counters we get the same results as from PerfMetrics
    vlm_raw_metrics = perf_metrics.vlm_raw_metrics

    raw_dur = np.array(vlm_raw_metrics.prepare_embeddings_durations) / 1000.0
    mean_dur, std_dur = perf_metrics.get_prepare_embeddings_duration()
    assert np.allclose(mean_dur, np.mean(raw_dur))
    assert np.allclose(std_dur, np.std(raw_dur))


@pytest.mark.precommit
@pytest.mark.nightly
# FIXME: katuni4ka/tiny-random-qwen2vl - fails on NPU
@pytest.mark.parametrize("model_id", model_ids[:-1])
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_vlm_npu_no_exception(model_id, cache):
    models_path = get_ov_model(model_ids[0], cache)
    properties = {
       "DEVICE_PROPERTIES":
       {
           "NPU": { "NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE" }
       }
    }

    ov_pipe = VLMPipeline(models_path, "NPU", config=properties)

    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    for link in image_links_for_testing[2]:
        image = get_image_by_link(link)
        out = ov_pipe.generate(prompts[0], images=[image], generation_config=generation_config)


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("iteration_images", [image_links_for_testing[1], []])
def test_vlm_pipeline_chat_streamer_cancel_second_generate(model_id, iteration_images, cache):
    callback_questions = [
        '1+1=',
        'Why is the Sun yellow?',
        'What is the previous answer?'
    ]

    current_iter = 0
    num_iters = 3
    def streamer(subword):
        nonlocal current_iter
        current_iter += 1
        return StreamingStatus.CANCEL if current_iter == num_iters else StreamingStatus.RUNNING


    models_path = get_ov_model(model_id, cache)
    ov_pipe = VLMPipeline(models_path, "CPU")
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())
    generation_config.ignore_eos = True

    images = []
    for link in iteration_images:
        images.append(get_image_by_link(link))

    results_with_cancel = ""
    ov_pipe.start_chat()
    results_with_cancel += ov_pipe.generate(callback_questions[0], images=images, generation_config=generation_config).texts[0]
    # doesn't add to results_with_cancel as it should be complitely removed from the history
    ov_pipe.generate(callback_questions[1], images=images, generation_config=generation_config, streamer=streamer)
    results_with_cancel += ov_pipe.generate(callback_questions[2], images=images, generation_config=generation_config).texts[0]
    ov_pipe.finish_chat()
    
    results = ""
    ov_pipe.start_chat()
    results += ov_pipe.generate(callback_questions[0], images=images, generation_config=generation_config).texts[0]

    generation_config.ignore_eos = True
    results += ov_pipe.generate(callback_questions[2], images=images, generation_config=generation_config).texts[0]
    ov_pipe.finish_chat()

    assert(results_with_cancel == results)

    results = ""
    ov_pipe.start_chat()
    results += ov_pipe.generate(callback_questions[0], images=images, generation_config=generation_config).texts[0]
    results += ov_pipe.generate(callback_questions[2], images=images, generation_config=generation_config).texts[0]
    ov_pipe.finish_chat()

    assert results_with_cancel == results


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("iteration_images", [image_links_for_testing[1], []])
def test_vlm_pipeline_chat_streamer_cancel_first_generate(model_id, iteration_images, cache):
    callback_questions = [
        'Why is the Sun yellow?',
        '1+1=',
    ]

    current_iter = 0
    num_iters = 3
    def streamer(subword):
        nonlocal current_iter
        current_iter += 1
        return StreamingStatus.CANCEL if current_iter == num_iters else StreamingStatus.RUNNING

    models_path = get_ov_model(model_id, cache)
    ov_pipe = VLMPipeline(models_path, "CPU")
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.ignore_eos = True
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    images = []
    for link in iteration_images:
        images.append(get_image_by_link(link))

    ov_pipe.start_chat()
    res_first = ov_pipe.generate(callback_questions[0], images=images, generation_config=generation_config, streamer=streamer).texts[0]
    current_iter = 0
    res_second = ov_pipe.generate(callback_questions[0], images=images, generation_config=generation_config, streamer=streamer).texts[0]
    ov_pipe.finish_chat()
    
    assert res_first == res_second


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("model_id", model_ids)
@pytest.mark.parametrize("iteration_images", [[[], image_links_for_testing[1]], [image_links_for_testing[1], image_links_for_testing[1]], [[], image_links_for_testing[1], []]])
def test_vlm_pipeline_chat_image_combination(model_id, iteration_images, cache):
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

    models_path = get_ov_model(model_id, cache)
    ov_pipe = VLMPipeline(models_path, "CPU")
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 30
    generation_config.set_eos_token_id(ov_pipe.get_tokenizer().get_eos_token_id())

    for images_links in iteration_images:
        ov_pipe.start_chat()

        images = []
        for link in images_links:
            images.append(get_image_by_link(link))

        result_from_streamer = []
        res = ov_pipe.generate(prompts[0], images=images, generation_config=generation_config, streamer=streamer)
        assert res.texts[0] == ''.join(result_from_streamer)

        ov_pipe.finish_chat()
