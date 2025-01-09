# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_tokenizers
import openvino
import pytest
import transformers
from optimum.intel.openvino import OVModelForVisualCausalLM
from openvino_genai import VLMPipeline, GenerationConfig
from common import get_image_by_link, get_beam_search, get_multinomial_all_parameters, get_default_properties

def get_ov_model(model_id, cache):
    model_dir = cache.mkdir(model_id.split('/')[-1])
    if (model_dir / "openvino_language_model.xml").exists():
        return model_dir
    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer.save_pretrained(model_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(processor.tokenizer, with_detokenizer=True)
    openvino.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, model_dir / "openvino_detokenizer.xml")
    model = OVModelForVisualCausalLM.from_pretrained(model_id, compile=False, device="CPU", export=True, load_in_8bit=False, trust_remote_code=True, ov_config=get_default_properties())
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

@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("model_id", [
    "katuni4ka/tiny-random-minicpmv-2_6",
    "katuni4ka/tiny-random-phi3-vision",
])
def test_vlm_pipeline(model_id, cache):
    def streamer(word: str) -> bool:
        return False

    models_path = get_ov_model(model_id, cache)
    generation_config = GenerationConfig(max_new_tokens=100)

    for links in image_links_for_testing:
        images = []
        for link in links:
            images.append(get_image_by_link(link))

        ov_pipe = VLMPipeline(models_path, "CPU")
        ov_pipe.start_chat()

        ov_pipe.generate(prompts[0], images=images, generation_config=generation_config, streamer=streamer)

        for prompt in prompts[1:]:
            ov_pipe.generate(prompt, generation_config=generation_config, streamer=streamer)

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
    models_path = get_ov_model("katuni4ka/tiny-random-minicpmv-2_6", cache)

    images = [get_image_by_link(image_links[0])]

    pipe = VLMPipeline(models_path, "CPU")
    result = pipe.generate(prompts[0], images=images, generation_config=GenerationConfig(max_new_tokens=30))

    perf_metrics = result.perf_metrics

    assert perf_metrics is not None

    assert 0 < perf_metrics.get_load_time() < 2000
    assert 0 < perf_metrics.get_num_generated_tokens() < 100
    assert 0 < perf_metrics.get_num_input_tokens() < 100
    assert 0 < perf_metrics.get_ttft().mean < 1000
    assert 0 < perf_metrics.get_tpot().mean < 100
    assert 0 < perf_metrics.get_ipot().mean < 100
    assert 0 < perf_metrics.get_throughput().mean < 1000
    assert 0 < perf_metrics.get_inference_duration().mean < 1000
    assert 0 < perf_metrics.get_generate_duration().mean < 1000
    assert 0 < perf_metrics.get_tokenization_duration().mean < 100
    assert 0 < perf_metrics.get_detokenization_duration().mean < 10
    assert 0 < perf_metrics.get_prepare_embeddings_duration().mean < 100

    # assert that calculating statistics manually from the raw counters we get the same results as from PerfMetrics
    vlm_raw_metrics = perf_metrics.vlm_raw_metrics

    raw_dur = np.array(vlm_raw_metrics.prepare_embeddings_durations) / 1000
    mean_dur, std_dur = perf_metrics.get_prepare_embeddings_duration()
    assert np.allclose(mean_dur, np.mean(raw_dur))
    assert np.allclose(std_dur, np.std(raw_dur))
