# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_tokenizers
import openvino
import pytest
import transformers
from optimum.intel.openvino import OVModelForVisualCausalLM
from openvino_genai import VLMPipeline
from common import get_greedy, get_image_by_link, get_beam_search, get_greedy, get_multinomial_all_parameters

def get_ov_model(model_dir):
    if (model_dir / "openvino_language_model.xml").exists():
        return model_dir
    model_id = "openbmb/MiniCPM-V-2_6"
    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer.save_pretrained(model_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(processor.tokenizer, with_detokenizer=True)
    openvino.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
    openvino.save_model(ov_detokenizer, model_dir / "openvino_detokenizer.xml")
    model = OVModelForVisualCausalLM.from_pretrained(model_id, compile=False, device="CPU", export=True, trust_remote_code=True)
    model.config.save_pretrained(model_dir)
    model.generation_config.save_pretrained(model_dir)
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
def test_vlm_pipeline(cache):
    def streamer(word: str) -> bool:
        return False

    models_path = get_ov_model(cache.mkdir("MiniCPM-V-2_6"))

    for links in image_links_for_testing:
        images = []
        for link in links:
            images.append(get_image_by_link(link))

        pipe = VLMPipeline(str(models_path), "CPU")
        pipe.start_chat()

        pipe.generate(prompts[0], images=images, generation_config=get_greedy(), streamer=streamer)

        for prompt in prompts[1:]:
            pipe.generate(prompt, generation_config=get_greedy(), streamer=streamer)

        pipe.finish_chat()


@pytest.mark.precommit
@pytest.mark.nightly
def test_vlm_get_tokenizer(cache):
    models_path = get_ov_model(cache.mkdir("MiniCPM-V-2_6"))
    pipe = VLMPipeline(str(models_path), "CPU")
    tokenizer = pipe.get_tokenizer()
    tokenizer.encode("")


@pytest.mark.precommit
@pytest.mark.nightly
@pytest.mark.parametrize("config", [
    get_beam_search(),
    get_multinomial_all_parameters(),
])
@pytest.mark.skip("Enable after sampler are enabled")
def test_sampling(config, cache):
    models_path = get_ov_model(cache.mkdir("MiniCPM-V-2_6"))
    image = get_image_by_link(image_links[0])
    pipe = VLMPipeline(str(models_path), "CPU")
    pipe.generate(prompts[0], image=image, generation_config=config)
