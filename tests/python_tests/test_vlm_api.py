# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai
import pytest
import gc
import os
import numpy as np
from PIL import Image
from multiprocessing import Process

from openvino_genai import VLMPipeline
from openvino import Tensor
from common import get_greedy, get_image_by_link, get_beam_search, get_greedy, get_multinomial_all_parameters

def get_ov_model(model_dir):
    os.system("optimum-cli export openvino -m openbmb/MiniCPM-V-2_6 " + model_dir + " --trust-remote-code")
    os.system("convert_tokenizer openbmb/MiniCPM-V-2_6 --with-detokenizer --trust-remote-code -o " + model_dir)
    return model_dir


sampling_configs = [
    get_beam_search(),
    get_greedy(),
    get_multinomial_all_parameters()
]

prompts = [
    "What is on the image?",
    "What is special about this image?",
    "Tell me more about this image."
]

image_links = [
    "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://github.com/user-attachments/assets/8c9ae017-7837-4abc-ae92-c1054c9ec350"
]

image_links_for_testing = [
    [],
    [image_links[0]],
    [image_links[1], image_links[0]],
    [image_links[0], image_links[2], image_links[1]]
]

@pytest.mark.parametrize("generation_config", sampling_configs)
@pytest.mark.parametrize("links", image_links_for_testing)
@pytest.mark.precommit
def test_vlm_pipeline(tmp_path, generation_config, links):
    import os

    def streamer(word: str) -> bool:
        print(word, end="")
        return False

    model_path = get_ov_model(os.path.join(tmp_path, "miniCPM"))
    images = []
    for link in links:
        images.append(get_image_by_link(link))

    pipe = VLMPipeline(model_path, "CPU")
    pipe.start_chat()

    if len(images):
        pipe.generate(prompts[0], images=images, generation_config=generation_config, streamer=streamer)
    else:
        pipe.generate(prompts[0], generation_config=generation_config, streamer=streamer)

    for prompt in prompts[1:]:
        pipe.generate(prompt, generation_config=generation_config, streamer=streamer)

    pipe.finish_chat()
    gc.collect()


