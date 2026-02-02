#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
from PIL import Image
from openvino import Tensor
import openvino_genai
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch

def streamer(subword: str) -> bool:
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return openvino_genai.StreamingStatus.RUNNING".


def read_image(path: str) -> Tensor:
    '''

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    '''
    pic = Image.open(path).convert("RGB")
    # image_data = np.array(pic)
    # return Tensor(image_data)

    # 3dim to 4dim with batch size 1
    return Tensor(np.stack([pic], axis=0))

def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', default="", help="Image file or dir with images")
    parser.add_argument('torch_model', default="", help="Path to the directory with torch models")
    args = parser.parse_args()

    rgbs = read_images(args.image_dir)
    prompt = "Please describle this image"

    config_file = "config.yaml"
    pipe = openvino_genai.ModulePipeline(config_file)

    pipe.generate(img1=rgbs[0], prompts_data=[prompt])
    merged_embedding = pipe.get_output("merged_embedding")

    mask = pipe.get_output("mask")
    torch_merged_embedding = torch.from_numpy(merged_embedding.data).to(dtype=torch.bfloat16)
    torch_mask = torch.from_numpy(mask.data)

    tokenizer = AutoTokenizer.from_pretrained(args.torch_model)

    model = AutoModelForImageTextToText.from_pretrained(
        args.torch_model)

    with torch.no_grad():
        outputs_ids = model.generate(
            inputs_embeds=torch_merged_embedding,
            attention_mask=torch_mask,
            max_new_tokens=100
        )
    
    result = tokenizer.decode(outputs_ids[0], skip_special_tokens=True)
    print("Generation result:\n", result)


if __name__ == "__main__":
    main()
