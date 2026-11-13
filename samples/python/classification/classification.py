#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image


def read_image(path: str) -> ov.Tensor:
    with Image.open(path) as image:
        return ov.Tensor(np.array(image.convert("RGB")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("text")
    parser.add_argument("images", nargs="*", help="Images for a visual language classifier")
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well

    pipeline = openvino_genai.ClassificationPipeline(args.model_dir, device)

    # Images are referenced in the text by <ov_genai_image_N> tags; untagged images are placed as in VLMPipeline.
    result = pipeline.classify(args.text, images=[read_image(path) for path in args.images])

    for label_id, score in enumerate(result.scores[0]):
        print(f"{label_id}: {score:.4f}")


if __name__ == "__main__":
    main()
