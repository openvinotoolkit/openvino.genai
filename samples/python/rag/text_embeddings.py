#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well

    config = openvino_genai.TextEmbeddingPipeline.Config()
    config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN

    pipeline = openvino_genai.TextEmbeddingPipeline(args.model_dir, device, config)

    text_embeddings = pipeline.embed_documents(args.texts)
    query_embeddings = pipeline.embed_query("What is the capital of France?")


if "__main__" == __name__:
    main()
