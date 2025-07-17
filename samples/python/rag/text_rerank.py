#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("query")
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well

    config = openvino_genai.TextRerankPipeline.Config()
    config.top_n = 3

    pipeline = openvino_genai.TextRerankPipeline(args.model_dir, device, config)

    rerank_result = pipeline.rerank(args.query, args.texts)

    print("Reranked documents:")
    for index, score in rerank_result:
        print(f"Document {index} (score: {score:.4f}): {args.texts[index]}")


if __name__ == "__main__":
    main()
