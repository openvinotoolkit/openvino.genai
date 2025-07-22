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

    pipeline = openvino_genai.TextEmbeddingPipeline(
        models_path,
        "${props.device || 'CPU'}",
        pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN,
        normalize = True
    )

    documents_embeddings = pipeline.embed_documents(documents)
    query_embeddings = pipeline.embed_query("What is the capital of France?")


if "__main__" == __name__:
    main()
