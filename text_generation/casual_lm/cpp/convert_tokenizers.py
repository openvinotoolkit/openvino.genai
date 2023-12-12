#!/usr/bin/env python3
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import openvino
import openvino_tokenizers
import transformers


def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: {sys.argv[0]} <SOURCE_MODEL_DIR>")
    tokenizer, detokenizer = openvino_tokenizers.convert_tokenizer(
        transformers.AutoTokenizer.from_pretrained(sys.argv[1]), with_detokenizer=True, streaming_decoder=True)
    openvino.save_model(tokenizer, "tokenizer.xml")
    openvino.save_model(detokenizer, "detokenizer.xml")


if __name__ == '__main__':
    main()
