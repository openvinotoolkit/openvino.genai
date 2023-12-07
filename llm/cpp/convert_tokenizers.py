#!/usr/bin/env python3
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import openvino
import ov_tokenizer
import transformers


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("Usage: {sys.argv[0]} <user_ov_extensions_LIB> <SOURCE_MODEL_DIR>")
    ov_tokenizer.init_extension(sys.argv[1])
    tokenizer, detokenizer = ov_tokenizer.convert_tokenizer(
        transformers.AutoTokenizer.from_pretrained(sys.argv[2]), with_decoder=True, streaming_decoder=False)
    openvino.save_model(tokenizer, "tokenizer.xml")
    openvino.save_model(detokenizer, "detokenizer.xml")


if __name__ == '__main__':
    main()
