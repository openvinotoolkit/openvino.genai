#!/usr/bin/env python3
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import openvino
import ov_tokenizer
import transformers


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("Usage: {sys.argv[0]} <user_ov_extensions lib> <source model dir>")
    if hasattr(os, "add_dll_directory"):
        for path in os.environ.get("PATH", "").split(";"):
            if os.path.isdir(path):
                os.add_dll_directory(path)
    ov_tokenizer.init_extension(sys.argv[1])
    tokenizer, detokenizer = ov_tokenizer.convert_tokenizer(
        transformers.AutoTokenizer.from_pretrained(sys.argv[2]), with_decoder=True, streaming_decoder=True)
    openvino.save_model(tokenizer, "tokenizer.xml")
    openvino.save_model(detokenizer, "detokenizer.xml")


if __name__ == '__main__':
    main()
