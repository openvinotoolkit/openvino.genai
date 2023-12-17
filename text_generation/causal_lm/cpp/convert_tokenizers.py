#!/usr/bin/env python3
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino
import openvino_tokenizers
import transformers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--streaming-detokenizer', action=argparse.BooleanOptionalAction)
    parser.add_argument('pretrained_model_name_or_path')
    args = parser.parse_args()
    tokenizer, detokenizer = openvino_tokenizers.convert_tokenizer(
        transformers.AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path),
        with_detokenizer=True, streaming_detokenizer=args.streaming_detokenizer)
    openvino.save_model(tokenizer, "tokenizer.xml")
    openvino.save_model(detokenizer, "detokenizer.xml")


if __name__ == '__main__':
    main()
