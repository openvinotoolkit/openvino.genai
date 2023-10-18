#!/usr/bin/env python3
# Copyright (C) 023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino
import ov_tokenizer
import sys
import transformers


def validate():
    import numpy as np
    ref = transformers.AutoTokenizer.from_pretrained(sys.argv[2])
    core = openvino.Core()
    core.add_extension(sys.argv[1])
    tokenizer = core.compile_model("tokenizer.xml", "CPU")
    detokenizer = core.compile_model("detokenizer.xml", "CPU")
    for token in range(ref.vocab_size):
        detokenized = ov_tokenizer.unpack_strings(detokenizer(np.array([[token]], np.int32))['string_output'])
        tokenized = tokenizer(ov_tokenizer.pack_strings(detokenized))['input_ids']
        assert tokenized == np.array(ref(detokenized)['input_ids'], np.int32)
        assert detokenized == ov_tokenizer.unpack_strings(detokenizer(tokenized)['string_output'])
    for test_str in (' Hi How', '你好。 你好嗎？'):
        assert test_str == ov_tokenizer.unpack_strings(detokenizer(tokenizer(ov_tokenizer.pack_strings([test_str]))["input_ids"])['string_output'])[0]


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("Usage: {sys.argv[0]} <user_ov_extensions lib> <source model dir>")
    ov_tokenizer.init_extension(sys.argv[1])
    tokenizer, detokenizer = ov_tokenizer.convert_tokenizer(
        transformers.AutoTokenizer.from_pretrained(sys.argv[2]),
        with_decoder=True,
        streaming_decoder=True
    )
    print(tokenizer, detokenizer)
    openvino.save_model(tokenizer, "tokenizer.xml")
    openvino.save_model(detokenizer, "detokenizer.xml")

    # validate()  # TODO: enable after ov_tokenizer is fixed


if __name__ == '__main__':
    main()
