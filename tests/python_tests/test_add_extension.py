# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import openvino_genai as ov_genai

@pytest.mark.precommit
def test_add_extension():
    print(ov_genai.get_version())
    # I don't know how to get tokenizer path.
    tokenizer_path = ""
    try:
        ov_genai.add_extension(tokenizer_path)
    except:
        assert(False)