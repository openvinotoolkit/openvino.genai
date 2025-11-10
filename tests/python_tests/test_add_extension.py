# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import openvino_genai as ov_genai
import platform
import os
import openvino_tokenizers;

@pytest.mark.precommit
def test_add_extension():
    # Path to the OpenVINO extension shared library (update as needed).
    os_name = platform.system()
    if os_name == "Windows":
        ov_tokenizer_path = os.path.join(os.path.dirname(openvino_tokenizers.__file__), "lib", "openvino_tokenizers.dll")
    elif os_name == "Linux":
        ov_tokenizer_path = os.path.join(os.path.dirname(openvino_tokenizers.__file__), "lib", "libopenvino_tokenizers.so")
    else:
        print(f"Skipped. Current test only support Windows and Linux")
        return

    try:
        ov_genai.add_extension(ov_tokenizer_path)
    except RuntimeError as e:
        raise RuntimeError(f"Add extension fail, maybe tokenizers' version mismatch. Original error: {e}")