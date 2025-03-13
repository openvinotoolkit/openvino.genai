# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

def generate_images(path):
    from PIL import Image
    import numpy as np
    import requests
    res = 28, 28
    lines = np.arange(res[0] * res[1] * 3, dtype=np.uint8) % 255
    lines = lines.reshape([*res, 3])
    lines_image = Image.fromarray(lines)
    cat = Image.open(requests.get("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11", stream=True).raw).convert('RGB')

    lines_image_path = path + "/lines.png"
    cat_path = path + "/cat.png"
    lines_image.save(lines_image_path)
    cat.save(cat_path)
    yield lines_image_path, cat_path

    os.remove(lines_image_path)
    os.remove(cat_path)

class TestEncryptedVLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["tiny-random-minicpmv-2_6"], indirect=True)
    @pytest.mark.parametrize("sample_args", ["Describe the images."])

    def test_sample_encrypted_lm(self, convert_model, sample_args, tmp_path):
        generate_images(tmp_path)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/encrypted_model_vlm.py")
        py_command = [sys.executable, py_script, convert_model, tmp_path, sample_args]
        py_result = run_sample(py_command)

        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'encrypted_model_vlm')
        cpp_command =[cpp_sample, convert_model, tmp_path, sample_args]
        cpp_result = run_sample(cpp_command)

        # Test common sample
        py_common_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/visual_language_chat.py")
        py_common_command = [sys.executable, py_common_script, convert_model, tmp_path]
        py_common_result = run_sample(py_common_command, sample_args)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
        assert py_result.stdout == py_common_result.stdout, f"Results should match"
