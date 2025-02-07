# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import shutil
from typing import List, Tuple
from pathlib import Path

import openvino
import openvino_genai as ov_genai

import openvino_genai as ov_genai
from utils.test_data import get_models_list
from utils.hf_utils import download_and_convert_model


# looks like w/a - to remove?
"""rt_info has the highest priority. Delete it to respect configs."""
def delete_rt_info(configs: List[Tuple], temp_path):
    core = openvino.Core()
    core.set_property({'ENABLE_MMAP': False})
    for model_path in temp_path / "openvino_tokenizer.xml", temp_path / "openvino_detokenizer.xml":
        tokenizer = core.read_model(model_path)
        rt_info = tokenizer.get_rt_info()
        for config, _ in configs:
            for key in config.keys():
                try:
                    del rt_info[key]
                except KeyError:
                    pass
        openvino.save_model(tokenizer, model_path)


@pytest.fixture(scope="module")
def model_tokenizers_tmp_path(tmpdir_factory):
    model_id, models_path, _, _, _ = read_model(get_models_list()[0])
    temp_path = tmpdir_factory.mktemp(model_id.replace('/', '_'))

    # If tokens were not found in IR, it fallback to reading from config.
    # There was no easy way to add tokens to IR in tests, so we remove them
    # and set tokens in configs and to check if they are read and validated correctly.
    import openvino as ov

    core = ov.Core()

    # copy openvino converted model and tokenizers
    for pattern in ['*.xml', '*.bin']:
        for src_file in models_path.glob(pattern):

            # Update files if they are openvino_tokenizer.xml or openvino_detokenizer.xml
            if src_file.name in ['openvino_tokenizer.xml', 'openvino_detokenizer.xml']:
                if src_file.exists():
                    # Load the XML content
                    ov_model = core.read_model(src_file)
                    # Add empty rt_info so that tokens will be read from config instead of IR
                    ov_model.set_rt_info("pad_token_id", "")
                    ov_model.set_rt_info("eos_token_id", "")
                    ov_model.set_rt_info("chat_template", "")
                    ov.save_model(ov_model, str(temp_path / src_file.name))

            if src_file in ['openvino_tokenizer.bin', 'openvino_detokenizer.bin']:
                continue

            if src_file.is_file():
                shutil.copy(src_file, temp_path / src_file.name)

    yield model_id, Path(temp_path)


@pytest.fixture(scope="module")
def model_tmp_path(tmpdir_factory):
    models_path, _, _ = download_and_convert_model(get_models_list()[0])
    model_id = get_models_list()[0][0]
    temp_path = tmpdir_factory.mktemp(model_id.replace('/', '_'))

    # copy openvino converted model and tokenizers
    for pattern in ['*.xml', '*.bin']:
        for src_file in models_path.glob(pattern):
            if src_file.is_file():
                shutil.copy(src_file, temp_path / src_file.name)

    yield model_id, Path(temp_path)



