# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import openvino
import pytest
from data.models import get_models_list

from utils.hugging_face import download_and_convert_model


@pytest.fixture(scope="module")
def model_tmp_path(tmpdir_factory):
    model_id = get_models_list()[0]
    _, _, models_path = download_and_convert_model(model_id)

    temp_path = tmpdir_factory.mktemp(model_id.replace("/", "_"))

    # copy openvino converted model and tokenizers
    for pattern in ["*.xml", "*.bin"]:
        for src_file in models_path.glob(pattern):
            if src_file.is_file():
                shutil.copy(src_file, temp_path / src_file.name)

    yield model_id, Path(temp_path)


def delete_rt_info(configs: list[tuple], temp_path):
    """rt_info has the highest priority. Delete it to respect configs."""
    core = openvino.Core()
    core.set_property({"ENABLE_MMAP": False})
    for model_path in (
        temp_path / "openvino_tokenizer.xml",
        temp_path / "openvino_detokenizer.xml",
    ):
        tokenizer = core.read_model(model_path)
        rt_info = tokenizer.get_rt_info()
        for config, _ in configs:
            for key in config.keys():
                # tokenizer_config.json contains strings instead of ids so the keys don't have "_id".
                for modified_key in (key, key + "_id"):
                    try:
                        del rt_info[modified_key]
                    except KeyError:
                        pass
        openvino.save_model(tokenizer, model_path)
