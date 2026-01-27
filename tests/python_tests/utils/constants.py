# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.properties.hint as hints
import openvino.properties as props
import openvino as ov
import os
from importlib import metadata
from datetime import datetime
from optimum.intel.openvino.utils import TemporaryDirectory
from pathlib import Path


def get_default_llm_properties():
    return {
        hints.inference_precision: ov.Type.f32,
        hints.kv_cache_precision: ov.Type.f16,
    }


def extra_generate_kwargs():
    from optimum.intel.utils.import_utils import is_transformers_version
    additional_args = {}
    if is_transformers_version(">=", "4.51"):
        additional_args["use_model_defaults"] = False

    return additional_args


OV_MODEL_FILENAME = "openvino_model.xml"
OV_TOKENIZER_FILENAME = "openvino_tokenizer.xml"
OV_DETOKENIZER_FILENAME = "openvino_detokenizer.xml"


def get_disabled_mmap_ov_config():
    return {props.enable_mmap: False}


def get_ov_cache_dir(temp_dir=TemporaryDirectory()):
    if "OV_CACHE" in os.environ:
        date_subfolder = datetime.now().strftime("%Y%m%d")
        ov_cache = os.path.join(os.environ["OV_CACHE"], date_subfolder)
        try:
            optimum_intel_version = metadata.version("optimum-intel")
            transformers_version = metadata.version("transformers")
            ov_cache = os.path.join(ov_cache, f"optimum-intel-{optimum_intel_version}_transformers-{transformers_version}")
        except metadata.PackageNotFoundError:
            pass
        ov_cache_path = Path(ov_cache)
        ov_cache_path.mkdir(parents=True, exist_ok=True)
        return ov_cache_path
    else:
        ov_cache = temp_dir.name
        return Path(ov_cache)


def get_ov_cache_downloaded_models_dir(temp_dir=TemporaryDirectory()):
    ov_cache = get_ov_cache_dir(temp_dir)
    downloaded_dir = Path(ov_cache) / "downloaded_models"
    downloaded_dir.mkdir(parents=True, exist_ok=True)
    return downloaded_dir


def get_ov_cache_converted_models_dir(temp_dir=TemporaryDirectory()):
    ov_cache = get_ov_cache_dir(temp_dir)
    converted_dir = Path(ov_cache) / "converted_models"
    converted_dir.mkdir(parents=True, exist_ok=True)
    return converted_dir
