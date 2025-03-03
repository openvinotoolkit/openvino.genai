# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.properties.hint as hints
import openvino.properties as props
import openvino as ov

def get_default_llm_properties():
    return {
        hints.inference_precision : ov.Type.f32,
        hints.kv_cache_precision : ov.Type.f16,
    }

OV_MODEL_FILENAME = "openvino_model.xml"
OV_TOKENIZER_FILENAME = "openvino_tokenizer.xml"
OV_DETOKENIZER_FILENAME = "openvino_detokenizer.xml"

def get_disabled_mmap_ov_config():
    return {
        props.enable_mmap : False
    }
