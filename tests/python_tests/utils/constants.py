# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.properties.hint as hints
import openvino as ov

def get_default_llm_properties():
    return {
        hints.inference_precision : ov.Type.f32,
        hints.kv_cache_precision : ov.Type.f16,
    }