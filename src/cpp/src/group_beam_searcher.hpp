// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/runtime/tensor.hpp>
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace ov {
    EncodedResults beam_search(ov::InferRequest& lm, ov::Tensor prompts, ov::Tensor attentin_mask, GenerationConfig sampling_params);
}
