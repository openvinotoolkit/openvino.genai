// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

bool prepare_npu_vlm_generation(VLMModelType model_type,
                                size_t prompt_len,
                                GenerationConfig& config,
                                size_t max_prompt_len,
                                size_t kv_cache_capacity);

void report_npu_vlm_generation_limit(bool capacity_limited, const std::vector<GenerationFinishReason>& finish_reasons);

}  // namespace ov::genai
