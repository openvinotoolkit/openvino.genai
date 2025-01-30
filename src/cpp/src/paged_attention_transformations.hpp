// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace genai {

/**
 * Per layer KV cache size configuration
 */
struct KVHeadConfig {
    size_t num_v_heads, num_k_heads;
    size_t v_head_size, k_head_size;
};

namespace utils {

/** Applies transformations to the ov::Model to enable paged attention inference.
 * @param model Pointer to the ov::Model representing one of the supported LLM architectures.
 * @param device_config Configuration struct for inferencing device specifics.
 * @param per_layer_cache_control If true, then the transformations will enable per-layer control of KV cache blocks, allowing to specify
 * different sets of KV cache blocks for different attention layers. If false, then the KV cache block structure will be identical across all
 * decoder layers.
 * @return Information about each decoder layer configuration 
 */
std::vector<KVHeadConfig> apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, bool per_layer_cache_control = false, bool allow_cache_rotation = false);

void apply_gather_before_matmul_transformation(std::shared_ptr<ov::Model> model);

}  // namespace utils
}  // namespace genai
}  // namespace ov
