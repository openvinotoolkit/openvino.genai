// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "device_config.hpp"

namespace ov {
namespace genai {
namespace utils {


/** Applies transformations to the ov::Model to enable paged attention inference.
 * @param model Pointer to the ov::Model representing one of the supported LLM architectures.
 * @param device_config Configuration struct for inferencing device specifics.
 * @param per_layer_cache_control If true, then the transformations will enable per-layer control of KV cache blocks, allowing to specify
 * different sets of KV cache blocks for different attention layers. If false, then the KV cache block structure will be identical across all
 * decoder layers.
 */
void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, DeviceConfig& device_config, bool per_layer_cache_control = false);

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, bool per_layer_cache_control = false);

size_t get_hidden_size(const std::shared_ptr<ov::Model> model);

void set_kv_cache_type_and_shape(std::shared_ptr<ov::Model> model, DeviceConfig& device_config);

}  // namespace utils
}  // namespace genai
}  // namespace ov
