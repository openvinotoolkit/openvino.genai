// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/kv_cache.hpp"

#include <iostream>
#include <regex>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

namespace {

/**
 * @brief Extract layer index from cache_prefix for NPUW-compatible Variable naming.
 * 
 * The StatefulToStateless pass in OpenVINO expects Variable names in a specific format:
 *   past_key_values.<layer_idx>.key + present.<layer_idx>.key
 * 
 * This function extracts the layer index from prefixes like:
 *   "model.layers[15].self_attn" -> 15
 *   "layers[0].attention" -> 0
 * 
 * @param cache_prefix The module path prefix (e.g., "model.layers[15].self_attn")
 * @return The extracted layer index, or -1 if not found
 */
int extract_layer_index(const std::string& cache_prefix) {
    // Match patterns like "layers[15]" or "layers.15"
    static const std::regex layer_pattern(R"(layers[\[\.](\d+)[\]\.]?)");
    std::smatch match;
    if (std::regex_search(cache_prefix, match, layer_pattern)) {
        return std::stoi(match[1].str());
    }
    return -1;  // Not found - will use original naming
}

/**
 * @brief Generate NPUW-compatible Variable names for KV cache.
 * 
 * NPUW's StatefulToStateless pass expects names like:
 *   past_key_values.N.keypresent.N.key  (for keys)
 *   past_key_values.N.valuepresent.N.value  (for values)
 * 
 * @param layer_idx The layer index (0-based)
 * @param is_key True for key cache, false for value cache
 * @return The Variable name compatible with StatefulToStateless
 */
std::string make_npuw_variable_name(int layer_idx, bool is_key) {
    const std::string type = is_key ? "key" : "value";
    return "past_key_values." + std::to_string(layer_idx) + "." + type +
           "present." + std::to_string(layer_idx) + "." + type;
}

}  // namespace

std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                          const Tensor& values,
                                          const Tensor& beam_idx,
                                          int32_t num_kv_heads,
                                          int32_t head_dim,
                                          const std::string& cache_prefix,
                                          const BuilderContext& ctx) {
    auto* op_ctx = keys.context();
    auto batch = shape::dim(keys, 0);
    auto kv_heads = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_kv_heads)});
    auto zero_len = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto head_dim_vec = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_dim)});
    auto cache_shape = shape::make({batch, kv_heads, zero_len, head_dim_vec});

    auto zero = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(keys.dtype());
    auto k_init = shape::broadcast_to(zero, cache_shape);
    auto v_init = shape::broadcast_to(zero, cache_shape);

    // Generate NPUW-compatible Variable names for KV cache
    // StatefulToStateless pass expects: past_key_values.N.keypresent.N.key format
    std::string k_name, v_name;
    int layer_idx = extract_layer_index(cache_prefix);
    if (layer_idx >= 0) {
        // Use NPUW-compatible naming for layers with index
        k_name = make_npuw_variable_name(layer_idx, true);
        v_name = make_npuw_variable_name(layer_idx, false);
    } else {
        // Fallback to original naming for non-indexed caches
        k_name = cache_prefix + ".key_cache";
        v_name = cache_prefix + ".value_cache";
    }

    // Create Variable with dynamic shapes
    ov::PartialShape var_shape{-1, num_kv_heads, -1, head_dim};
    ov::op::util::VariableInfo k_info{var_shape, keys.dtype(), k_name};
    auto k_var = std::make_shared<ov::op::util::Variable>(k_info);
    auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_init.output(), k_var);

    ov::op::util::VariableInfo v_info{var_shape, values.dtype(), v_name};
    auto v_var = std::make_shared<ov::op::util::Variable>(v_info);
    auto v_read = std::make_shared<ov::op::v6::ReadValue>(v_init.output(), v_var);

    auto k_cached = ops::gather(Tensor(k_read->output(0), op_ctx), beam_idx, 0);
    auto v_cached = ops::gather(Tensor(v_read->output(0), op_ctx), beam_idx, 0);

    auto k_combined = ops::concat({k_cached, keys}, 2);
    auto v_combined = ops::concat({v_cached, values}, 2);

    auto k_assign = std::make_shared<ov::opset13::Assign>(k_combined.output(), k_var);
    auto v_assign = std::make_shared<ov::opset13::Assign>(v_combined.output(), v_var);
    ctx.register_sink(k_assign);
    ctx.register_sink(v_assign);

    return {k_combined, v_combined};
}

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
