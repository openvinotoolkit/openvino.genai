// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/paged_attention_transformations.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

namespace ov {
namespace genai {
namespace utils {

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, bool per_layer_cache_control, bool allow_cache_rotation, bool allow_xattention) {
    const ov::op::util::VariableVector& variables = model->get_variables();
    OPENVINO_ASSERT(!variables.empty(), "Model is supposed to be stateful");

    bool use_block_indices_inputs = per_layer_cache_control;
    bool use_score_outputs = per_layer_cache_control;
    ov::pass::SDPAToPagedAttention(use_block_indices_inputs, use_score_outputs, /* allow_score_aggregation = */ true, allow_cache_rotation, allow_xattention).run_on_model(model);

    std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>> key_cache_params, value_cache_params;
    for (const auto& param_ptr : model->get_parameters()) {
        const auto& name = param_ptr->get_friendly_name();
        if (name.find("key_cache.") == 0) {
            key_cache_params[name] = param_ptr;
        } else if (name.find("value_cache.") == 0) {
            value_cache_params[name] = param_ptr;
        }
    }

    OPENVINO_ASSERT(key_cache_params.size() == value_cache_params.size() && key_cache_params.size() > 0);

    size_t num_decoder_layers = key_cache_params.size();
    for (size_t idx = 0; idx < num_decoder_layers; idx++) {
        auto k = key_cache_params[std::string("key_cache.") + std::to_string(idx)];
        auto key_shape = k->get_partial_shape();
        size_t num_k_heads = key_shape[1].get_length();
        size_t k_head_size = key_shape[2].get_length();

        auto v = value_cache_params[std::string("value_cache.") + std::to_string(idx)];
        auto value_shape = v->get_partial_shape();
        size_t num_v_heads = value_shape[1].get_length();
        size_t v_head_size = value_shape[2].get_length();

        // reset information in KV cache parameters and set PagedAttention's rt_info
        // allow a plugin to automatically set KV cache precisions
        k->set_element_type(ov::element::dynamic);
        v->set_element_type(ov::element::dynamic);

        // order of dimensions within shapes are not required for plugin during compilation
        k->set_partial_shape(ov::PartialShape::dynamic(4));
        v->set_partial_shape(ov::PartialShape::dynamic(4));

        // set KV cache parameters as rt_info for PagedAttention op, so plugins can apply
        // model compile-time optimizations based on them
        auto pa_op = k->get_output_target_inputs(0).begin()->get_node();
        pa_op->get_rt_info()["num_k_heads"] = num_k_heads;
        pa_op->get_rt_info()["k_head_size"] = k_head_size;
        pa_op->get_rt_info()["num_v_heads"] = num_v_heads;
        pa_op->get_rt_info()["v_head_size"] = v_head_size;
    }

    model->validate_nodes_and_infer_types();
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
