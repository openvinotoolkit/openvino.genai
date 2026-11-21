// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Stateful conversion adapted for the GGML frontend from @mvafin's work:
// https://github.com/mvafin/llama.cpp/tree/ov-frontend-swap

#include "llama_cpp_to_stateful.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/variable.hpp"

namespace ov::genai::gguf_utils {

namespace {

struct ReadChain {
    std::shared_ptr<ov::op::v1::Reshape> relayout;
    std::shared_ptr<ov::op::v1::Transpose> transpose;
};

ReadChain find_read_chain(const std::shared_ptr<ov::Node>& scatter_update) {
    for (const auto& target : scatter_update->output(0).get_target_inputs()) {
        auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(target.get_node()->shared_from_this());
        if (!reshape) {
            continue;
        }

        std::shared_ptr<ov::Node> current = reshape;
        for (int depth = 0; depth < 4 && current; ++depth) {
            std::shared_ptr<ov::Node> next;
            for (const auto& consumer_input : current->output(0).get_target_inputs()) {
                auto consumer = consumer_input.get_node()->shared_from_this();
                if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(consumer)) {
                    return {reshape, transpose};
                }
                if (ov::is_type<ov::op::v8::Slice>(consumer)) {
                    next = consumer;
                }
            }
            current = next;
        }
    }

    return {nullptr, nullptr};
}

void rename_parameter(const std::shared_ptr<ov::Model>& model,
                      const std::string& old_name,
                      const std::string& new_name) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == old_name) {
            parameter->set_friendly_name(new_name);
            parameter->output(0).get_tensor().set_names({new_name});
        }
    }
}

void add_genai_inputs(const std::shared_ptr<ov::Model>& model) {
    rename_parameter(model, "attn_inp_kq_mask", "self_kq_mask");
    rename_parameter(model, "attn_inp_kq_mask_swa", "self_kq_mask_swa");
    rename_parameter(model, "out_ids", "inp_out_ids");

    const auto parameters = model->get_parameters();
    const auto has_beam_idx = std::any_of(parameters.begin(), parameters.end(), [](const auto& parameter) {
        return parameter->get_friendly_name() == "beam_idx";
    });
    if (!has_beam_idx) {
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        beam_idx->set_friendly_name("beam_idx");
        beam_idx->output(0).get_tensor().set_names({"beam_idx"});
        model->add_parameters({beam_idx});
    }
}

void match_attention_mask_precision(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa || sdpa->get_input_size() < 4 || !sdpa->input_value(3).get_node_shared_ptr()) {
            continue;
        }

        auto mask = std::make_shared<ov::op::v1::ConvertLike>(sdpa->input_value(3), sdpa->input_value(0));
        sdpa->input(3).replace_source_output(mask);
    }
}

void specialize_single_sequence_input(const std::shared_ptr<ov::Model>& model) {
    const auto parameters = model->get_parameters();
    for (const auto& parameter : parameters) {
        if (parameter->get_friendly_name() != "n_seq_active") {
            continue;
        }

        // The GGUF bridge reserves a single-sequence graph, so this extra decoder input is fixed.
        auto n_seq_active = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        parameter->output(0).replace(n_seq_active->output(0));
        model->remove_parameter(parameter);
    }
}

}  // namespace

bool LlamaCppToStateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::op::v3::ScatterUpdate>> kv_writes;
    std::vector<ReadChain> read_chains;

    for (const auto& node : model->get_ops()) {
        auto scatter_update = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(node);
        if (!scatter_update ||
            !ov::as_type_ptr<ov::op::v0::Parameter>(scatter_update->input_value(0).get_node_shared_ptr())) {
            continue;
        }

        auto chain = find_read_chain(scatter_update);
        if (!chain.transpose) {
            continue;
        }
        kv_writes.push_back(scatter_update);
        read_chains.push_back(chain);
    }

    OPENVINO_ASSERT(!kv_writes.empty(),
                    "llama.cpp stateful conversion found no KV-cache ScatterUpdate feeding the attention read path");

    std::vector<std::shared_ptr<ov::op::v0::Parameter>> params_to_remove;
    std::vector<std::shared_ptr<ov::op::v0::Result>> results_to_remove;
    ov::SinkVector new_sinks;
    auto beam_idx = std::shared_ptr<ov::op::v0::Parameter>();
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "beam_idx") {
            beam_idx = parameter;
            break;
        }
    }
    const bool created_beam_idx = !beam_idx;
    if (created_beam_idx) {
        beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{ov::Dimension()});
        beam_idx->set_friendly_name("beam_idx");
        beam_idx->output(0).get_tensor().set_names({"beam_idx"});
    }
    const auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    for (size_t i = 0; i < kv_writes.size(); ++i) {
        const auto& scatter_update = kv_writes[i];
        const auto& chain = read_chains[i];
        const auto cache_param =
            ov::as_type_ptr<ov::op::v0::Parameter>(scatter_update->input_value(0).get_node_shared_ptr());
        const auto& cache_name = cache_param->get_friendly_name();
        const auto& cache_shape = cache_param->get_partial_shape();

        constexpr int64_t append_axis = 2;
        OPENVINO_ASSERT(cache_shape.rank().is_static() && cache_shape.size() == 4,
                        "llama.cpp KV cache '",
                        cache_name,
                        "' must have a rank-4 shape");

        ov::PartialShape variable_shape = cache_shape;
        variable_shape[append_axis] = ov::Dimension::dynamic();

        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{variable_shape, cache_param->get_element_type(), cache_name});

        ov::Shape init_shape;
        std::vector<int64_t> split_pattern;
        for (int64_t axis = 0; axis < cache_shape.rank().get_length(); ++axis) {
            if (axis == append_axis) {
                init_shape.push_back(0);
                split_pattern.push_back(-1);
            } else {
                OPENVINO_ASSERT(cache_shape[axis].is_static(),
                                "llama.cpp KV cache '",
                                cache_name,
                                "' must have static non-token dimensions");
                init_shape.push_back(static_cast<size_t>(cache_shape[axis].get_length()));
                split_pattern.push_back(axis < append_axis ? 0 : cache_shape[axis].get_length());
            }
        }
        auto init = ov::op::v0::Constant::create(cache_param->get_element_type(), init_shape, {});
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(init, variable);
        auto split_shape =
            ov::op::v0::Constant::create(ov::element::i64, {split_pattern.size()}, split_pattern);
        auto new_rows = std::make_shared<ov::op::v1::Reshape>(scatter_update->input_value(2), split_shape, true);
        auto past = std::make_shared<ov::op::v8::Gather>(read_value, beam_idx, axis0);
        auto concat =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, new_rows}, append_axis);
        concat->set_friendly_name(scatter_update->get_friendly_name());
        new_sinks.push_back(std::make_shared<ov::op::v6::Assign>(concat, variable));

        for (const auto& target : scatter_update->output(0).get_target_inputs()) {
            if (auto result =
                    ov::as_type_ptr<ov::op::v0::Result>(target.get_node()->shared_from_this())) {
                results_to_remove.push_back(result);
            }
        }
        ov::replace_node(scatter_update, concat);
        chain.transpose->input(0).replace_source_output(chain.relayout->output(0));

        params_to_remove.push_back(cache_param);
        model->add_variables({variable});
    }

    if (created_beam_idx) {
        model->add_parameters({beam_idx});
    }
    std::sort(results_to_remove.begin(), results_to_remove.end());
    results_to_remove.erase(std::unique(results_to_remove.begin(), results_to_remove.end()), results_to_remove.end());
    for (const auto& result : results_to_remove) {
        model->remove_result(result);
    }
    model->add_sinks(new_sinks);
    for (const auto& parameter : params_to_remove) {
        model->remove_parameter(parameter);
    }

    add_genai_inputs(model);
    specialize_single_sequence_input(model);
    match_attention_mask_precision(model);
    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace ov::genai::gguf_utils
