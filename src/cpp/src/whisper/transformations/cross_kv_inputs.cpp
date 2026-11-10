// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cross_kv_inputs.hpp"

#include <algorithm>
#include <regex>
#include <unordered_set>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/read_value_base.hpp"

namespace {

struct CrossState {
    std::string name;
    std::string variable_id;
    size_t layer = 0;
    bool key = true;
};

bool depends_on(std::vector<std::shared_ptr<ov::Node>> stack, const ov::Node* target) {
    std::unordered_set<const ov::Node*> visited;
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (node.get() == target) {
            return true;
        }
        if (!visited.insert(node.get()).second) {
            continue;
        }
        for (const auto& input : node->input_values()) {
            stack.push_back(input.get_node_shared_ptr());
        }
    }
    return false;
}

// The projector is built from the initializer alone, so it may read encoder_hidden_states and constants only.
bool is_encoder_projection(const ov::Output<ov::Node>& initializer, const ov::Node* encoder) {
    bool reads_encoder = false;
    std::unordered_set<const ov::Node*> visited;
    std::vector<std::shared_ptr<ov::Node>> stack{initializer.get_node_shared_ptr()};
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        if (node.get() == encoder) {
            reads_encoder = true;
            continue;
        }
        if (ov::is_type<ov::op::v0::Parameter>(node) || ov::is_type<ov::op::util::ReadValueBase>(node)) {
            return false;
        }
        for (const auto& input : node->input_values()) {
            stack.push_back(input.get_node_shared_ptr());
        }
    }
    return reads_encoder;
}

std::shared_ptr<ov::op::v0::Parameter> find_parameter(const std::shared_ptr<ov::Model>& model,
                                                      const std::string& name) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->output(0).get_names().count(name)) {
            return parameter;
        }
    }
    return nullptr;
}

// Returns nullptr unless exactly one ReadValue reads the variable.
std::shared_ptr<ov::op::util::ReadValueBase> find_read_value(const std::shared_ptr<ov::Model>& model,
                                                             const std::string& variable_id) {
    std::shared_ptr<ov::op::util::ReadValueBase> found;
    for (const auto& op : model->get_ops()) {
        auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(op);
        if (read_value && read_value->get_variable_id() == variable_id) {
            if (found) {
                return nullptr;
            }
            found = read_value;
        }
    }
    return found;
}

std::optional<std::vector<CrossState>> find_cross_states(const std::shared_ptr<ov::Model>& model) {
    const auto encoder = find_parameter(model, "encoder_hidden_states");
    if (!encoder) {
        return std::nullopt;
    }
    const std::regex layer_pattern(R"(layers\.(\d+)\.encoder_attn)");
    std::vector<CrossState> states;
    for (const auto& sink : model->get_sinks()) {
        auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink);
        if (!assign) {
            return std::nullopt;
        }
        const std::string variable_id = assign->get_variable_id();
        auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(assign->input_value(0).get_node_shared_ptr());
        if (!read_value || read_value->get_variable_id() != variable_id) {
            continue;
        }
        if (find_read_value(model, variable_id) != read_value || read_value->get_input_size() != 1 ||
            read_value->get_input_element_type(0) != read_value->get_output_element_type(0) ||
            !is_encoder_projection(read_value->input_value(0), encoder.get())) {
            return std::nullopt;
        }
        const auto consumers = read_value->output(0).get_target_inputs();
        if (consumers.size() != 2) {
            return std::nullopt;
        }
        CrossState state;
        bool has_sdpa_consumer = false;
        for (const auto& consumer : consumers) {
            if (consumer.get_node() == assign.get()) {
                continue;
            }
            std::smatch match;
            const std::string sdpa_name = consumer.get_node()->get_friendly_name();
            if (!ov::is_type<ov::op::v13::ScaledDotProductAttention>(consumer.get_node()) ||
                (consumer.get_index() != 1 && consumer.get_index() != 2) ||
                !std::regex_search(sdpa_name, match, layer_pattern)) {
                return std::nullopt;
            }
            state.layer = std::stoul(match[1]);
            state.key = consumer.get_index() == 1;
            has_sdpa_consumer = true;
        }
        if (!has_sdpa_consumer) {
            return std::nullopt;
        }
        state.name = "cross_kv." + std::to_string(state.layer) + (state.key ? ".key" : ".value");
        state.variable_id = variable_id;
        states.push_back(std::move(state));
    }
    if (states.empty()) {
        return std::nullopt;
    }
    std::unordered_set<std::string> names;
    for (const auto& state : states) {
        if (!names.insert(state.name).second) {
            return std::nullopt;
        }
    }
    std::sort(states.begin(), states.end(), [](const CrossState& a, const CrossState& b) {
        return a.layer != b.layer ? a.layer < b.layer : a.key > b.key;
    });
    return states;
}

}  // namespace

namespace ov::genai {

std::optional<WhisperCrossKVModels> split_whisper_cross_kv(const std::shared_ptr<ov::Model>& decoder) {
    const auto states = find_cross_states(decoder);
    if (!states.has_value()) {
        return std::nullopt;
    }

    WhisperCrossKVModels result;
    {
        auto clone = decoder->clone();
        ov::ResultVector results;
        for (const auto& state : *states) {
            auto read_value = find_read_value(clone, state.variable_id);
            OPENVINO_ASSERT(read_value,
                            "Whisper cross-attention state ",
                            state.variable_id,
                            " is missing in the clone.");
            ov::Output<ov::Node> projection = read_value->input_value(0);
            projection.get_tensor().set_names({state.name});
            auto output = std::make_shared<ov::op::v0::Result>(projection);
            output->set_friendly_name(state.name);
            results.push_back(output);
        }
        result.projector =
            std::make_shared<ov::Model>(results,
                                        ov::ParameterVector{find_parameter(clone, "encoder_hidden_states")},
                                        "whisper_cross_kv_projector");
    }

    result.decoder = decoder->clone();
    ov::ParameterVector cross_kv_inputs;
    for (const auto& state : *states) {
        auto read_value = find_read_value(result.decoder, state.variable_id);
        std::shared_ptr<ov::op::Sink> assign;
        for (const auto& sink : result.decoder->get_sinks()) {
            auto candidate = ov::as_type_ptr<ov::op::util::AssignBase>(sink);
            if (candidate && candidate->get_variable_id() == state.variable_id) {
                assign = sink;
            }
        }
        OPENVINO_ASSERT(read_value && assign, "Whisper cross-attention state ", state.variable_id, " is missing.");

        auto input = std::make_shared<ov::op::v0::Parameter>(read_value->get_output_element_type(0),
                                                             read_value->get_output_partial_shape(0));
        input->set_friendly_name(state.name);
        input->output(0).get_tensor().set_names({state.name});
        for (auto consumer : read_value->output(0).get_target_inputs()) {
            if (consumer.get_node() != assign.get()) {
                consumer.replace_source_output(input->output(0));
            }
        }
        result.decoder->remove_sink(assign);
        result.decoder->remove_variable(read_value->get_variable());
        cross_kv_inputs.push_back(input);
        result.names.push_back(state.name);
    }
    result.decoder->add_parameters(cross_kv_inputs);
    const auto encoder = find_parameter(result.decoder, "encoder_hidden_states");
    std::vector<std::shared_ptr<ov::Node>> outputs(result.decoder->get_results().begin(),
                                                   result.decoder->get_results().end());
    outputs.insert(outputs.end(), result.decoder->get_sinks().begin(), result.decoder->get_sinks().end());
    if (depends_on(std::move(outputs), encoder.get())) {
        return std::nullopt;
    }
    result.decoder->remove_parameter(encoder);
    result.decoder->validate_nodes_and_infer_types();
    return result;
}

}  // namespace ov::genai
