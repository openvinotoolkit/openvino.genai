// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "content_hash.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/variable.hpp"

namespace ov::genai {

namespace model_identity_detail {

/**
 * @brief Feeds a graph into a ContentHash: every constant, and every operator's type, attributes and connectivity.
 *
 * Node/tensor names are never used (deliberately not visited here): OpenVINO auto-generates them from a
 * process-wide counter when not set explicitly, so they are not reproducible across processes. The bodies of
 * sub-graph operations (Loop, If, TensorIterator) are hashed recursively together with the descriptions that connect
 * them to the outer graph. An attribute of a type this class cannot hash is rejected instead of skipped, since
 * skipping it would let two different models share one identity.
 */
class ModelHasher final : public ov::AttributeVisitor {
public:
    explicit ModelHasher(ContentHash& state) : m_state(state) {}

    void hash_model(const ov::Model& model) {
        const auto ordered_ops = model.get_ordered_ops();
        std::unordered_map<const ov::Node*, size_t> ordinal_of;
        ordinal_of.reserve(ordered_ops.size());
        for (size_t i = 0; i < ordered_ops.size(); ++i) {
            ordinal_of.emplace(ordered_ops[i].get(), i);
        }

        for (const auto& op : ordered_ops) {
            const auto constant = std::dynamic_pointer_cast<const ov::op::v0::Constant>(op);
            if (constant) {
                std::ostringstream description;
                description << constant->get_element_type() << ' ' << constant->get_shape();
                const std::string text = description.str();
                mix_bytes(text.data(), text.size());
                if (constant->get_element_type() != ov::element::string) {
                    mix_bytes(constant->get_data_ptr(), constant->get_byte_size());
                }
                // Constant has no other attributes worth visiting; its identity is the bytes above.
                continue;
            }

            const char* type_name = op->get_type_info().name;
            const char* version_id = op->get_type_info().version_id;
            mix_bytes(type_name, std::strlen(type_name));
            if (version_id != nullptr) {
                mix_bytes(version_id, std::strlen(version_id));
            }

            // Connectivity: which op (by position in topological order, never by name) feeds which input.
            for (const auto& input : op->inputs()) {
                const ov::Output<ov::Node> source = input.get_source_output();
                const size_t source_ordinal = ordinal_of.at(source.get_node());
                const size_t source_port = source.get_index();
                mix_bytes(&source_ordinal, sizeof(source_ordinal));
                mix_bytes(&source_port, sizeof(source_port));
            }

            for (const auto& output : op->outputs()) {
                std::ostringstream description;
                description << output.get_element_type() << ' ' << output.get_partial_shape();
                const std::string text = description.str();
                mix_bytes(text.data(), text.size());
            }

            m_current_op_type = type_name;
            const_cast<ov::Node&>(*op).visit_attributes(*this);
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        mix_name(name);
        if (auto* body = dynamic_cast<ov::AttributeAdapter<std::shared_ptr<ov::Model>>*>(&adapter)) {
            const std::shared_ptr<ov::Model> model = body->get();
            OPENVINO_ASSERT(model != nullptr, "Operation '", m_current_op_type, "' has an empty body");
            const std::string current_op_type = m_current_op_type;
            hash_model(*model);
            m_current_op_type = current_op_type;
        } else if (auto* inputs = dynamic_cast<ov::AttributeAdapter<ov::op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector>*>(&adapter)) {
            for (const auto& description : inputs->get()) {
                mix_input_description(*description);
            }
        } else if (auto* outputs = dynamic_cast<ov::AttributeAdapter<ov::op::util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector>*>(&adapter)) {
            for (const auto& description : outputs->get()) {
                mix_output_description(*description);
            }
        } else if (auto* ports = dynamic_cast<ov::AttributeAdapter<ov::op::v5::Loop::SpecialBodyPorts>*>(&adapter)) {
            const auto value = ports->get();
            mix_bytes(&value.current_iteration_input_idx, sizeof(value.current_iteration_input_idx));
            mix_bytes(&value.body_condition_output_idx, sizeof(value.body_condition_output_idx));
        } else if (auto* variable = dynamic_cast<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>*>(&adapter)) {
            const auto info = variable->get()->get_info();
            std::ostringstream description;
            description << info.variable_id << ' ' << info.data_type << ' ' << info.data_shape;
            const std::string text = description.str();
            mix_bytes(text.data(), text.size());
        } else if (auto* values = dynamic_cast<ov::AttributeAdapter<std::set<std::string>>*>(&adapter)) {
            for (const auto& value : values->get()) {
                mix_bytes(value.data(), value.size());
            }
        } else if (auto* partial_shape = dynamic_cast<ov::AttributeAdapter<ov::PartialShape>*>(&adapter)) {
            mix_text(partial_shape->get());
        } else if (auto* dimension = dynamic_cast<ov::AttributeAdapter<ov::Dimension>*>(&adapter)) {
            mix_text(dimension->get());
        } else {
            OPENVINO_THROW("The model identity cannot cover attribute '", name, "' of operation '", m_current_op_type,
                           "'; its type is not supported, so two models differing only in it would share cached blocks");
        }
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        mix_name(name);
        mix_bytes(adapter.get().data(), adapter.get().size());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int8_t>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int16_t>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int32_t>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<uint8_t>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<uint16_t>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<uint32_t>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<uint64_t>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<float>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        mix_value(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int8_t>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int16_t>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint8_t>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint16_t>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint32_t>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<double>>& adapter) override {
        mix_vector(name, adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        mix_name(name);
        for (const auto& item : adapter.get()) {
            mix_bytes(item.data(), item.size());
        }
    }
    void on_adapter(const std::string& name, ov::VisitorAdapter& adapter) override {
        mix_name(name);
        adapter.visit_attributes(*this);
    }

private:
    void mix_input_description(const ov::op::util::MultiSubGraphOp::InputDescription& description) {
        const char* type_name = description.get_type_info().name;
        mix_bytes(type_name, std::strlen(type_name));
        mix_bytes(&description.m_input_index, sizeof(description.m_input_index));
        mix_bytes(&description.m_body_parameter_index, sizeof(description.m_body_parameter_index));
        if (const auto* slice = dynamic_cast<const ov::op::util::MultiSubGraphOp::SliceInputDescription*>(&description)) {
            const int64_t values[] = {slice->m_start, slice->m_stride, slice->m_part_size, slice->m_end, slice->m_axis};
            mix_bytes(values, sizeof(values));
        } else if (const auto* merged = dynamic_cast<const ov::op::util::MultiSubGraphOp::MergedInputDescription*>(&description)) {
            mix_bytes(&merged->m_body_value_index, sizeof(merged->m_body_value_index));
        } else {
            OPENVINO_ASSERT(dynamic_cast<const ov::op::util::MultiSubGraphOp::InvariantInputDescription*>(&description) != nullptr,
                            "The model identity cannot cover input description '", type_name, "'");
        }
    }
    void mix_output_description(const ov::op::util::MultiSubGraphOp::OutputDescription& description) {
        const char* type_name = description.get_type_info().name;
        mix_bytes(type_name, std::strlen(type_name));
        mix_bytes(&description.m_body_value_index, sizeof(description.m_body_value_index));
        mix_bytes(&description.m_output_index, sizeof(description.m_output_index));
        if (const auto* concat = dynamic_cast<const ov::op::util::MultiSubGraphOp::ConcatOutputDescription*>(&description)) {
            const int64_t values[] = {concat->m_start, concat->m_stride, concat->m_part_size, concat->m_end, concat->m_axis};
            mix_bytes(values, sizeof(values));
        } else if (const auto* body = dynamic_cast<const ov::op::util::MultiSubGraphOp::BodyOutputDescription*>(&description)) {
            mix_bytes(&body->m_iteration, sizeof(body->m_iteration));
        } else {
            OPENVINO_THROW("The model identity cannot cover output description '", type_name, "'");
        }
    }
    void mix_name(const std::string& name) {
        mix_bytes(name.data(), name.size());
    }
    template <typename T>
    void mix_text(const T& streamable) {
        std::ostringstream description;
        description << streamable;
        const std::string text = description.str();
        mix_bytes(text.data(), text.size());
    }
    template <typename T>
    void mix_value(const std::string& name, const T& value) {
        mix_name(name);
        mix_bytes(&value, sizeof(value));
    }
    template <typename T>
    void mix_vector(const std::string& name, const std::vector<T>& values) {
        mix_name(name);
        if (!values.empty()) {
            mix_bytes(values.data(), values.size() * sizeof(T));
        }
    }
    void mix_bytes(const void* data, size_t size) {
        m_state = compute_content_hash(data, size, m_state.lo ^ (m_state.hi << 1));
    }

    ContentHash& m_state;
    std::string m_current_op_type;
};

}  // namespace model_identity_detail

/**
 * @brief Digest of a model's weights and graph structure, as 32 hex characters.
 *
 * Block hashes of the prefix cache depend only on tokens, so a store that outlives the pipeline needs
 * something that tells two models apart. The digest covers every constant's element type, shape and bytes
 * (in graph order), plus every operator's type, attributes and connectivity (by position in topological
 * order, never by name - see ModelHasher), so the same weights and graph give the same value
 * across processes and builds, and two graphs that happen to share constants but differ in operators,
 * attributes, wiring or sub-graph bodies do not collide.
 * @throws ov::Exception if an operator has an attribute of a type the digest cannot cover.
 */
inline std::string compute_model_identity(const std::shared_ptr<const ov::Model>& model) {
    ContentHash state{0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL};
    model_identity_detail::ModelHasher hasher(state);
    hasher.hash_model(*model);

    char hex[33];
    std::snprintf(hex, sizeof(hex), "%016llx%016llx", static_cast<unsigned long long>(state.hi),
                  static_cast<unsigned long long>(state.lo));
    return hex;
}

}  // namespace ov::genai

