// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "content_hash.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"

namespace ov::genai {

namespace model_identity_detail {

/**
 * @brief Feeds every attribute an op visits into a ContentHash.
 *
 * Node/tensor names are never used (deliberately not visited here): OpenVINO auto-generates them from a
 * process-wide counter when not set explicitly, so they are not reproducible across processes. Attribute
 * types without one of AttributeVisitor's typed overloads - sub-graphs (If/Loop bodies) and shared_ptr<Node>
 * references - are skipped; this covers the overwhelming majority of real op attributes (ints, floats,
 * bools, strings and their vectors, plus anything backed by one of those, such as element::Type or a
 * Shape), which is enough to tell apart ops of the same type with different attributes.
 */
class GraphAttributeHasher final : public ov::AttributeVisitor {
public:
    explicit GraphAttributeHasher(ContentHash& state) : m_state(state) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>&) override {
        mix_name(name);
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
    void mix_name(const std::string& name) {
        mix_bytes(name.data(), name.size());
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
};

}  // namespace model_identity_detail

/**
 * @brief Digest of a model's weights and graph structure, as 32 hex characters.
 *
 * Block hashes of the prefix cache depend only on tokens, so a store that outlives the pipeline needs
 * something that tells two models apart. The digest covers every constant's element type, shape and bytes
 * (in graph order), plus every operator's type, attributes and connectivity (by position in topological
 * order, never by name - see GraphAttributeHasher), so the same weights and graph give the same value
 * across processes and builds, and two graphs that happen to share constants but differ in operators,
 * attributes or wiring do not collide.
 */
inline std::string compute_model_identity(const std::shared_ptr<const ov::Model>& model) {
    ContentHash state{0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL};
    const auto mix = [&state](const void* data, size_t size) {
        // Seeding each step with the previous digest chains the pieces in order.
        state = compute_content_hash(data, size, state.lo ^ (state.hi << 1));
    };

    const auto ordered_ops = model->get_ordered_ops();
    std::unordered_map<const ov::Node*, size_t> ordinal_of;
    ordinal_of.reserve(ordered_ops.size());
    for (size_t i = 0; i < ordered_ops.size(); ++i) {
        ordinal_of.emplace(ordered_ops[i].get(), i);
    }

    model_identity_detail::GraphAttributeHasher attribute_hasher(state);
    for (const auto& op : ordered_ops) {
        const auto constant = std::dynamic_pointer_cast<const ov::op::v0::Constant>(op);
        if (constant) {
            std::ostringstream description;
            description << constant->get_element_type() << ' ' << constant->get_shape();
            const std::string text = description.str();
            mix(text.data(), text.size());
            if (constant->get_element_type() != ov::element::string) {
                mix(constant->get_data_ptr(), constant->get_byte_size());
            }
            // Constant has no other attributes worth visiting; its identity is the bytes above.
            continue;
        }

        const char* type_name = op->get_type_info().name;
        const char* version_id = op->get_type_info().version_id;
        mix(type_name, std::strlen(type_name));
        if (version_id != nullptr) {
            mix(version_id, std::strlen(version_id));
        }

        // Connectivity: which op (by position in topological order, never by name) feeds which input.
        for (const auto& input : op->inputs()) {
            const ov::Output<ov::Node> source = input.get_source_output();
            const size_t source_ordinal = ordinal_of.at(source.get_node());
            const size_t source_port = source.get_index();
            mix(&source_ordinal, sizeof(source_ordinal));
            mix(&source_port, sizeof(source_port));
        }

        for (const auto& output : op->outputs()) {
            std::ostringstream description;
            description << output.get_element_type() << ' ' << output.get_partial_shape();
            const std::string text = description.str();
            mix(text.data(), text.size());
        }

        const_cast<ov::Node&>(*op).visit_attributes(attribute_hasher);
    }

    char hex[33];
    std::snprintf(hex, sizeof(hex), "%016llx%016llx", static_cast<unsigned long long>(state.hi),
                  static_cast<unsigned long long>(state.lo));
    return hex;
}

}  // namespace ov::genai

