// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "content_hash.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/version.hpp"
#include "openvino/genai/version.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

namespace model_identity_detail {

/// Bytes of a constant that are hashed as one piece; larger constants are split so threads can share the work.
constexpr size_t SHARD_BYTES = 16ull * 1024 * 1024;

/// Digest of @p size bytes: the digests of their shards of SHARD_BYTES, combined in order. It does not depend on
/// how many threads computed the shards.
inline ContentHash combine_shard_digests(const std::vector<ContentHash>& shards, size_t size) {
    return compute_content_hash(shards.data(), shards.size() * sizeof(ContentHash), size);
}

inline ContentHash digest_of_bytes(const void* data, size_t size) {
    const size_t num_shards = std::max<size_t>(1, (size + SHARD_BYTES - 1) / SHARD_BYTES);
    std::vector<ContentHash> shards(num_shards);
    for (size_t i = 0; i < num_shards; ++i) {
        const size_t begin = i * SHARD_BYTES;
        shards[i] = compute_content_hash(static_cast<const uint8_t*>(data) + begin, std::min(SHARD_BYTES, size - begin));
    }
    return combine_shard_digests(shards, size);
}

/**
 * @brief The digests of all numeric constants of a model, computed on all cores before the graph is walked.
 *
 * A constant that several operators share (tied weights) is hashed once. A constant this class did not see, for
 * example one inside a sub-graph it cannot reach, is hashed on demand with the same result.
 */
class ConstantDigests final {
public:
    explicit ConstantDigests(const ov::Model& model) {
        std::vector<std::shared_ptr<const ov::op::v0::Constant>> constants;
        collect(model, constants);

        struct Shard {
            size_t constant;
            size_t index;
        };
        std::vector<Shard> shards;
        std::vector<std::vector<ContentHash>> digests(constants.size());
        for (size_t i = 0; i < constants.size(); ++i) {
            const size_t size = constants[i]->get_byte_size();
            const size_t num_shards = std::max<size_t>(1, (size + SHARD_BYTES - 1) / SHARD_BYTES);
            digests[i].resize(num_shards);
            for (size_t shard = 0; shard < num_shards; ++shard) {
                shards.push_back({i, shard});
            }
        }

        std::atomic<size_t> next{0};
        const auto work = [&] {
            for (size_t task = next.fetch_add(1); task < shards.size(); task = next.fetch_add(1)) {
                const Shard& shard = shards[task];
                const size_t size = constants[shard.constant]->get_byte_size();
                const size_t begin = shard.index * SHARD_BYTES;
                digests[shard.constant][shard.index] = compute_content_hash(
                    static_cast<const uint8_t*>(constants[shard.constant]->get_data_ptr()) + begin,
                    std::min(SHARD_BYTES, size - begin));
            }
        };
        const size_t num_threads = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), shards.size()));
        std::vector<std::thread> threads;
        for (size_t i = 1; i < num_threads; ++i) {
            threads.emplace_back(work);
        }
        work();
        for (std::thread& thread : threads) {
            thread.join();
        }

        std::string described;
        for (size_t i = 0; i < constants.size(); ++i) {
            const ContentHash digest = combine_shard_digests(digests[i], constants[i]->get_byte_size());
            m_digests.emplace(key_of(*constants[i]), digest);
            std::ostringstream description;
            description << constants[i]->get_element_type() << ' ' << constants[i]->get_shape() << ';';
            described += description.str();
            described.append(reinterpret_cast<const char*>(&digest), sizeof(digest));
        }
        m_combined = compute_content_hash(described.data(), described.size());
    }

    /// One digest of every constant, in graph order, with its element type and shape.
    const ContentHash& combined() const {
        return m_combined;
    }

    ContentHash digest_of(const ov::op::v0::Constant& constant) const {
        const auto found = m_digests.find(key_of(constant));
        return found != m_digests.end() ? found->second : digest_of_bytes(constant.get_data_ptr(), constant.get_byte_size());
    }

private:
    using Key = std::pair<const void*, size_t>;

    static Key key_of(const ov::op::v0::Constant& constant) {
        return {constant.get_data_ptr(), constant.get_byte_size()};
    }

    static void collect(const ov::Model& model, std::vector<std::shared_ptr<const ov::op::v0::Constant>>& constants) {
        std::set<Key> seen;
        for (const auto& known : constants) {
            seen.insert(key_of(*known));
        }
        for (const auto& op : model.get_ordered_ops()) {
            if (const auto constant = std::dynamic_pointer_cast<const ov::op::v0::Constant>(op)) {
                if (constant->get_element_type() != ov::element::string && seen.insert(key_of(*constant)).second) {
                    constants.push_back(constant);
                }
            } else if (const auto sub_graph_op = std::dynamic_pointer_cast<const ov::op::util::MultiSubGraphOp>(op)) {
                for (size_t i = 0; i < sub_graph_op->get_internal_subgraphs_size(); ++i) {
                    collect(*sub_graph_op->get_function(static_cast<int>(i)), constants);
                }
            }
        }
    }

    std::map<Key, ContentHash> m_digests;
    ContentHash m_combined;
};

/// Constants up to this size are covered by their bytes in the structure digest; of a larger one only its type, shape
/// and the first and last 4 KiB, since its full bytes belong to the weights digest.
constexpr size_t STRUCTURE_FULL_CONSTANT_BYTES = 64ull * 1024;
constexpr size_t STRUCTURE_EDGE_BYTES = 4ull * 1024;

inline ContentHash structure_digest_of(const ov::op::v0::Constant& constant) {
    const size_t size = constant.get_byte_size();
    const auto* data = static_cast<const uint8_t*>(constant.get_data_ptr());
    if (size <= STRUCTURE_FULL_CONSTANT_BYTES) {
        return digest_of_bytes(data, size);
    }
    const ContentHash edges[] = {compute_content_hash(data, STRUCTURE_EDGE_BYTES),
                                 compute_content_hash(data + size - STRUCTURE_EDGE_BYTES, STRUCTURE_EDGE_BYTES)};
    return compute_content_hash(edges, sizeof(edges), size);
}

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
    /// @param constant_digest What a numeric constant contributes besides its type and shape.
    ModelHasher(ContentHash& state, std::function<ContentHash(const ov::op::v0::Constant&)> constant_digest)
        : m_state(state),
          m_constant_digest(std::move(constant_digest)) {}

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
                if (constant->get_element_type() == ov::element::string) {
                    for (const std::string& item : constant->get_vector<std::string>()) {
                        const size_t size = item.size();
                        mix_bytes(&size, sizeof(size));
                        mix_bytes(item.data(), size);
                    }
                } else {
                    const ContentHash digest = m_constant_digest(*constant);
                    mix_bytes(&digest, sizeof(digest));
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
    std::function<ContentHash(const ov::op::v0::Constant&)> m_constant_digest;
    std::string m_current_op_type;
};

}  // namespace model_identity_detail

namespace model_identity_detail {

inline std::string to_hex(const ContentHash& hash) {
    char hex[33];
    std::snprintf(hex, sizeof(hex), "%016llx%016llx", static_cast<unsigned long long>(hash.hi),
                  static_cast<unsigned long long>(hash.lo));
    return hex;
}

/// Bumped when the way the digests are computed changes, so identities of an older format are never matched.
constexpr uint32_t IDENTITY_FORMAT_VERSION = 2;

}  // namespace model_identity_detail

/**
 * @brief Digest of a model's graph structure, as 32 hex characters, cheap enough to compute before the model is compiled.
 *
 * Covers every operator's type, attributes and connectivity (by position in topological order, never by name - see
 * ModelHasher) and the element type and shape of every constant. A constant of up to 64 KiB (scales, RoPE tables) is
 * covered by its bytes, a larger one only by its first and last 4 KiB: its full bytes belong to the weights digest.
 * @throws ov::Exception if an operator has an attribute of a type the digest cannot cover.
 */
inline std::string compute_structure_digest(const std::shared_ptr<const ov::Model>& model) {
    ContentHash state{0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL};
    model_identity_detail::ModelHasher hasher(state, &model_identity_detail::structure_digest_of);
    hasher.hash_model(*model);
    return model_identity_detail::to_hex(state);
}

/// Digest of the bytes of every numeric constant (tied weights hashed once), computed on all cores. Slow for a large model.
inline std::string compute_weights_digest(const std::shared_ptr<const ov::Model>& model) {
    return model_identity_detail::to_hex(model_identity_detail::ConstantDigests(*model).combined());
}

/// The identity of a model from its two digests; see compute_model_identity().
inline std::string combine_model_identity(const std::string& structure_digest, const std::string& weights_digest) {
    const std::string text = structure_digest + '|' + weights_digest + '|' +
                             std::to_string(model_identity_detail::IDENTITY_FORMAT_VERSION);
    return model_identity_detail::to_hex(compute_content_hash(text.data(), text.size()));
}

/**
 * @brief Digest of a model's weights and graph structure, as 32 hex characters.
 *
 * Block hashes of the prefix cache depend only on tokens, so a store that outlives the pipeline needs
 * something that tells two models apart. The digest covers the structure digest and the weights digest, so the same
 * weights and graph give the same value across processes and builds, and two graphs that happen to share constants
 * but differ in operators, attributes, wiring or sub-graph bodies do not collide.
 * @throws ov::Exception if an operator has an attribute of a type the digest cannot cover.
 */
inline std::string compute_model_identity(const std::shared_ptr<const ov::Model>& model) {
    return combine_model_identity(compute_structure_digest(model), compute_weights_digest(model));
}

/**
 * @brief What the compiled model computes with, which its weights do not tell: precisions, quantization settings, the
 * hardware generation and the versions of the runtime.
 *
 * The same weights give other numbers under another inference precision or on another device generation, so blocks
 * stored under one of these must not be reused under another. A property the device does not have is left out.
 */
inline std::string compute_numerics_fingerprint(const ov::CompiledModel& compiled_model, ov::Core& core) {
    std::ostringstream fingerprint;
    fingerprint << ov::get_openvino_version().buildNumber << "|genai=" << ov::genai::get_version().buildNumber;
    for (const char* name : {"INFERENCE_PRECISION_HINT",
                             "KV_CACHE_PRECISION",
                             "KEY_CACHE_PRECISION",
                             "VALUE_CACHE_PRECISION",
                             "KEY_CACHE_GROUP_SIZE",
                             "VALUE_CACHE_GROUP_SIZE",
                             "DYNAMIC_QUANTIZATION_GROUP_SIZE",
                             "ACTIVATIONS_SCALE_FACTOR"}) {
        try {
            fingerprint << '|' << name << '=' << compiled_model.get_property(name).as<std::string>();
        } catch (const ov::Exception&) {
            // The device has no such property.
        }
    }
    try {
        fingerprint << "|architecture=" << core.get_property(compiled_model.get_property(ov::execution_devices).at(0), ov::device::architecture);
    } catch (const ov::Exception&) {
        // The device does not report its architecture.
    }
    return fingerprint.str();
}

}  // namespace ov::genai
