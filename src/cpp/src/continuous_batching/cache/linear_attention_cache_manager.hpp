// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <list>

#include "openvino/runtime/tensor.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"

namespace ov::genai {

/**
 * @brief Manages physical state tensors for linear attention operations
 *        (CausalConv1D, GatedDeltaNet, and similar ops).
 *
 * Discovers all model inputs matching the "<prefix>_state_table.N" naming
 * convention and manages their allocation, zero-initialization, and block
 * copies uniformly.  Each state table prefix (e.g. "conv_state_table",
 * "gated_delta_state_table") forms a separate tensor group, but they all
 * share one logical block pool.
 *
 * State tensors have shape [num_blocks, ...dims...] where dim-0 is the
 * block dimension.  Unlike KV cache, these states are fixed-size and
 * overwritten in-place.
 */
class LinearAttentionCacheManager : public ICacheManager {
    /// Per-layer-index metadata within a single state table group.
    struct LayerInfo {
        ov::element::Type precision;
        ov::PartialShape partial_shape;
        ov::Tensor tensor;              ///< Allocated state tensor (empty until allocate_cache_if_needed).
        std::string full_name;          ///< e.g. "conv_state_table.3"
    };

    /// All layers belonging to one state table prefix (e.g. "conv_state_table").
    struct StateTableGroup {
        std::string prefix;                   ///< e.g. "conv_state_table"
        std::map<size_t, LayerInfo> layers;   ///< layer index -> info
        std::vector<size_t> sorted_indices;   ///< layer indices in ascending order
    };

    std::string m_device;
    std::vector<StateTableGroup> m_groups;
    size_t m_total_num_layers = 0;
    size_t m_num_allocated_blocks = 0;
    size_t m_block_size_in_bytes = 0;
    ov::InferRequest m_request;

    static ov::Shape make_shape(const ov::PartialShape& pshape, size_t num_blocks) {
        ov::PartialShape ps = pshape;
        ps[0] = num_blocks;
        return ps.get_shape();
    }

    /// Bytes per block for a single layer (product of all dims except dim-0).
    static size_t layer_bytes_per_block(const ov::PartialShape& pshape, ov::element::Type precision) {
        size_t elems = 1;
        for (size_t d = 1; d < static_cast<size_t>(pshape.rank().get_length()); ++d) {
            elems *= static_cast<size_t>(pshape[d].get_length());
        }
        return elems * precision.size();
    }

    /// Extract prefix and layer index from a name like "conv_state_table.7".
    /// Returns {"conv_state_table", 7}.  Returns {"", 0} if not a state table name.
    static std::pair<std::string, size_t> parse_state_table_name(const std::string& name) {
        const std::string suffix = "_state_table.";
        auto pos = name.find(suffix);
        if (pos == std::string::npos)
            return {"", 0};
        std::string prefix = name.substr(0, pos + suffix.size() - 1);  // e.g. "conv_state_table"
        size_t layer_idx = std::stoul(name.substr(pos + suffix.size()));
        return {prefix, layer_idx};
    }

public:
    /**
     * @brief Check whether the compiled model has any state table inputs (*_state_table.*).
     */
    static bool has_cache_inputs(const ov::CompiledModel& compiled_model) {
        for (const auto& input : compiled_model.inputs()) {
            for (const auto& name : input.get_names()) {
                if (name.find("_state_table.") != std::string::npos)
                    return true;
            }
        }
        return false;
    }

    explicit LinearAttentionCacheManager(ov::InferRequest request)
        : m_request(std::move(request)) {
        ov::CompiledModel compiled_model = m_request.get_compiled_model();
        std::vector<std::string> execution_devices = compiled_model.get_property(ov::execution_devices);
        OPENVINO_ASSERT(!execution_devices.empty(), "Continuous batching: no execution devices found");
        m_device = execution_devices[0];

        // Discover state table inputs and group by prefix.
        std::map<std::string, StateTableGroup> groups_map;

        for (const auto& input : compiled_model.inputs()) {
            for (const auto& name : input.get_names()) {
                auto [prefix, layer_idx] = parse_state_table_name(name);
                if (prefix.empty())
                    continue;

                auto precision = input.get_element_type();
                ov::PartialShape pshape = input.get_partial_shape();
                OPENVINO_ASSERT(pshape.rank().get_length() >= 2,
                                "State table ", name, " must be at least rank 2, got ",
                                pshape.rank().get_length());

                m_block_size_in_bytes += layer_bytes_per_block(pshape, precision);

                auto& group = groups_map[prefix];
                group.prefix = prefix;
                group.layers[layer_idx] = {precision, pshape, {}, name};
                break;  // only first matching name per input
            }
        }

        // Finalize groups: sort indices and count layers.
        for (auto& [prefix, group] : groups_map) {
            for (const auto& [idx, _] : group.layers) {
                group.sorted_indices.push_back(idx);
            }
            std::sort(group.sorted_indices.begin(), group.sorted_indices.end());
            m_total_num_layers += group.layers.size();
            m_groups.push_back(std::move(group));
        }

        OPENVINO_ASSERT(m_total_num_layers > 0,
                        "LinearAttentionCacheManager: no *_state_table.* inputs found");
    }

    // --- ICacheManager interface ---

    size_t get_num_layers() const override {
        return m_total_num_layers;
    }

    std::string get_device() const override {
        return m_device;
    }

    size_t get_block_size() const override {
        return 1;  // Each block holds the full state for one sequence.
    }

    size_t get_block_size_in_bytes() const override {
        return m_block_size_in_bytes;
    }

    size_t get_num_allocated_blocks() const override {
        return m_num_allocated_blocks;
    }

    void allocate_cache_if_needed(size_t num_blocks) override {
        if (m_num_allocated_blocks >= num_blocks)
            return;
        m_num_allocated_blocks = num_blocks;

        for (auto& group : m_groups) {
            for (size_t layer_idx : group.sorted_indices) {
                auto& info = group.layers.at(layer_idx);
                ov::Shape shape = make_shape(info.partial_shape, num_blocks);
                ov::Tensor new_tensor(info.precision, shape);

                // Zero-initialize: new blocks must start with zero state.
                std::memset(new_tensor.data(), 0, new_tensor.get_byte_size());

                // Preserve existing data when growing.
                if (info.tensor) {
                    std::memcpy(new_tensor.data(), info.tensor.data(), info.tensor.get_byte_size());
                }

                info.tensor = new_tensor;
                m_request.set_tensor(info.full_name, info.tensor);
            }
        }
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override {
        for (const auto& [src_block, dst_blocks] : block_copy_map) {
            for (size_t dst_block : dst_blocks) {
                for (auto& group : m_groups) {
                    for (size_t layer_idx : group.sorted_indices) {
                        auto& info = group.layers.at(layer_idx);
                        size_t byte_stride = layer_bytes_per_block(info.partial_shape, info.precision);

                        const uint8_t* src = static_cast<const uint8_t*>(info.tensor.data())
                                             + src_block * byte_stride;
                        uint8_t* dst = static_cast<uint8_t*>(info.tensor.data())
                                       + dst_block * byte_stride;
                        std::memcpy(dst, src, byte_stride);
                    }
                }
            }
        }
    }

    void clear() override {
        for (auto& group : m_groups) {
            for (auto& [_, info] : group.layers) {
                info.tensor = ov::Tensor();
            }
        }
        m_num_allocated_blocks = 0;
    }
};

}  // namespace ov::genai
