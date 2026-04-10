// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>
#include <list>

#include "openvino/runtime/tensor.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"

namespace ov::genai {

/**
 * @brief Manages physical causal convolution state tensors on a device.
 *
 * Each decoder layer that uses causal conv1d has a single state tensor named
 * "conv_state_table.N" with shape [num_blocks, channels, kernel_size - 1].
 * Unlike KV cache, the conv state is fixed-size and does not grow with new tokens —
 * it is overwritten in-place.
 */
class CausalConv1DCacheManager : public ICacheManager {
    size_t m_num_conv_layers = 0;
    std::string m_device;
    std::map<size_t, ov::element::Type> m_conv_precisions;    ///< layer index -> precision
    std::map<size_t, ov::PartialShape> m_conv_shapes;         ///< layer index -> partial shape
    std::map<size_t, ov::Tensor> m_conv_state;                ///< layer index -> allocated tensor
    size_t m_num_allocated_blocks = 0;
    size_t m_block_size_in_bytes = 0;
    size_t m_cache_interval = 0;  ///< kernel_size - 1 (size of the conv state per sequence per layer)
    ov::InferRequest m_request;
    std::vector<size_t> m_layer_indices;  ///< sorted layer indices for iteration

    static ov::Shape set_num_blocks(ov::PartialShape pshape, size_t num_blocks) {
        pshape[0] = num_blocks;
        return pshape.get_shape();
    }

    void update_request_tensor(size_t layer_idx) {
        m_request.set_tensor(std::string("conv_state_table.") + std::to_string(layer_idx), m_conv_state.at(layer_idx));
    }

public:
    /**
     * @brief Check whether the compiled model has conv state inputs (conv_state_table.*).
     * @param compiled_model The compiled model to inspect.
     * @return true if at least one conv_state_table input is found.
     */
    static bool has_cache_inputs(const ov::CompiledModel& compiled_model) {
        for (const auto& input : compiled_model.inputs()) {
            for (const auto& name : input.get_names()) {
                if (name.find("conv_state_table.") == 0)
                    return true;
            }
        }
        return false;
    }

    explicit CausalConv1DCacheManager(ov::InferRequest request)
        : m_request(request) {
        ov::CompiledModel compiled_model = request.get_compiled_model();
        std::vector<std::string> execution_devices = compiled_model.get_property(ov::execution_devices);
        OPENVINO_ASSERT(!execution_devices.empty(), "Continuous batching: no execution devices found");
        m_device = execution_devices[0];

        const std::string prefix = "conv_state_table.";
        for (const auto& input : compiled_model.inputs()) {
            for (const auto& name : input.get_names()) {
                if (name.find(prefix) == 0) {
                    size_t layer_idx = std::stoul(name.substr(prefix.size()));
                    auto precision = input.get_element_type();
                    ov::PartialShape pshape = input.get_partial_shape();
                    // shape: [num_blocks, channels, kernel_size - 1]
                    OPENVINO_ASSERT(pshape.rank().get_length() == 3,
                                    "conv_state_table shape must be rank 3, got ", pshape.rank().get_length());
                    m_block_size_in_bytes += pshape[1].get_length() * pshape[2].get_length() * precision.size();

                    if (m_cache_interval == 0) {
                        m_cache_interval = static_cast<size_t>(pshape[2].get_length());
                    }

                    m_conv_shapes[layer_idx] = pshape;
                    m_conv_precisions[layer_idx] = precision;
                    m_layer_indices.push_back(layer_idx);
                    break;
                }
            }
        }

        std::sort(m_layer_indices.begin(), m_layer_indices.end());

        m_num_conv_layers = m_layer_indices.size();
        OPENVINO_ASSERT(m_num_conv_layers > 0, "CausalConv1DCacheManager: no conv_state_table inputs found");
        OPENVINO_ASSERT(m_cache_interval > 0, "CausalConv1DCacheManager: invalid cache interval");
    }

    /// @return Number of conv layers.
    size_t get_num_conv_layers() const {
        return m_num_conv_layers;
    }

    /// @return The conv cache interval (kernel_size - 1).
    size_t get_cache_interval() const {
        return m_cache_interval;
    }

    // --- ICacheManager interface ---

    size_t get_num_layers() const override {
        return m_num_conv_layers;
    }

    std::string get_device() const override {
        return m_device;
    }

    size_t get_block_size() const override {
        // Each block holds the full conv state for one sequence.
        return 1;
    }

    size_t get_block_size_in_bytes() const override {
        return m_block_size_in_bytes;
    }

    size_t get_num_allocated_blocks() const override {
        return m_num_allocated_blocks;
    }

    void allocate_cache_if_needed(size_t num_blocks) override {
        if (m_num_allocated_blocks >= num_blocks) {
            return;
        }
        m_num_allocated_blocks = num_blocks;

        for (size_t layer_idx : m_layer_indices) {
            ov::Shape cache_shape = set_num_blocks(m_conv_shapes.at(layer_idx), num_blocks);
            ov::Tensor new_state(m_conv_precisions.at(layer_idx), cache_shape);

            // Zero-initialize: new blocks must start with zero conv state
            std::memset(new_state.data(), 0, new_state.get_byte_size());

            // Copy existing data if present
            auto it = m_conv_state.find(layer_idx);
            if (it != m_conv_state.end() && it->second) {
                size_t existing_bytes = it->second.get_byte_size();
                std::memcpy(new_state.data(), it->second.data(), existing_bytes);
            }

            m_conv_state[layer_idx] = new_state;
            update_request_tensor(layer_idx);
        }
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override {
        for (const auto& [src_block_id, dst_block_ids] : block_copy_map) {
            for (size_t dst_block_id : dst_block_ids) {
                for (size_t layer_idx : m_layer_indices) {
                    ov::Shape shape = set_num_blocks(m_conv_shapes.at(layer_idx), m_num_allocated_blocks);
                    // stride = product of dims after dim 0
                    size_t stride = std::accumulate(std::next(shape.begin()), shape.end(),
                                                    size_t{1}, std::multiplies<size_t>());
                    size_t byte_stride = stride * m_conv_precisions.at(layer_idx).size();

                    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(m_conv_state.at(layer_idx).data())
                                             + src_block_id * byte_stride;
                    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(m_conv_state.at(layer_idx).data())
                                       + dst_block_id * byte_stride;
                    std::memcpy(dst_ptr, src_ptr, byte_stride);
                }
            }
        }
    }

    void clear() override {
        for (size_t layer_idx : m_layer_indices) {
            m_conv_state[layer_idx] = ov::Tensor();
        }
        m_num_allocated_blocks = 0;
    }

    // --- Conv-state-specific accessors ---

    ov::Tensor get_conv_state(size_t layer_idx) const {
        auto it = m_conv_state.find(layer_idx);
        OPENVINO_ASSERT(it != m_conv_state.end(),
                        "layer_idx = ", layer_idx, " not found in conv state map");
        return it->second;
    }
};

}  // namespace ov::genai
