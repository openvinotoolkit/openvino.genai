// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <charconv>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <system_error>
#include <vector>
#include <list>

#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"

namespace ov::genai {

/**
 * @brief Manages physical state tensors for linear attention operations
 *        (CausalConv1D, GatedDeltaNet, and similar ops).
 *
 * Discovers all model inputs matching the "<prefix>_state_table.N" naming
 * convention and manages their allocation, explicit block zeroing, and block
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
    ov::RemoteContext m_context;  ///< Valid only on GPU; empty on CPU.
    std::vector<StateTableGroup> m_groups;
    size_t m_num_cache_tensors = 0;
    size_t m_num_allocated_blocks = 0;
    size_t m_block_size_in_bytes = 0;
    ov::InferRequest m_request;

    static ov::Shape make_shape(const ov::PartialShape& pshape, size_t num_blocks) {
        ov::PartialShape ps = pshape;
        ps[0] = num_blocks;
        OPENVINO_ASSERT(ps.is_static(), "State table shape must be static after setting block dimension");
        return ps.get_shape();
    }

    /// Bytes per block for a single layer (product of all dims except dim-0).
    static size_t layer_bytes_per_block(const ov::PartialShape& pshape, ov::element::Type precision) {
        size_t elems = 1;
        for (size_t d = 1; d < static_cast<size_t>(pshape.rank().get_length()); ++d) {
            OPENVINO_ASSERT(pshape[d].is_static(), "State table dimension ", d, " must be static");
            elems *= static_cast<size_t>(pshape[d].get_length());
        }
        return elems * precision.size();
    }

    /// Extract prefix and layer index from a name like "conv_state_table.7".
    /// Returns {"conv_state_table", 7}.  Returns {"", 0} if not a state table name.
    static std::pair<std::string, size_t> parse_state_table_name(const std::string& name) {
        const std::string suffix = "_state_table.";
        const size_t pos = name.find(suffix);
        if (pos == std::string::npos) {
            return {"", 0};
        }

        const size_t layer_idx_pos = pos + suffix.size();
        if (layer_idx_pos == name.size()) {
            return {"", 0};
        }

        size_t layer_idx = 0;
        const char* begin = name.data() + layer_idx_pos;
        const char* end = name.data() + name.size();
        const auto [ptr, error_code] = std::from_chars(begin, end, layer_idx);
        if (error_code != std::errc{} || ptr != end) {
            return {"", 0};
        }

        std::string prefix = name.substr(0, pos + suffix.size() - 1);  // e.g. "conv_state_table"
        return {prefix, layer_idx};
    }

    // --- Block-level raw byte I/O, used by LinearAttentionOffloadCache. Mirrors KVCacheManager's
    // equivalent helpers; kept separate since the tensor layouts (state-table groups vs key/value) differ. ---

    static void assert_block_stride(const ov::Tensor& tensor, size_t block_bytes) {
        OPENVINO_ASSERT(tensor.get_byte_size() == block_bytes * tensor.get_shape()[0],
                        "Linear attention cache tensor byte size does not match the per-block layout");
    }

    static ov::Shape get_single_block_shape(const ov::Tensor& tensor) {
        ov::Shape shape = tensor.get_shape();
        shape[0] = 1;
        return shape;
    }

    static ov::RemoteTensor make_block_roi(const ov::Tensor& tensor, size_t block_id) {
        const ov::Shape shape = tensor.get_shape();
        ov::Coordinate begin(shape.size(), 0);
        ov::Coordinate end(shape.begin(), shape.end());
        begin[0] = block_id;
        end[0] = block_id + 1;
        return ov::RemoteTensor(tensor.as<ov::RemoteTensor>(), begin, end);
    }

    static void copy_block_from_tensor(uint8_t* destination, const ov::Tensor& tensor, size_t block_id, size_t block_bytes) {
        assert_block_stride(tensor, block_bytes);
        if (tensor.is<ov::RemoteTensor>()) {
            // Device memory exposes no host pointer, so the copy targets the caller's buffer directly.
            ov::Tensor destination_view(tensor.get_element_type(), get_single_block_shape(tensor), destination);
            make_block_roi(tensor, block_id).copy_to(destination_view);
            return;
        }
        const auto* source = static_cast<const uint8_t*>(tensor.data()) + block_id * block_bytes;
        std::memcpy(destination, source, block_bytes);
    }

    static void copy_block_to_tensor(ov::Tensor& tensor, size_t block_id, const uint8_t* source, size_t block_bytes) {
        assert_block_stride(tensor, block_bytes);
        if (tensor.is<ov::RemoteTensor>()) {
            ov::Tensor source_view(tensor.get_element_type(), get_single_block_shape(tensor), const_cast<uint8_t*>(source));
            make_block_roi(tensor, block_id).copy_from(source_view);
            return;
        }
        auto* destination = static_cast<uint8_t*>(tensor.data()) + block_id * block_bytes;
        std::memcpy(destination, source, block_bytes);
    }

public:
    /**
     * @brief Check whether the compiled model has any state table inputs (*_state_table.*).
     */
    static bool has_cache_inputs(const ov::CompiledModel& compiled_model) {
        const auto inputs = compiled_model.inputs();
        for (const auto& input : inputs) {
            for (const auto& name : input.get_names()) {
                const auto [prefix, layer_idx] = parse_state_table_name(name);
                if (!prefix.empty()) {
                    return true;
                }
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

        const bool all_gpu = std::all_of(execution_devices.begin(), execution_devices.end(),
            [](const std::string& d) { return d.find("GPU") != std::string::npos; });
        if (all_gpu) {
            m_context = compiled_model.get_context();
        }

        // Discover state table inputs and group by prefix.
        std::map<std::string, StateTableGroup> groups_map;

        const auto inputs = compiled_model.inputs();
        for (const auto& input : inputs) {
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
            m_num_cache_tensors += group.layers.size();
            m_groups.push_back(std::move(group));
        }

        OPENVINO_ASSERT(!m_groups.empty(),
                        "LinearAttentionCacheManager: no *_state_table.* inputs found");
    }

    // --- ICacheManager interface ---

    size_t get_num_layers() const override {
        const auto& group = m_groups.front();
        return group.layers.size();
    }

    size_t get_num_cache_tensors() const override {
        return m_num_cache_tensors;
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

                if (m_context) {
                    ov::Tensor new_tensor = m_context.create_tensor(info.precision, shape);

                    if (info.tensor) {
                        const size_t rank = shape.size();
                        ov::Coordinate full_start(rank, 0);
                        const ov::Shape& old_shape = info.tensor.get_shape();
                        ov::RemoteTensor dst_old(new_tensor, full_start, old_shape);
                        dst_old.copy_from(info.tensor);
                    }

                    info.tensor = new_tensor;
                } else {
                    ov::Tensor new_tensor(info.precision, shape);

                    if (info.tensor) {
                        std::memcpy(new_tensor.data(), info.tensor.data(), info.tensor.get_byte_size());
                    }

                    info.tensor = new_tensor;
                }

                m_request.set_tensor(info.full_name, info.tensor);
            }
        }
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override {
        for (const auto& [src_block, dst_blocks] : block_copy_map) {
            for (size_t dst_block : dst_blocks) {
                for (const auto& group : m_groups) {
                    for (size_t layer_idx : group.sorted_indices) {
                        const auto& info = group.layers.at(layer_idx);
                        const ov::Shape& shape = info.tensor.get_shape();
                        const size_t rank = shape.size();

                        ov::Coordinate src_start(rank, 0), src_end = shape;
                        src_start[0] = src_block;
                        src_end[0]   = src_block + 1;

                        ov::Coordinate dst_start(rank, 0), dst_end = shape;
                        dst_start[0] = dst_block;
                        dst_end[0]   = dst_block + 1;

                        ov::Tensor src_roi(info.tensor, src_start, src_end);
                        ov::Tensor dst_roi(info.tensor, dst_start, dst_end);
                        src_roi.copy_to(dst_roi);
                    }
                }
            }
        }
    }

    void zero_blocks(const std::set<size_t>& block_indices) override {
        if (block_indices.empty()) {
            return;
        }

        for (const auto& group : m_groups) {
            for (size_t layer_idx : group.sorted_indices) {
                const auto& info = group.layers.at(layer_idx);
                OPENVINO_ASSERT(info.tensor, "Linear attention state tensor is not allocated");
                const ov::Shape& shape = info.tensor.get_shape();
                const size_t rank = shape.size();
                ov::Shape block_shape = shape;
                block_shape[0] = 1;

                ov::Tensor zero_block(info.precision, block_shape);
                std::memset(zero_block.data(), 0, zero_block.get_byte_size());

                for (size_t block_index : block_indices) {
                    OPENVINO_ASSERT(block_index < shape[0],
                                    "Linear attention block index ", block_index,
                                    " is out of allocated range ", shape[0]);
                    ov::Coordinate dst_start(rank, 0), dst_end = shape;
                    dst_start[0] = block_index;
                    dst_end[0] = block_index + 1;

                    ov::Tensor dst_roi(info.tensor, dst_start, dst_end);
                    zero_block.copy_to(dst_roi);
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

    /**
     * @brief Byte layout of one physical block across all state-table groups, used as the single
     * source of truth for offload slot sizes and offsets. Mirrors KVCacheManager::get_block_layout();
     * linear attention blocks have no key/value split, so the "value" side of the layout is left empty.
     */
    KVCacheDiskLayout get_block_layout() const {
        std::vector<size_t> segment_sizes;
        for (const auto& group : m_groups) {
            for (size_t layer_idx : group.sorted_indices) {
                const auto& info = group.layers.at(layer_idx);
                segment_sizes.push_back(layer_bytes_per_block(info.partial_shape, info.precision));
            }
        }
        return KVCacheDiskLayout(segment_sizes, std::vector<size_t>(segment_sizes.size(), 0));
    }

    std::vector<uint8_t> read_block(size_t block_id) const {
        OPENVINO_ASSERT(block_id < m_num_allocated_blocks, "Invalid linear attention cache block ID ", block_id);
        const KVCacheDiskLayout layout = get_block_layout();
        std::vector<uint8_t> block_data(layout.get_slot_size());
        size_t segment_idx = 0;
        for (const auto& group : m_groups) {
            for (size_t layer_idx : group.sorted_indices) {
                const auto& info = group.layers.at(layer_idx);
                const auto segment = layout.get_key_segment(segment_idx++);
                copy_block_from_tensor(block_data.data() + segment.offset, info.tensor, block_id, segment.size);
            }
        }
        return block_data;
    }

    void write_block(size_t block_id, const std::vector<uint8_t>& block_data) {
        OPENVINO_ASSERT(block_id < m_num_allocated_blocks, "Invalid linear attention cache block ID ", block_id);
        const KVCacheDiskLayout layout = get_block_layout();
        OPENVINO_ASSERT(block_data.size() == layout.get_slot_size(),
                        "Unexpected linear attention cache block byte size: got ", block_data.size(),
                        ", expected ", layout.get_slot_size());
        size_t segment_idx = 0;
        for (auto& group : m_groups) {
            for (size_t layer_idx : group.sorted_indices) {
                auto& info = group.layers.at(layer_idx);
                const auto segment = layout.get_key_segment(segment_idx++);
                copy_block_to_tensor(info.tensor, block_id, block_data.data() + segment.offset, segment.size);
            }
        }
    }
};

}  // namespace ov::genai
