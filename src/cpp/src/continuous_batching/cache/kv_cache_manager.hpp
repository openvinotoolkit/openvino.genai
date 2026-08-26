// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <list>
#include <cstdint>
#include <cstring>

#include "openvino/runtime/tensor.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "logger.hpp"
#include "utils.hpp"
namespace ov::genai {

class KVCacheManager : public ICacheManager {
    size_t m_num_layers = 0;
    std::string m_device;
    size_t m_block_size = 0; // block size is per inference device 
    std::vector<ov::element::Type> m_key_precisions, m_value_precisions;
    std::vector<ov::PartialShape> m_key_shapes, m_value_shapes;
    std::vector<ov::Tensor> m_key_cache, m_value_cache;
    size_t m_num_allocated_kv_blocks = 0, m_block_size_in_bytes = 0;
    ov::InferRequest m_request;
    ov::RemoteContext m_context;

    static ov::Shape set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks) {
        pshape[0] = num_kv_blocks;
        return pshape.get_shape();
    }

    void update_request_tensor(size_t decoder_layer_id) {
        m_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id), m_key_cache[decoder_layer_id]);
        m_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id), m_value_cache[decoder_layer_id]);
    }

public:
    /**
     * @brief Check whether the compiled model has KV cache inputs (key_cache.* / value_cache.*).
     * @param compiled_model The compiled model to inspect.
     * @return true if at least one key_cache and one value_cache input are found.
     */
    static bool has_cache_inputs(const ov::CompiledModel& compiled_model) {
        bool has_key = false, has_value = false;
        for (const auto& input : compiled_model.inputs()) {
            for (const auto& name : input.get_names()) {
                if (name.find("key_cache.") == 0)
                    has_key = true;
                else if (name.find("value_cache.") == 0)
                    has_value = true;
                if (has_key && has_value)
                    return true;
            }
        }
        return false;
    }

    explicit KVCacheManager(ov::InferRequest request) :
        m_request(request) {
        // extract information about inference device
        ov::CompiledModel compiled_model = request.get_compiled_model();
        std::vector<std::string> execution_devices = compiled_model.get_property(ov::execution_devices);
        const bool all_gpu_device =
            std::all_of(execution_devices.begin(), execution_devices.end(), [&](const std::string& device) {
                return device.find("GPU") != std::string::npos;
            });
        OPENVINO_ASSERT(all_gpu_device || execution_devices.size() == 1,
                        "Continuous batching: execution device is expected to be single CPU / single GPU / multi GPUs");
        m_device = execution_devices[0];

        if (all_gpu_device) {
            m_context = m_request.get_compiled_model().get_context();
        }
        // extract information about KV cache precisions and shapes
        size_t kv_input_index = 0;
        for (const auto& input : compiled_model.inputs()) {
            for (auto & name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                ov::PartialShape pshape;

                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    m_block_size_in_bytes += pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                    m_key_shapes.push_back(pshape);
                    m_key_precisions.push_back(cache_precision);
                    break;
                } else if (name.find("value_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    m_block_size_in_bytes += pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                    m_value_shapes.push_back(pshape);
                    m_value_precisions.push_back(cache_precision);
                    ++kv_input_index;
                    break;
                }
            }
        }

        // set block_size depending on device
        const size_t cpu_block_size = 32, gpu_block_size = 16, gpu_block_size_xattn = 256;
        bool has_xattention = false;
        if (all_gpu_device) {
            if (m_value_shapes[0][2].get_length() == gpu_block_size_xattn) {
                has_xattention = true;
            }
            if (utils::env_setup_for_print_debug_info()) {
                if (has_xattention)
                    std::cout << "[XAttention]: ENABLED on GPU device." << std::endl;
                else
                    std::cout << "[XAttention]: DISABLED on GPU device." << std::endl;
            }
        }
        m_block_size = all_gpu_device ? ( has_xattention ? gpu_block_size_xattn : gpu_block_size ) : cpu_block_size;
        m_num_layers = m_value_precisions.size();
        OPENVINO_ASSERT(m_num_layers == m_key_precisions.size(), "Invalid case: a different number of K and V caches in a LLM model");
        GENAI_INFO("[KV_TRACE] physical_manager device=%s layers=%zu block_size=%zu block_bytes=%zu remote=%s",
               m_device.c_str(),
               m_num_layers,
               m_block_size,
               m_block_size_in_bytes,
               m_context ? "true" : "false");
    }

    // --- ICacheManager interface ---

    size_t get_num_layers() const override {
        return m_num_layers;
    }

    size_t get_num_cache_tensors() const override {
        return m_num_layers * 2;
    }

    std::string get_device() const override {
        return m_device;
    }

    size_t get_block_size() const override {
        return m_block_size;
    }

    size_t get_block_size_in_bytes() const override {
        return m_block_size_in_bytes;
    }

    size_t get_num_allocated_blocks() const override {
        return m_num_allocated_kv_blocks;
    }

    ov::element::Type get_key_cache_precision(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_precisions.size());
        return m_key_precisions[decoder_layer_id];
    }

    ov::element::Type get_value_cache_precision(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_precisions.size());
        return m_value_precisions[decoder_layer_id];
    }

    // --- KV-cache-specific accessors ---

    size_t sub_byte_data_type_multiplier(const ov::element::Type data_type) const {
        if (data_type == ov::element::i4 || data_type == ov::element::u4)
            return 2;
        return 1;
    }

    void allocate_cache_if_needed(size_t num_kv_blocks) override {
        if (m_num_allocated_kv_blocks >= num_kv_blocks) {
            return;
        }
        GENAI_INFO("[KV_TRACE] allocate_physical_cache old_blocks=%zu new_blocks=%zu layers=%zu",
                   m_num_allocated_kv_blocks,
                   num_kv_blocks,
                   m_num_layers);
        try {
            m_num_allocated_kv_blocks = num_kv_blocks;

            ov::Coordinate start_key{0,0,0,0};
            ov::Coordinate start_value{0,0,0,0};

            if (m_context) {// Allocate KV caches
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
                    ov::Shape value_cache_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], num_kv_blocks);
                    ov::Shape key_cache_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], num_kv_blocks);

                    ov::Tensor key_cache = m_context.create_tensor(get_key_cache_precision(decoder_layer_id), key_cache_shape);
                    ov::Tensor value_cache = m_context.create_tensor(get_value_cache_precision(decoder_layer_id), value_cache_shape);

                    if (m_key_cache.size() > decoder_layer_id && m_key_cache[decoder_layer_id]) {
                        ov::Coordinate end_key = m_key_cache[decoder_layer_id].get_shape();
                        ov::Coordinate end_value = m_value_cache[decoder_layer_id].get_shape();

                        // copy current cache data
                        ov::RemoteTensor dst_key_roi(key_cache, start_key, end_key);
                        ov::RemoteTensor dst_value_roi(value_cache, start_value, end_value);
                        dst_key_roi.copy_from(m_key_cache[decoder_layer_id]);
                        dst_value_roi.copy_from(m_value_cache[decoder_layer_id]);
                    }

                    // set new cache tensors
                    if (m_key_cache.size() > decoder_layer_id) {
                        m_key_cache[decoder_layer_id] = key_cache;
                        m_value_cache[decoder_layer_id] = value_cache;
                    } else {
                        m_key_cache.emplace_back(key_cache);
                        m_value_cache.emplace_back(value_cache);
                    }

                    update_request_tensor(decoder_layer_id);
                }
            } else {
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
                    ov::Shape value_cache_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], num_kv_blocks);
                    ov::Shape key_cache_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], num_kv_blocks);

                    ov::element::Type key_precision = get_key_cache_precision(decoder_layer_id);
                    ov::element::Type value_precision = get_value_cache_precision(decoder_layer_id);

                    ov::Tensor key_cache(key_precision, key_cache_shape);
                    ov::Tensor value_cache(value_precision, value_cache_shape);

                    auto key_cache_roi_end = static_cast<unsigned char*>(key_cache.data());
                    auto value_cache_roi_end = static_cast<unsigned char*>(value_cache.data());
                    size_t key_roi_size_byte = 0;
                    size_t value_roi_size_byte = 0;

                    if (m_key_cache.size() > decoder_layer_id && m_key_cache[decoder_layer_id]) {
                        ov::Coordinate end_key = m_key_cache[decoder_layer_id].get_shape();
                        ov::Coordinate end_value = m_value_cache[decoder_layer_id].get_shape();
                        // copy current cache data
                        if (key_precision == ov::element::u4) {
                            size_t key_stride = std::accumulate(end_key.begin(), end_key.end(), 1, std::multiplies<size_t>());
                            size_t key_roi_size_byte = key_stride + (key_stride & 1) / sub_byte_data_type_multiplier(key_precision);
                            std::memcpy(reinterpret_cast<uint8_t*>(key_cache.data()), reinterpret_cast<uint8_t*>(m_key_cache[decoder_layer_id].data()), key_roi_size_byte);
                        } else {
                            key_roi_size_byte = m_key_cache[decoder_layer_id].get_byte_size();
                            ov::Tensor dst_key_roi(key_cache, start_key, end_key);
                            key_cache_roi_end = static_cast<unsigned char*>(key_cache.data()) + key_roi_size_byte;
                            m_key_cache[decoder_layer_id].copy_to(dst_key_roi);
                        }

                        if (value_precision == ov::element::u4) {
                            size_t value_stride = std::accumulate(end_value.begin(), end_value.end(), 1, std::multiplies<size_t>());
                            size_t value_roi_size_byte = value_stride + (value_stride & 1) / sub_byte_data_type_multiplier(value_precision);
                            std::memcpy(reinterpret_cast<uint8_t*>(value_cache.data()), reinterpret_cast<uint8_t*>(m_value_cache[decoder_layer_id].data()), value_roi_size_byte);
                        } else {
                            value_roi_size_byte = m_value_cache[decoder_layer_id].get_byte_size();
                            value_cache_roi_end = static_cast<unsigned char*>(value_cache.data()) + value_roi_size_byte;
                            ov::Tensor dst_value_roi(value_cache, start_value, end_value);
                            m_value_cache[decoder_layer_id].copy_to(dst_value_roi);
                        }
                    }

                    // set new cache tensors
                    if (m_key_cache.size() > decoder_layer_id) {
                        m_key_cache[decoder_layer_id] = key_cache;
                        m_value_cache[decoder_layer_id] = value_cache;
                    } else {
                        m_key_cache.emplace_back(key_cache);
                        m_value_cache.emplace_back(value_cache);
                    }

                    update_request_tensor(decoder_layer_id);
                }
            }
        }
        catch (ov::Exception& e) {
            if (std::string(e.what()).find("bad allocation") != std::string::npos) {
                OPENVINO_THROW("Requested KV-cache size is larger than available memory size on the system.");
            } else {
                throw;
            }
        }
    }

    ov::Tensor get_key_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size(), "decoder_layer_id = ", decoder_layer_id, ", num_layers = ", m_key_cache.size());
        return m_key_cache[decoder_layer_id];
    }

    ov::Tensor get_value_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size(), "decoder_layer_id = ", decoder_layer_id, ", num_layers = ", m_value_cache.size());
        return m_value_cache[decoder_layer_id];
    }

    /**
     * @brief Byte layout of one physical block across all layers, used as the single source of truth
     * for offload slot sizes and offsets.
     * Unlike get_block_size_in_bytes(), this accounts for sub-byte packing of u4/i4 caches.
     */
    KVCacheDiskLayout get_block_layout() const {
        std::vector<size_t> key_block_sizes(m_num_layers), value_block_sizes(m_num_layers);
        for (size_t layer = 0; layer < m_num_layers; ++layer) {
            key_block_sizes[layer] = get_block_byte_size(m_key_shapes[layer], m_key_precisions[layer]);
            value_block_sizes[layer] = get_block_byte_size(m_value_shapes[layer], m_value_precisions[layer]);
        }
        return KVCacheDiskLayout(key_block_sizes, value_block_sizes);
    }

    std::vector<uint8_t> read_block(size_t block_id) const {
        OPENVINO_ASSERT(block_id < m_num_allocated_kv_blocks, "Invalid KV cache block ID ", block_id);

        const KVCacheDiskLayout layout = get_block_layout();
        std::vector<uint8_t> block_data(layout.get_slot_size());
        for (size_t layer = 0; layer < m_num_layers; ++layer) {
            const auto key_segment = layout.get_key_segment(layer);
            const auto value_segment = layout.get_value_segment(layer);
            copy_block_from_tensor(block_data.data() + key_segment.offset, m_key_cache[layer], block_id, key_segment.size);
            copy_block_from_tensor(block_data.data() + value_segment.offset, m_value_cache[layer], block_id, value_segment.size);
        }
        return block_data;
    }

    void write_block(size_t block_id, const std::vector<uint8_t>& block_data) {
        OPENVINO_ASSERT(block_id < m_num_allocated_kv_blocks, "Invalid KV cache block ID ", block_id);

        const KVCacheDiskLayout layout = get_block_layout();
        OPENVINO_ASSERT(block_data.size() == layout.get_slot_size(),
                        "Unexpected KV cache block byte size: got ", block_data.size(),
                        ", expected ", layout.get_slot_size());

        for (size_t layer = 0; layer < m_num_layers; ++layer) {
            const auto key_segment = layout.get_key_segment(layer);
            const auto value_segment = layout.get_value_segment(layer);
            copy_block_to_tensor(m_key_cache[layer], block_id, block_data.data() + key_segment.offset, key_segment.size);
            copy_block_to_tensor(m_value_cache[layer], block_id, block_data.data() + value_segment.offset, value_segment.size);
        }
    }

private:
    static size_t get_block_byte_size(const ov::PartialShape& cache_shape, const ov::element::Type& precision) {
        const size_t elements = cache_shape[1].get_length() * cache_shape[2].get_length() * cache_shape[3].get_length();
        if (precision.bitwidth() < 8) {
            OPENVINO_ASSERT((elements * precision.bitwidth()) % 8 == 0,
                            "Sub-byte KV cache block is not byte-aligned and cannot be offloaded");
            return elements * precision.bitwidth() / 8;
        }
        return elements * precision.size();
    }

    static void assert_block_stride(const ov::Tensor& tensor, size_t block_bytes) {
        OPENVINO_ASSERT(tensor.get_byte_size() == block_bytes * tensor.get_shape()[0],
                        "KV cache tensor byte size does not match the per-block layout");
    }

    /// @return The shape of a single block, i.e. the cache shape with the block axis collapsed.
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

    static void copy_block_from_tensor(uint8_t* destination,
                                       const ov::Tensor& tensor,
                                       size_t block_id,
                                       size_t block_bytes) {
        assert_block_stride(tensor, block_bytes);
        if (tensor.is<ov::RemoteTensor>()) {
            // Device memory exposes no host pointer, so the block is staged through a host tensor.
            ov::Tensor host_block(tensor.get_element_type(), get_single_block_shape(tensor));
            make_block_roi(tensor, block_id).copy_to(host_block);
            std::memcpy(destination, host_block.data(), block_bytes);
            return;
        }
        const auto* source = static_cast<const uint8_t*>(tensor.data()) + block_id * block_bytes;
        std::memcpy(destination, source, block_bytes);
    }

    static void copy_block_to_tensor(ov::Tensor& tensor,
                                     size_t block_id,
                                     const uint8_t* source,
                                     size_t block_bytes) {
        assert_block_stride(tensor, block_bytes);
        if (tensor.is<ov::RemoteTensor>()) {
            ov::Tensor host_block(tensor.get_element_type(), get_single_block_shape(tensor));
            std::memcpy(host_block.data(), source, block_bytes);
            make_block_roi(tensor, block_id).copy_from(host_block);
            return;
        }
        auto* destination = static_cast<uint8_t*>(tensor.data()) + block_id * block_bytes;
        std::memcpy(destination, source, block_bytes);
    }

public:

    size_t get_v_head_size(size_t layer_id) const {
        return m_value_shapes[layer_id][3].get_length();
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override {
        size_t copied_blocks = 0;
        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
                ++copied_blocks;
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
                    ov::Shape key_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], m_num_allocated_kv_blocks);
                    ov::Shape value_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], m_num_allocated_kv_blocks);
                    ov::Coordinate key_src_start_roi(key_shape.size(), 0);
                    ov::Coordinate key_src_end_roi = key_shape;
                    ov::Coordinate key_dst_start_roi(key_shape.size(), 0);
                    ov::Coordinate key_dst_end_roi = key_shape;

                    ov::Coordinate value_src_start_roi(value_shape.size(), 0);
                    ov::Coordinate value_src_end_roi = value_shape;
                    ov::Coordinate value_dst_start_roi(value_shape.size(), 0);
                    ov::Coordinate value_dst_end_roi = value_shape;
                    key_src_end_roi[0] = (key_src_start_roi[0] = src_block_id) + 1;
                    value_src_end_roi[0] = (value_src_start_roi[0] = src_block_id) + 1;
                    key_dst_end_roi[0] = (key_dst_start_roi[0] = dst_block_id) + 1;
                    value_dst_end_roi[0] = (value_dst_start_roi[0] = dst_block_id) + 1;

                    auto copy_one_block = [&](ov::Tensor& dst, const ov::Tensor& src, size_t src_start, size_t dst_start, size_t stride) {
                        const bool is_remote = dst.is<ov::RemoteTensor>() || src.is<ov::RemoteTensor>();
                        if (is_remote) {
                            return;
                        }
                        auto sub_byte_multipyer = sub_byte_data_type_multiplier(dst.get_element_type());
                        OPENVINO_SUPPRESS_DEPRECATED_START
                        const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src.data()) + src_start * stride;
                        uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.data()) + dst_start * stride;
                        OPENVINO_SUPPRESS_DEPRECATED_END
                        std::memcpy(dst_ptr, src_ptr, 1 * stride);
                    };

                    const auto& key_cache_prec = m_key_cache[decoder_layer_id].get_element_type();
                    if (key_cache_prec == ov::element::u4 || key_cache_prec == ov::element::i4) {
                        size_t stride = std::accumulate(std::next(key_shape.begin()), key_shape.end(), 1, std::multiplies<size_t>()) / 2;
                        copy_one_block(m_key_cache[decoder_layer_id], m_key_cache[decoder_layer_id], key_src_start_roi[0], key_dst_start_roi[0], stride);
                    } else {
                        ov::Tensor key_src_cache_roi(m_key_cache[decoder_layer_id], key_src_start_roi, key_src_end_roi);
                        ov::Tensor key_dst_cache_roi(m_key_cache[decoder_layer_id], key_dst_start_roi, key_dst_end_roi);
                        key_src_cache_roi.copy_to(key_dst_cache_roi);
                    }

                    const auto& value_cache_prec = m_value_cache[decoder_layer_id].get_element_type();
                    if (value_cache_prec == ov::element::u4 || value_cache_prec == ov::element::i4) {
                        size_t stride = std::accumulate(std::next(value_shape.begin()), value_shape.end(), 1, std::multiplies<size_t>()) / 2;
                        copy_one_block(m_value_cache[decoder_layer_id], m_value_cache[decoder_layer_id], value_src_start_roi[0], value_dst_start_roi[0], stride);
                    } else {
                        ov::Tensor value_src_cache_roi(m_value_cache[decoder_layer_id], value_src_start_roi, value_src_end_roi);
                        ov::Tensor value_dst_cache_roi(m_value_cache[decoder_layer_id], value_dst_start_roi, value_dst_end_roi);
                        value_src_cache_roi.copy_to(value_dst_cache_roi);
                    }
                }
            }
        }
        if (copied_blocks > 0) {
            GENAI_INFO("[KV_TRACE] copy_physical_blocks count=%zu allocated_blocks=%zu",
                       copied_blocks,
                       m_num_allocated_kv_blocks);
        }
    }

    void clear() override {
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
            m_key_cache[decoder_layer_id] = ov::Tensor();
            m_value_cache[decoder_layer_id] = ov::Tensor();
        }
        m_num_allocated_kv_blocks = 0;
    }
};

}
