// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <list>

#include "openvino/runtime/tensor.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"
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

    struct QSAIndexCache {
        std::string name;
        ov::element::Type precision;
        ov::PartialShape partial_shape;
        ov::Tensor tensor;
    };
    std::vector<QSAIndexCache> m_qsa_index_caches;

    static bool is_qsa_index_cache_input(const std::string& name) {
        // Per-layer caches of sparse attention (PagedQSAIndexer), paged by the same block table as the KV cache.
        for (const char* prefix : {"index_key_cache.", "index_block_rope_cache.", "block_index_keys_cache."}) {
            if (name.find(prefix) == 0) {
                return true;
            }
        }
        return false;
    }

    void allocate_qsa_index_caches(size_t num_kv_blocks) {
        for (auto& qsa_index_cache : m_qsa_index_caches) {
            const ov::Shape shape = set_kv_blocks(qsa_index_cache.partial_shape, num_kv_blocks);
            ov::Tensor new_tensor = m_context ? m_context.create_tensor(qsa_index_cache.precision, shape)
                                              : ov::Tensor(qsa_index_cache.precision, shape);
            if (qsa_index_cache.tensor) {
                // copy current cache data
                ov::Coordinate start(shape.size(), 0);
                ov::Coordinate end = qsa_index_cache.tensor.get_shape();
                if (m_context) {
                    ov::RemoteTensor dst_roi(new_tensor, start, end);
                    dst_roi.copy_from(qsa_index_cache.tensor);
                } else {
                    ov::Tensor dst_roi(new_tensor, start, end);
                    qsa_index_cache.tensor.copy_to(dst_roi);
                }
            }
            qsa_index_cache.tensor = new_tensor;
            m_request.set_tensor(qsa_index_cache.name, qsa_index_cache.tensor);
        }
    }

    void copy_qsa_index_cache_block(size_t src_block_id, size_t dst_block_id) {
        for (auto& qsa_index_cache : m_qsa_index_caches) {
            const ov::Shape& shape = qsa_index_cache.tensor.get_shape();
            ov::Coordinate src_start(shape.size(), 0), src_end = shape;
            ov::Coordinate dst_start(shape.size(), 0), dst_end = shape;
            src_end[0] = (src_start[0] = src_block_id) + 1;
            dst_end[0] = (dst_start[0] = dst_block_id) + 1;
            ov::Tensor src_roi(qsa_index_cache.tensor, src_start, src_end);
            ov::Tensor dst_roi(qsa_index_cache.tensor, dst_start, dst_end);
            src_roi.copy_to(dst_roi);
        }
    }

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
                } else if (is_qsa_index_cache_input(name)) {
                    pshape = input.get_partial_shape();
                    const size_t expected_rank = name.find("index_block_rope_cache.") == 0 ? 3 : 4;
                    OPENVINO_ASSERT(pshape.rank().is_static() && pshape.rank().get_length() == expected_rank,
                        "QSA index cache ", name, " must have static, positive rank ", expected_rank, ", got ", pshape);
                    size_t block_bytes = cache_precision.size();
                    for (size_t axis = 1; axis < pshape.size(); ++axis) {
                        OPENVINO_ASSERT(pshape[axis].is_static(),
                            "QSA index cache ", name, " must have static dimensions except axis 0, got ", pshape);
                        block_bytes *= pshape[axis].get_length();
                    }
                    m_block_size_in_bytes += block_bytes;
                    m_qsa_index_caches.push_back({name, cache_precision, pshape, {}});
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

        // Validate QSA index caches
        for (const auto& qsa_index_cache : m_qsa_index_caches) {
            if (qsa_index_cache.name.find("index_key_cache.") == 0) {
                OPENVINO_ASSERT(static_cast<size_t>(qsa_index_cache.partial_shape[2].get_length()) == m_block_size,
                    "Sparse attention index cache ", qsa_index_cache.name, " must have ", m_block_size,
                    " tokens per block to match the KV cache, got ", qsa_index_cache.partial_shape
                );
            }
        }
    }

    bool has_qsa_index_caches() const {
        return !m_qsa_index_caches.empty();
    }

    // --- ICacheManager interface ---

    size_t get_num_layers() const override {
        return m_num_layers;
    }

    size_t get_num_cache_tensors() const override {
        return m_num_layers * 2 + m_qsa_index_caches.size();
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
            allocate_qsa_index_caches(num_kv_blocks);
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

    size_t get_v_head_size(size_t layer_id) const {
        return m_value_shapes[layer_id][3].get_length();
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override {
        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
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
                copy_qsa_index_cache_block(src_block_id, dst_block_id);
            }
        }
    }

    void clear() override {
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_layers; ++decoder_layer_id) {
            m_key_cache[decoder_layer_id] = ov::Tensor();
            m_value_cache[decoder_layer_id] = ov::Tensor();
        }
        for (auto& qsa_index_cache : m_qsa_index_caches) {
            qsa_index_cache.tensor = ov::Tensor();
        }
        m_num_allocated_kv_blocks = 0;
    }
};

}
