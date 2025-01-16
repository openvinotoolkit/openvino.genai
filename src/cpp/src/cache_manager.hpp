// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <list>
#include "openvino/runtime/tensor.hpp"
#include "device_config.hpp"

#ifndef _WIN32
#include <sys/mman.h>
#include "openvino/core/shape.hpp"


class TensorMmapAllocator { 
    size_t m_total_size;
    void* m_data;
 
public: 
    TensorMmapAllocator(size_t total_size) : 
        m_total_size(total_size) { } 
  
    void* allocate(size_t bytes, size_t) { 
        if (m_total_size == bytes) { 
            m_data = mmap(NULL,  bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            OPENVINO_ASSERT(m_data != MAP_FAILED);
            return m_data; 
        } 
        throw std::runtime_error{"Unexpected number of bytes was requested to allocate."}; 
    } 
  
    void deallocate(void*, size_t bytes, size_t) { 
        if (m_total_size != bytes) { 
            throw std::runtime_error{"Unexpected number of bytes was requested to deallocate."}; 
        }
        munmap(m_data, bytes);
    } 
  
    bool is_equal(const TensorMmapAllocator& other) const noexcept { 
        return this == &other; 
    } 
}; 

#endif

namespace ov::genai {
class CacheManager {
    DeviceConfig m_device_config;
    std::vector<ov::Tensor> m_key_cache;
    std::vector<ov::Tensor> m_value_cache;
    size_t m_num_allocated_kv_blocks = 0;
    ov::Core m_core;
    ov::InferRequest m_request;

    ov::Shape set_first_dim_and_make_static(const ov::PartialShape& shape, size_t dim) {
        ov::PartialShape res_shape = shape;
        res_shape[0] = dim;
        OPENVINO_ASSERT(res_shape.is_static());
        return res_shape.to_shape();
    }

    void update_request_tensor(size_t decoder_layer_id) {
        m_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id), m_key_cache[decoder_layer_id]);
        m_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id), m_value_cache[decoder_layer_id]);
    }

public:
    explicit CacheManager(const DeviceConfig &device_config, ov::InferRequest request, ov::Core core) :
            m_device_config(device_config),
            m_request(request),
            m_core(core) {
        m_key_cache.reserve(m_device_config.get_num_layers());
        m_value_cache.reserve(m_device_config.get_num_layers());
    }

    void allocate_cache_if_needed(size_t num_kv_blocks) {
        if (m_num_allocated_kv_blocks >= num_kv_blocks) {
            return;
        }
        OPENVINO_ASSERT(m_key_cache.size() == m_value_cache.size());
        m_num_allocated_kv_blocks = num_kv_blocks;

        const std::string device_name = m_device_config.get_device();

        ov::Coordinate start_key{0,0,0,0};
        ov::Coordinate start_value{0,0,0,0};

        if (device_name.find("GPU") == std::string::npos) {// Allocate KV caches
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                ov::Shape value_cache_shape = set_first_dim_and_make_static(m_device_config.get_value_cache_shape(decoder_layer_id), num_kv_blocks);
                ov::Shape key_cache_shape = set_first_dim_and_make_static(m_device_config.get_key_cache_shape(decoder_layer_id), num_kv_blocks);
#ifdef _WIN32
                ov::Tensor key_cache(m_device_config.get_cache_precision(), key_cache_shape);
                ov::Tensor value_cache(m_device_config.get_cache_precision(), value_cache_shape);
#else
                auto key_size = ov::shape_size(key_cache_shape) * m_device_config.get_cache_precision().size();
                auto value_size = ov::shape_size(value_cache_shape) * m_device_config.get_cache_precision().size();

                ov::Tensor key_cache = ov::Tensor(m_device_config.get_cache_precision(), key_cache_shape, TensorMmapAllocator(key_size));
                ov::Tensor value_cache = ov::Tensor(m_device_config.get_cache_precision(), value_cache_shape, TensorMmapAllocator(value_size));

#endif

                auto key_cache_roi_end = static_cast<unsigned char*>(key_cache.data());
                auto value_cache_roi_end = static_cast<unsigned char*>(value_cache.data());
                size_t key_roi_size_byte = 0;
                size_t value_roi_size_byte = 0;

                if (m_key_cache.size() > decoder_layer_id) {
                    ov::Coordinate end_key = m_key_cache[decoder_layer_id].get_shape();
                    ov::Coordinate end_value = m_value_cache[decoder_layer_id].get_shape();

                    key_roi_size_byte = m_key_cache[decoder_layer_id].get_byte_size();
                    value_roi_size_byte = m_value_cache[decoder_layer_id].get_byte_size();
                    key_cache_roi_end = static_cast<unsigned char*>(key_cache.data()) + key_roi_size_byte;
                    value_cache_roi_end = static_cast<unsigned char*>(value_cache.data()) + value_roi_size_byte;
                    
                    // copy current cache data
                    ov::Tensor dst_key_roi(key_cache, start_key, end_key);
                    ov::Tensor dst_value_roi(value_cache, start_value, end_value);

                    m_key_cache[decoder_layer_id].copy_to(dst_key_roi);
                    m_value_cache[decoder_layer_id].copy_to(dst_value_roi);

                }

#ifdef _WIN32
                // Some optimizations like AVX2, AVX512, AMX require a minimal shape and 
                // perform multiplying by zero on the excess data. Uninitialized tensor data contain NAN's, 
                // so NAN * 0 returns non-zero invalid data.
                // So we need to set zeros to all newly allocated tensors data.
                std::memset(key_cache_roi_end, 0, key_cache.get_byte_size() - key_roi_size_byte);
                std::memset(value_cache_roi_end, 0, value_cache.get_byte_size() - value_roi_size_byte);
#endif
                // set new cache tensors
                if (m_key_cache.size() > decoder_layer_id) {
                    m_key_cache[decoder_layer_id] = key_cache;
                    m_value_cache[decoder_layer_id] = value_cache;
                }
                else {
                    m_key_cache.emplace_back(key_cache);
                    m_value_cache.emplace_back(value_cache);
                }

                update_request_tensor(decoder_layer_id);
            }
        } else {
            auto remote_context = m_core.get_default_context(device_name);
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                ov::Shape value_cache_shape = set_first_dim_and_make_static(m_device_config.get_value_cache_shape(decoder_layer_id), num_kv_blocks);
                ov::Shape key_cache_shape = set_first_dim_and_make_static(m_device_config.get_key_cache_shape(decoder_layer_id), num_kv_blocks);
                ov::Tensor key_cache = remote_context.create_tensor(m_device_config.get_cache_precision(),
                                                                    key_cache_shape);
                ov::Tensor value_cache = remote_context.create_tensor(m_device_config.get_cache_precision(),
                                                                      value_cache_shape);
                
                if (m_key_cache.size() > decoder_layer_id) {
                    ov::Coordinate end_key = m_key_cache[decoder_layer_id].get_shape();
                    ov::Coordinate end_value = m_value_cache[decoder_layer_id].get_shape();

                    // copy current cache data
                    ov::RemoteTensor dst_key_roi(key_cache, start_key, end_key);
                    ov::RemoteTensor dst_value_roi(value_cache, start_value, end_value);
                    dst_key_roi.copy_from(m_key_cache[decoder_layer_id]);
                    dst_value_roi.copy_from(m_value_cache[decoder_layer_id]);

                    m_key_cache[decoder_layer_id] = key_cache;
                    m_value_cache[decoder_layer_id] = value_cache;
                }
                else {
                    m_key_cache.emplace_back(key_cache);
                    m_value_cache.emplace_back(value_cache);
                }
                update_request_tensor(decoder_layer_id);
            }
        }
    }

    ov::Tensor get_key_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size());
        return m_key_cache[decoder_layer_id];
    }

    ov::Tensor get_value_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size());
        return m_value_cache[decoder_layer_id];
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
                for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                    ov::Shape key_shape = set_first_dim_and_make_static(m_device_config.get_key_cache_shape(decoder_layer_id), m_num_allocated_kv_blocks);
                    ov::Shape value_shape = set_first_dim_and_make_static(m_device_config.get_value_cache_shape(decoder_layer_id), m_num_allocated_kv_blocks);
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

                    ov::Tensor key_src_cache_roi(m_key_cache[decoder_layer_id], key_src_start_roi, key_src_end_roi);
                    ov::Tensor key_dst_cache_roi(m_key_cache[decoder_layer_id], key_dst_start_roi, key_dst_end_roi);

                    ov::Tensor value_src_cache_roi(m_value_cache[decoder_layer_id], value_src_start_roi, value_src_end_roi);
                    ov::Tensor value_dst_cache_roi(m_value_cache[decoder_layer_id], value_dst_start_roi, value_dst_end_roi);

                    key_src_cache_roi.copy_to(key_dst_cache_roi);
                    value_src_cache_roi.copy_to(value_dst_cache_roi);
                }
            }
        }
    }

    std::shared_ptr<Core> get_core() {
        return std::make_shared<Core>(m_core);
    }

    std::shared_ptr<DeviceConfig> get_device_config() {
        return std::make_shared<DeviceConfig>(m_device_config);
    }
};
}
