// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <list>

#include "openvino/runtime/tensor.hpp"

#include "device_config.hpp"

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
        if (m_num_allocated_kv_blocks > 0) {
            increase_cache(num_kv_blocks);
            return;
        }
        m_num_allocated_kv_blocks = num_kv_blocks;
        ov::Shape value_cache_shape = set_first_dim_and_make_static(m_device_config.get_value_cache_shape(), num_kv_blocks);
        ov::Shape key_cache_shape = set_first_dim_and_make_static(m_device_config.get_key_cache_shape(), num_kv_blocks);

        const std::string device_name = m_device_config.get_device();

        if (device_name.find("GPU") == std::string::npos) {// Allocate KV caches
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                ov::Tensor key_cache(m_device_config.get_cache_precision(), key_cache_shape);
                ov::Tensor value_cache(m_device_config.get_cache_precision(), value_cache_shape);

                // Some optimizations like AVX2, AVX512, AMX require a minimal shape and 
                // perform multiplying by zero on the excess data. Uninitialized tensor data contain NAN's, 
                // so NAN * 0 returns non-zero invalid data.
                // So we need to set zeros to all newly allocated tensors data.
                std::memset(key_cache.data(), 0, key_cache.get_byte_size());
                std::memset(value_cache.data(), 0, value_cache.get_byte_size());

                m_key_cache.emplace_back(key_cache);
                m_value_cache.emplace_back(value_cache);

                update_request_tensor(decoder_layer_id);
            }
        } else {
            auto remote_context = m_core.get_default_context(device_name);
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                ov::Tensor key_cache = remote_context.create_tensor(m_device_config.get_cache_precision(),
                                                                    key_cache_shape);
                ov::Tensor value_cache = remote_context.create_tensor(m_device_config.get_cache_precision(),
                                                                      value_cache_shape);

                m_key_cache.emplace_back(key_cache);
                m_value_cache.emplace_back(value_cache);

                update_request_tensor(decoder_layer_id);
            }
        }
    }

    void increase_cache(size_t num_kv_blocks) {
        OPENVINO_ASSERT(num_kv_blocks > m_num_allocated_kv_blocks);
        ov::Shape new_value_cache_shape = set_first_dim_and_make_static(m_device_config.get_value_cache_shape(), num_kv_blocks);
        ov::Shape new_key_cache_shape = set_first_dim_and_make_static(m_device_config.get_key_cache_shape(), num_kv_blocks);

        const std::string device_name = m_device_config.get_device();
        ov::Coordinate start_key{0,0,0,0};
        ov::Coordinate start_value{0,0,0,0};

        if (device_name.find("GPU") == std::string::npos) {
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                ov::Coordinate end_key(m_key_cache[decoder_layer_id].get_shape());
                ov::Coordinate end_value(m_value_cache[decoder_layer_id].get_shape());

                ov::Tensor key_cache(m_device_config.get_cache_precision(), new_value_cache_shape);
                ov::Tensor value_cache(m_device_config.get_cache_precision(), new_key_cache_shape);
                
                // copy current cache data
                ov::Tensor dst_key_roi(key_cache, start_key, end_key);
                ov::Tensor dst_value_roi(value_cache, start_value, end_value);
                m_key_cache[decoder_layer_id].copy_to(dst_key_roi);
                m_value_cache[decoder_layer_id].copy_to(dst_value_roi);

                // Some optimizations like AVX2, AVX512, AMX require a minimal shape and 
                // perform multiplying by zero on the excess data. Uninitialized tensor data contain NAN's, 
                // so NAN * 0 returns non-zero invalid data.
                // So we need to set zeros to all newly allocated tensors data.
                auto key_cache_roi_end = static_cast<unsigned char*>(key_cache.data()) + dst_key_roi.get_byte_size();
                auto value_cache_roi_end = static_cast<unsigned char*>(value_cache.data()) + dst_value_roi.get_byte_size();
                std::memset(key_cache_roi_end, 0, key_cache.get_byte_size() - dst_key_roi.get_byte_size());
                std::memset(value_cache_roi_end, 0, value_cache.get_byte_size() - dst_value_roi.get_byte_size());
                
                // set new cache tensors
                m_key_cache[decoder_layer_id] = key_cache;
                m_value_cache[decoder_layer_id] = value_cache;

                update_request_tensor(decoder_layer_id);
            }
        } else {
            auto remote_context = m_core.get_default_context(device_name);
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                ov::Coordinate end_key(m_key_cache[decoder_layer_id].get_shape());
                ov::Coordinate end_value(m_value_cache[decoder_layer_id].get_shape());

                ov::Tensor key_cache = remote_context.create_tensor(m_device_config.get_cache_precision(), new_value_cache_shape);
                ov::Tensor value_cache = remote_context.create_tensor(m_device_config.get_cache_precision(), new_key_cache_shape);
                
                // copy current cache data
                ov::Tensor dst_key_roi(key_cache, start_key, end_key);
                ov::Tensor dst_value_roi(value_cache, start_value, end_value);
                m_key_cache[decoder_layer_id].copy_to(dst_key_roi);
                m_value_cache[decoder_layer_id].copy_to(dst_value_roi);

                // set new cache tensors
                m_key_cache[decoder_layer_id] = key_cache;
                m_value_cache[decoder_layer_id] = value_cache;
                
                update_request_tensor(decoder_layer_id);
            }
        }
        m_num_allocated_kv_blocks = num_kv_blocks;
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
        ov::Shape key_shape = set_first_dim_and_make_static(m_device_config.get_key_cache_shape(), m_num_allocated_kv_blocks);
        ov::Shape value_shape = set_first_dim_and_make_static(m_device_config.get_value_cache_shape(), m_num_allocated_kv_blocks);

        ov::Coordinate key_src_start_roi(key_shape.size(), 0);
        ov::Coordinate key_src_end_roi = key_shape;
        ov::Coordinate key_dst_start_roi(key_shape.size(), 0);
        ov::Coordinate key_dst_end_roi = key_shape;

        ov::Coordinate value_src_start_roi(value_shape.size(), 0);
        ov::Coordinate value_src_end_roi = value_shape;
        ov::Coordinate value_dst_start_roi(value_shape.size(), 0);
        ov::Coordinate value_dst_end_roi = value_shape;

        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            key_src_end_roi[0] = (key_src_start_roi[0] = src_block_id) + 1;
            value_src_end_roi[0] = (value_src_start_roi[0] = src_block_id) + 1;

            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
                key_dst_end_roi[0] = (key_dst_start_roi[0] = dst_block_id) + 1;
                value_dst_end_roi[0] = (value_dst_start_roi[0] = dst_block_id) + 1;

                for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
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
};
}
