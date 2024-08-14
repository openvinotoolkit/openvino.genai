// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <list>

#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"

#include "device_config.hpp"

namespace ov::genai {
class CacheManager {
    std::vector<ov::Tensor> m_key_cache;
    std::vector<ov::Tensor> m_value_cache;

public:
    CacheManager(const DeviceConfig& device_config, ov::InferRequest infer_request) {
        m_key_cache.reserve(device_config.get_num_layers());
        m_value_cache.reserve(device_config.get_num_layers());

        for (size_t decoder_layer_id = 0; decoder_layer_id < device_config.get_num_layers(); ++decoder_layer_id) {
            ov::Tensor key_cache = infer_request.get_input_tensor(2 + decoder_layer_id * 2);
            std::memset(key_cache.data(), 0, key_cache.get_byte_size());
            m_key_cache.emplace_back(key_cache);

            ov::Tensor value_cache = infer_request.get_input_tensor(2 + decoder_layer_id * 2 + 1);
            std::memset(value_cache.data(), 0, value_cache.get_byte_size());
            m_value_cache.emplace_back(value_cache);
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
        const size_t num_decoder_layers = m_key_cache.size();
        OPENVINO_ASSERT(num_decoder_layers > 0 && m_value_cache.size() == num_decoder_layers,
            "Internal error: KV caches must be allocated");

        ov::Shape key_shape = m_key_cache[0].get_shape();
        ov::Shape value_shape = m_value_cache[0].get_shape();

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

                for (size_t decoder_layer_id = 0; decoder_layer_id < num_decoder_layers; ++decoder_layer_id) {
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
