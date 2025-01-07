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
    ov::Core m_core;

public:
    explicit CacheManager(const DeviceConfig &device_config, ov::Core core) :
            m_device_config(device_config),
            m_core(core) {
        m_key_cache.reserve(m_device_config.get_num_layers());
        m_value_cache.reserve(m_device_config.get_num_layers());

        const std::string device_name = device_config.get_device();
        if (device_name.find("GPU") == std::string::npos) {// Allocate KV caches
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                ov::Tensor key_cache(device_config.get_key_cache_precision(), device_config.get_key_cache_shape());
                ov::Tensor value_cache(device_config.get_value_cache_precision(), device_config.get_value_cache_shape());

                // force allocation
                std::memset(key_cache.data(), 0, key_cache.get_byte_size());
                std::memset(value_cache.data(), 0, value_cache.get_byte_size());

                m_key_cache.emplace_back(key_cache);
                m_value_cache.emplace_back(value_cache);
            }
        } else {
            auto remote_context = m_core.get_default_context(device_name);
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                ov::Tensor key_cache = remote_context.create_tensor(device_config.get_key_cache_precision(),
                                                                    device_config.get_key_cache_shape());
                ov::Tensor value_cache = remote_context.create_tensor(device_config.get_value_cache_precision(),
                                                                      device_config.get_value_cache_shape());

                m_key_cache.emplace_back(key_cache);
                m_value_cache.emplace_back(value_cache);
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
        ov::Shape key_shape = m_device_config.get_key_cache_shape();
        ov::Shape value_shape = m_device_config.get_value_cache_shape();

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

                    key_src_cache_roi.copy_to(key_dst_cache_roi);
                    const auto& value_cache_prec = m_value_cache[decoder_layer_id].get_element_type();
                    if (value_cache_prec != ov::element::u4 && value_cache_prec != ov::element::i4)  {
                        ov::Tensor value_src_cache_roi(m_value_cache[decoder_layer_id], value_src_start_roi, value_src_end_roi);
                        ov::Tensor value_dst_cache_roi(m_value_cache[decoder_layer_id], value_dst_start_roi, value_dst_end_roi);
                        value_src_cache_roi.copy_to(value_dst_cache_roi);
                    } else {
                        auto copy_data = [&](ov::Tensor& dst, const ov::Tensor& src) {
                            auto sub_byte_multipyer = 8 / dst.get_element_type().bitwidth();
                            size_t stride = std::accumulate(std::next(value_shape.begin()), value_shape.end(), 1, std::multiplies<size_t>()) / sub_byte_multipyer;
                            const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src.data()) + value_src_start_roi[0] * stride;
                            uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.data()) + value_dst_start_roi[0] * stride;
                            std::memcpy(dst_ptr, src_ptr, (value_src_end_roi[0] - value_src_start_roi[0]) * stride);
                        };
                        copy_data(m_value_cache[decoder_layer_id], m_value_cache[decoder_layer_id]);
                    }
                }
            }
        }
    }
};
}
