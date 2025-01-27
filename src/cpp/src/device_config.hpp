// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/core.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

#include "openvino/genai/scheduler_config.hpp"

namespace ov::genai {

/**
 * Per layer KV cache size configuration
 */
struct KVHeadConfig {
    size_t num_v_heads, num_k_heads;
    size_t v_head_size, k_head_size;
};

class DeviceConfig {
    std::vector<ov::PartialShape> m_key_cache_shape, m_value_cache_shape;
    std::vector<KVHeadConfig> m_kv_heads_config;
    size_t m_block_size = 0; // block size is per inference device 
    std::string m_device;

    size_t get_block_size_by_device(const std::string& device) const {
        const size_t cpu_block_size = 32, gpu_block_size = 16;
        const bool is_gpu = device.find("GPU") != std::string::npos;
        return is_gpu ? gpu_block_size : cpu_block_size;
    }

public:
    explicit DeviceConfig(const std::string& device) {
        m_device = device;
        m_block_size = get_block_size_by_device(device);
    }

    void set_kv_head_configs(const std::vector<KVHeadConfig>& kv_heads_config) {
        m_kv_heads_config = kv_heads_config;
        m_key_cache_shape.reserve(m_kv_heads_config.size());
        m_value_cache_shape.reserve(m_kv_heads_config.size());

        for (size_t layer_id = 0; layer_id < kv_heads_config.size(); layer_id++) {
            const KVHeadConfig& config = m_kv_heads_config[layer_id];

            m_value_cache_shape.push_back(ov::PartialShape{ov::Dimension::dynamic(),
                                                           ov::Dimension(config.num_v_heads),
                                                           ov::Dimension(m_block_size),
                                                           ov::Dimension(config.v_head_size)});

            if (m_device.find("CPU") != std::string::npos) {
                m_key_cache_shape.push_back(ov::PartialShape{ov::Dimension::dynamic(),
                                                             ov::Dimension(config.num_k_heads),
                                                             ov::Dimension(m_block_size),
                                                             ov::Dimension(config.k_head_size)});
            } else if (m_device.find("GPU") != std::string::npos) {
                // Update key shape, as the key's shape is different from the value's shape
                m_key_cache_shape.push_back(ov::PartialShape{ov::Dimension::dynamic(),
                                                             ov::Dimension(config.num_k_heads),
                                                             ov::Dimension(config.k_head_size),
                                                             ov::Dimension(m_block_size)});
            }
        }
    }

    std::string get_device() const {
        return m_device;
    }

    ov::PartialShape get_key_cache_shape(size_t id) const {
        OPENVINO_ASSERT(m_key_cache_shape.size());
        return m_key_cache_shape[id];
    }

    ov::PartialShape get_value_cache_shape(size_t id) const {
        OPENVINO_ASSERT(m_value_cache_shape.size());
        return m_value_cache_shape[id];
    }

    size_t get_k_head_size(size_t layer_id) const {
        return m_kv_heads_config[layer_id].k_head_size;
    }

    size_t get_block_size() const {
        return m_block_size;
    }
};

}
