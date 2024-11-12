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
    ov::element::Type m_kv_cache_type;
    std::vector<ov::PartialShape> m_key_cache_shape, m_value_cache_shape;
    std::vector<KVHeadConfig> m_kv_heads_config;
    size_t m_num_decoder_layers = 0;
    size_t m_num_kv_blocks = 0, m_cache_size = 0; // KV cache sizes in either blocks or GBs
    size_t m_block_size = 0; // block size is per inference device 
    std::string m_device;

    size_t get_block_size_by_device(const std::string& device) const {
        const size_t cpu_block_size = 32, gpu_block_size = 16;
        const bool is_gpu = device.find("GPU") != std::string::npos;
        return is_gpu ? gpu_block_size : cpu_block_size;
    }

public:
    DeviceConfig(ov::Core& core, const SchedulerConfig& scheduling_config, const std::string& device, const ov::AnyMap& plugin_config = {}) {
        m_device = device;

        // keep information about blocsk
        m_block_size = get_block_size_by_device(device);

        if (m_device == "CPU") {
            auto inference_precision = core.get_property(device, ov::hint::inference_precision);
            m_kv_cache_type = inference_precision == ov::element::bf16 ? ov::element::bf16 : ov::element::f16;

            // if user sets precision hint, kv cache type should be changed
            const auto inference_precision_it = plugin_config.find(ov::hint::inference_precision.name());
            if (inference_precision_it != plugin_config.end()) {
                const auto inference_precision = inference_precision_it->second.as<ov::element::Type>();
                if (inference_precision == ov::element::f32) {
                    m_kv_cache_type = ov::element::f32;
                } else if (inference_precision == ov::element::f16) {
                    m_kv_cache_type = ov::element::f16;
                } else if (inference_precision == ov::element::bf16) {
                    m_kv_cache_type = ov::element::bf16;
                } else {
                    // use default f32
                    m_kv_cache_type = ov::element::f32;
                }
            }

            // if user sets ov::kv_cache_precision hint
            const auto kv_cache_precision_it = plugin_config.find(ov::hint::kv_cache_precision.name());
            if (kv_cache_precision_it != plugin_config.end()) {
                const auto kv_cache_precision = kv_cache_precision_it->second.as<ov::element::Type>();
                m_kv_cache_type = kv_cache_precision;
            }
        } else if (m_device.find("GPU") != std::string::npos) {
            auto inference_precision = core.get_property(device, ov::hint::inference_precision);
            m_kv_cache_type = inference_precision == ov::element::f16 ? ov::element::f16 : ov::element::f32;

            // if user sets precision hint, kv cache type should be changed
            const auto inference_precision_it = plugin_config.find(ov::hint::inference_precision.name());
            if (inference_precision_it != plugin_config.end()) {
                const auto inference_precision = inference_precision_it->second.as<ov::element::Type>();
                if (inference_precision == ov::element::f16) {
                    m_kv_cache_type = ov::element::f16;
                } else {
                    // use default f32
                    m_kv_cache_type = ov::element::f32;
                }
            }
        } else {
            OPENVINO_THROW(m_device, " is not supported by OpenVINO Continuous Batching");
        }

        if (scheduling_config.num_kv_blocks > 0) {
            m_num_kv_blocks = scheduling_config.num_kv_blocks;
        } else if (scheduling_config.cache_size > 0) {
            m_cache_size = scheduling_config.cache_size;
        }
    }

    void set_kv_head_configs(std::vector<KVHeadConfig> kv_heads_config) {
        m_kv_heads_config = kv_heads_config;
        m_num_decoder_layers = m_kv_heads_config.size();
        m_key_cache_shape.reserve(m_num_decoder_layers);
        m_value_cache_shape.reserve(m_num_decoder_layers);

        if (m_device == "CPU") {
            // Scale, zero point and quantized data will be stored together.
            // The layout for per token per head:
            // |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)|
            // so, we have to extend head_size by 8, which is sizeof(float)
            // for scale and sizeof(float) for zeropoint
            if (m_kv_cache_type == ov::element::u8) {
                for (size_t layer_id = 0; layer_id < m_num_decoder_layers; ++layer_id) {
                    m_kv_heads_config[layer_id].k_head_size += 8;
                    m_kv_heads_config[layer_id].v_head_size += 8;
                }
            }
        }

        if (m_num_kv_blocks == 0 && m_cache_size > 0) {
            size_t size_in_bytes = m_cache_size * 1024 * 1024 * 1024; // convert GBs to bytes
            m_num_kv_blocks = size_in_bytes / get_block_size_in_bytes();
        }

        for (size_t layer_id = 0; layer_id < m_num_decoder_layers; layer_id++) {
            const KVHeadConfig& config = m_kv_heads_config[layer_id];

            m_value_cache_shape.push_back(ov::PartialShape{ov::Dimension::dynamic(),
                                                           ov::Dimension(config.num_v_heads),
                                                           ov::Dimension(m_block_size),
                                                           ov::Dimension(config.v_head_size)});

            if (m_device.find("GPU") == std::string::npos) {
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

    ov::element::Type get_cache_precision() const {
        return m_kv_cache_type;
    }

    size_t get_num_layers() const {
        return m_num_decoder_layers;
    }

    ov::PartialShape get_key_cache_shape(size_t id) const {
        OPENVINO_ASSERT(m_key_cache_shape.size());
        return m_key_cache_shape[id];
    }

    size_t get_head_size() const {
        return m_head_size;
    }

    ov::PartialShape get_value_cache_shape(size_t id) const {
        OPENVINO_ASSERT(m_value_cache_shape.size());
        return m_value_cache_shape[id];
    }

    size_t get_num_kv_blocks() const {
        return m_num_kv_blocks;
    }

    size_t get_block_size() const {
        return m_block_size;
    }

    size_t get_block_size_in_bytes() const {
        size_t block_size_in_bytes = 0;
        for (size_t layer_id = 0; layer_id < m_num_decoder_layers; layer_id++) {
            const KVHeadConfig& config = m_kv_heads_config[layer_id];
            block_size_in_bytes += config.k_head_size * config.num_k_heads + config.v_head_size * config.num_v_heads;
        }
        block_size_in_bytes *= get_block_size() * get_cache_precision().size();
        return block_size_in_bytes;
    }
};
}
