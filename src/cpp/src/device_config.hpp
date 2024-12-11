// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/core.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

#include "openvino/genai/scheduler_config.hpp"

namespace ov::genai {
class DeviceConfig {
    ov::element::Type m_key_cache_type;
    ov::element::Type m_value_cache_type;
    size_t m_key_cache_group_size = 0;
    size_t m_value_cache_group_size = 0;
    ov::Shape m_key_cache_shape, m_value_cache_shape;
    ov::Shape::value_type m_num_kv_heads, m_head_size, m_num_decoder_layers;
    size_t m_num_kv_blocks = 0;
    size_t m_block_size = 0;
    size_t m_cache_size = 0;
    std::string m_device;

    size_t get_block_size_by_device(const std::string& device) const {
        const size_t cpu_block_size = 32;
        const size_t gpu_block_size = 16;

        bool is_gpu = device.find("GPU") != std::string::npos;

        return is_gpu ? gpu_block_size : cpu_block_size;
    }

public:
    DeviceConfig(ov::Core& core, const SchedulerConfig& scheduling_config, const std::string& device, const ov::AnyMap& plugin_config = {}) {
        m_device = device;

        // keep information about blocsk
        m_block_size = get_block_size_by_device(device);

        if (m_device == "CPU") {
            auto inference_precision = core.get_property(device, ov::hint::inference_precision);
            m_key_cache_type = m_value_cache_type = inference_precision == ov::element::bf16 ? ov::element::bf16 : ov::element::f16;

            // if user sets precision hint, kv cache type should be changed
            const auto inference_precision_it = plugin_config.find(ov::hint::inference_precision.name());
            if (inference_precision_it != plugin_config.end()) {
                const auto inference_precision = inference_precision_it->second.as<ov::element::Type>();
                if (inference_precision == ov::element::f32) {
                    m_key_cache_type = m_value_cache_type = ov::element::f32;
                } else if (inference_precision == ov::element::f16) {
                    m_key_cache_type = m_value_cache_type = ov::element::f16;
                } else if (inference_precision == ov::element::bf16) {
                    m_key_cache_type = m_value_cache_type = ov::element::bf16;
                } else {
                    // use default f32
                    m_key_cache_type = m_value_cache_type = ov::element::f32;
                }
            }
            const auto key_group_size_it = plugin_config.find(ov::hint::key_cache_group_size.name());
            if (key_group_size_it != plugin_config.end()) {
                m_key_cache_group_size = key_group_size_it->second.as<size_t>();
            } else {
                m_key_cache_group_size = 0;
            }

            const auto value_group_size_it = plugin_config.find(ov::hint::value_cache_group_size.name());
            if (value_group_size_it != plugin_config.end()) {
                m_value_cache_group_size = key_group_size_it->second.as<size_t>();
            } else {
                m_value_cache_group_size = 0;
            }
            // if user sets ov::kv_cache_precision hint
            const auto key_cache_precision_it = plugin_config.find(ov::hint::key_cache_precision.name());
            if (key_cache_precision_it != plugin_config.end()) {
                const auto key_cache_precision = key_cache_precision_it->second.as<ov::element::Type>();
                m_key_cache_type = key_cache_precision;
            }

            const auto value_cache_precision_it = plugin_config.find(ov::hint::value_cache_precision.name());
            if (value_cache_precision_it != plugin_config.end()) {
                const auto value_cache_precision = value_cache_precision_it->second.as<ov::element::Type>();
                m_value_cache_type = value_cache_precision;
            }
        } else if (m_device.find("GPU") != std::string::npos) {
            auto inference_precision = core.get_property(device, ov::hint::inference_precision);
            m_key_cache_type = m_value_cache_type = inference_precision == ov::element::f16 ? ov::element::f16 : ov::element::f32;

            // if user sets precision hint, kv cache type should be changed
            const auto inference_precision_it = plugin_config.find(ov::hint::inference_precision.name());
            if (inference_precision_it != plugin_config.end()) {
                const auto inference_precision = inference_precision_it->second.as<ov::element::Type>();
                if (inference_precision == ov::element::f16) {
                    m_key_cache_type = m_value_cache_type = ov::element::f16;
                } else {
                    // use default f32
                    m_key_cache_type = m_value_cache_type = ov::element::f32;
                }
            }
        } else {
            OPENVINO_THROW(m_device, " is not supported by OpenVINO Continuous Batching");
        }

        OPENVINO_ASSERT(scheduling_config.num_kv_blocks > 0 || scheduling_config.cache_size > 0, "num_kv_blocks or cache_size should be more than zero.");
        if (scheduling_config.num_kv_blocks > 0) {
            m_num_kv_blocks = scheduling_config.num_kv_blocks;
        }
        else {
            m_cache_size = scheduling_config.cache_size;
        }
    }

    void set_model_params(size_t num_kv_heads, size_t head_size, size_t num_decoder_layers) {
        m_num_kv_heads = num_kv_heads;
        m_head_size = head_size;
        m_num_decoder_layers = num_decoder_layers;
        size_t group_size = m_head_size;
        m_key_cache_group_size = m_key_cache_group_size ? m_key_cache_group_size : m_head_size;
        m_value_cache_group_size = m_value_cache_group_size ? m_value_cache_group_size : m_head_size;
        if (m_num_kv_blocks == 0) {
            OPENVINO_ASSERT(m_cache_size > 0, "num_kv_blocks or cache_size should be more than zero.");
            size_t size_in_bytes = m_cache_size * 1024 * 1024 * 1024;
            m_num_kv_blocks = size_in_bytes / (m_num_decoder_layers * 2 * m_num_kv_heads * m_block_size * m_head_size * m_key_cache_type.size());
        }
        if (m_device == "CPU") {
            // Scale, zero point and quantized data will be stored together.
            // The layout for per token per head:
            // |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)|
            // so, we have to extend head_size by 8, which is sizeof(float)
            // for scale and sizeof(float) for zeropoint
            auto init_cache_shape = [&](ov::element::Type precision) {
                size_t head_size = m_head_size;
                if (m_head_size % group_size != 0)
                    throw std::runtime_error("head_size cannot be divided by group_size");
                size_t group_num = m_head_size / group_size;
                if (precision == ov::element::u8) {
                    head_size += sizeof(float) * 2 * group_num;
                } else if (precision == ov::element::i4) {
                    head_size += sizeof(float) * group_num * 2;
                } else if (precision == ov::element::u4) {
                    head_size += sizeof(float) * 2 * group_num * 2;                  
                }
                return ov::Shape{m_num_kv_blocks, m_num_kv_heads, m_block_size, head_size};
            };
            m_key_cache_shape = init_cache_shape(m_key_cache_type);
            m_value_cache_shape = init_cache_shape(m_value_cache_type);
        }

        if (m_device.find("GPU") != std::string::npos) {
            // Update key shape, as the key's shape is different from the value's shape
            m_key_cache_shape = ov::Shape{m_num_kv_blocks,
                                          m_num_kv_heads,
                                          m_head_size,
                                          m_block_size};
        }
    }

    std::string get_device() const {
        return m_device;
    }

    ov::element::Type get_key_cache_precision() const {
        return m_key_cache_type;
    }

    ov::element::Type get_value_cache_precision() const {
        return m_value_cache_type;
    }

    size_t get_num_layers() const {
        return m_num_decoder_layers;
    }

    ov::Shape get_key_cache_shape() const {
        OPENVINO_ASSERT(!m_key_cache_shape.empty());
        return m_key_cache_shape;
    }

    ov::Shape get_value_cache_shape() const {
        OPENVINO_ASSERT(!m_value_cache_shape.empty());
        return m_value_cache_shape;
    }

    size_t get_num_kv_blocks() const {
        return m_num_kv_blocks;
    }

    size_t get_block_size() const {
        return m_block_size;
    }
};
}
