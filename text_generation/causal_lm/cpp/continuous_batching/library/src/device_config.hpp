// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/core.hpp"
#include "openvino/core/type/element_type.hpp"

#include "model_config.hpp"
#include "scheduler_config.hpp"

class DeviceConfig {
    ov::element::Type m_kv_cache_type;
    ov::Shape m_key_cache_shape;
    ov::Shape m_value_cache_shape;
    std::string m_device;

public:
    DeviceConfig(ov::Core& core, const SchedulerConfig& scheduling_config, const ModelConfig& model_config, const std::string& device) {
        m_device = device;
        
        if (m_device == "CPU") {
            auto inference_precision = core.get_property(device, ov::hint::inference_precision);
            m_kv_cache_type = inference_precision == ov::element::bf16 ? ov::element::bf16 : ov::element::f16;
            m_key_cache_shape = m_value_cache_shape = ov::Shape{scheduling_config.num_kv_blocks,
                                                                model_config.get_num_kv_heads(),
                                                                scheduling_config.block_size,
                                                                model_config.get_head_size()};
        } else if (m_device == "GPU") {
            OPENVINO_ASSERT("GPU is not currently supported. Please, remove this assert and fill configuration");
        } else {
            OPENVINO_THROW(m_device, " is not supported by OpenVINO Continuous Batching");
        }
    }

    std::string get_device() const {
        return m_device;
    }

    ov::element::Type get_cache_precision() const {
        return m_kv_cache_type;
    }

    ov::Shape get_key_cache_shape() const {
        return m_key_cache_shape;
    }

    ov::Shape get_value_cache_shape() const {
        return m_value_cache_shape;
    }
};
