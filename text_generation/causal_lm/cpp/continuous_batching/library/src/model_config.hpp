
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <memory>

#include "openvino/core/model.hpp"

class ModelConfig {
    std::size_t m_num_kv_heads = 0;
    std::size_t m_num_heads = 0;
    std::size_t m_hidden_dims = 0;
    std::size_t m_head_size = 0;
    std::size_t m_num_layers = 0;

public:
    explicit ModelConfig(std::shared_ptr<ov::Model> model) {
        for (const auto& op : model->get_ops()) {
            if (op->get_type_name() == "PagedAttentionExtension") {
                ++m_num_layers;

                // hidden_size from 0 input
                ov::PartialShape activations_shape = op->get_input_partial_shape(0);
                OPENVINO_ASSERT(activations_shape.size() == 3 && activations_shape[2].is_static());
                m_hidden_dims = activations_shape[2].get_length();

                // num heads from PA -> reshape -> output_shape
                auto reshape = op->output(0).get_target_inputs().begin()->get_node();
                ov::PartialShape reshape_out_shape = reshape->get_output_partial_shape(0);
                OPENVINO_ASSERT(reshape_out_shape.size() == 4 && reshape_out_shape[2].is_static());
                m_num_heads = reshape_out_shape[2].get_length();
            }
        }

        // TODO: extract from original non-PA model
        // currently, just assume we don't use grouped query attention
        // For models like mistralai/Mistral-7B-Instruct-v0.2, please, explicitly override this value
        m_num_kv_heads = m_num_heads;

        // compute other parameters
        m_head_size = m_hidden_dims / m_num_heads;

        std::cout << "Auto-extracted model parameters: " << std::endl;
        std::cout << "m_num_layers = " << m_num_layers << std::endl;
        std::cout << "m_num_kv_heads = " << m_num_kv_heads << std::endl;
        std::cout << "m_num_heads = " << m_num_heads << std::endl;
        std::cout << "m_hidden_dims = " << m_hidden_dims << std::endl;
        std::cout << "m_head_size = " << m_head_size << std::endl;
    }

    std::size_t get_num_kv_heads() const {
        return m_num_kv_heads;
    }

    std::size_t get_hidden_dims() const {
        return m_hidden_dims;
    }

    std::size_t get_head_size() const {
        return m_head_size;
    }

    std::size_t get_num_layers() const {
        return m_num_layers;
    }
};
