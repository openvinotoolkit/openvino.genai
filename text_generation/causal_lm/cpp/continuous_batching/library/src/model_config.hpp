
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <memory>

#include "openvino/core/model.hpp"

class ModelConfig {
    std::size_t m_num_kv_heads = 0;
    std::size_t m_head_size = 0;
    std::size_t m_num_layers = 0;

public:
    explicit ModelConfig(std::shared_ptr<ov::Model> model) {
        const ov::op::util::VariableVector& variables = model->get_variables();
        OPENVINO_ASSERT(!variables.empty(), "Model is supposed to be stateful");

        // number of variables is 2 (K and V) multiplied by number of decoder layers
        m_num_layers = variables.size() >> 1;
        ov::PartialShape variable_shape = variables[0]->get_info().data_shape;
        OPENVINO_ASSERT(variable_shape.size() == 4, "Partial shape should have 3 dimensions, got ", variable_shape);
        m_num_kv_heads = variable_shape[1].get_length();
        m_head_size = variable_shape[3].get_length();

        std::cout << "Auto-extracted model parameters: " << std::endl;
        std::cout << "m_num_layers = " << m_num_layers << std::endl;
        std::cout << "m_num_kv_heads = " << m_num_kv_heads << std::endl;
        std::cout << "m_head_size = " << m_head_size << std::endl;
    }

    std::size_t get_num_kv_heads() const {
        return m_num_kv_heads;
    }

    std::size_t get_head_size() const {
        return m_head_size;
    }

    std::size_t get_num_layers() const {
        return m_num_layers;
    }
};
