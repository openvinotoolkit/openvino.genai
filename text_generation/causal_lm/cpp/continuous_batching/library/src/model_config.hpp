
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <memory>

#include "openvino/core/model.hpp"

// for CPU we use block_size = 1 instead of vLLM's = 1
constexpr std::size_t BLOCK_SIZE = 16;

// TODO: extract from the model
constexpr int64_t SPECIAL_EOS_TOKEN = 2; // llm_model->get_rt_info()["eos_token_id"].as<int64_t>();

class ModelConfig {
    std::size_t m_num_kv_heads;
    std::size_t m_num_heads;
    std::size_t m_hidden_dims;
    std::size_t m_head_size;
    std::size_t m_num_layers;

public:
    explicit ModelConfig(std::shared_ptr<ov::Model> model) {
        // TODO: extract from model, while currently these values are hardcoded

        m_num_kv_heads = 12;
        m_num_heads = 12;
        m_hidden_dims = 768;
        m_head_size = m_hidden_dims / m_num_heads;
        m_num_layers = 12;
    }

    std::size_t get_num_kv_heads() const {
        return m_num_kv_heads;
    }

    std::size_t get_num_heads() const {
        return m_num_heads;
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
