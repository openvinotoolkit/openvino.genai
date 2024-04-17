
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>

#include "openvino/runtime/tensor.hpp"

class Tokenizer {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    explicit Tokenizer(const std::string& models_path);

    ov::Tensor encode(std::string prompt);

    std::string decode(std::vector<int64_t> tokens);

    size_t get_eos_token_id() const;
};
