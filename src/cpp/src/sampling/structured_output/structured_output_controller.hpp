// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <optional>
#include <vector>
#include <unordered_map>
#include <functional>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "sampling/logit_transformers.hpp"

namespace ov {
namespace genai {

class IStructuredOutputImpl {
public:
    virtual ~IStructuredOutputImpl() = default;
    virtual std::shared_ptr<ov::genai::LogitTransformers::ILogitTransformer>
        get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters) = 0;
};

class StructuredOutputController {
public:
    using BackendFactory = std::function<std::unique_ptr<ov::genai::IStructuredOutputImpl>(
        const ov::genai::Tokenizer&, std::optional<int>)>;

    StructuredOutputController(const ov::genai::Tokenizer& tokenizer,
                              std::optional<int> vocab_size=std::nullopt);


    std::shared_ptr<ov::genai::LogitTransformers::ILogitTransformer>
        get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters);

    static void register_backend(const std::string& name, BackendFactory factory);
    static void set_default_backend(const std::string& name);

private:
    std::unordered_map<std::string, std::unique_ptr<IStructuredOutputImpl>> m_impls;
    const ov::genai::Tokenizer& m_tokenizer;
    std::optional<int> m_vocab_size;

    static std::unordered_map<std::string, BackendFactory>& get_backend_registry();
    static std::string& get_default_backend_name();
};

} // namespace genai
} // namespace ov
