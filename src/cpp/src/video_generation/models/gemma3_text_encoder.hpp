// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

class Gemma3TextEncoder {
public:
    struct EncodeResult {
        ov::Tensor prompt_embeds;
        ov::Tensor attention_mask;
    };

    explicit Gemma3TextEncoder(const std::filesystem::path& root_dir);

    Gemma3TextEncoder(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    Gemma3TextEncoder(const Gemma3TextEncoder&);

    std::shared_ptr<Gemma3TextEncoder> clone();

    Gemma3TextEncoder& reshape(int batch_size, int max_sequence_length);

    Gemma3TextEncoder& compile(const std::string& device, const ov::AnyMap& properties = {});

    EncodeResult infer(const std::string& pos_prompt,
                       const std::string& neg_prompt,
                       bool do_classifier_free_guidance,
                       int max_sequence_length);

private:
    Tokenizer m_tokenizer;
    std::shared_ptr<ov::Model> m_model;
    ov::InferRequest m_request;
    std::vector<std::string> m_hidden_state_names;
};

}  // namespace ov::genai
