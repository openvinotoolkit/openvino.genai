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

class GemmaTextEncoder {
public:
    struct EncodeResult {
        ov::Tensor prompt_embeds;
        ov::Tensor attention_mask;
    };

    explicit GemmaTextEncoder(const std::filesystem::path& root_dir);

    GemmaTextEncoder(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    GemmaTextEncoder(const GemmaTextEncoder&);

    std::shared_ptr<GemmaTextEncoder> clone();

    GemmaTextEncoder& reshape(int batch_size, int max_sequence_length);

    GemmaTextEncoder& compile(const std::string& device, const ov::AnyMap& properties = {});

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
