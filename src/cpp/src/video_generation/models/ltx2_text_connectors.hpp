// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

class LTX2TextConnectors {
public:
    struct Output {
        ov::Tensor video_text_embedding;
        ov::Tensor audio_text_embedding;
        ov::Tensor connector_attention_mask;
    };

    explicit LTX2TextConnectors(const std::filesystem::path& root_dir);

    LTX2TextConnectors(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    LTX2TextConnectors(const LTX2TextConnectors&);

    std::shared_ptr<LTX2TextConnectors> clone();

    LTX2TextConnectors& reshape(int batch_size);

    LTX2TextConnectors& compile(const std::string& device, const ov::AnyMap& properties = {});

    Output infer(const ov::Tensor& text_encoder_hidden_states, const ov::Tensor& attention_mask);

private:
    std::shared_ptr<ov::Model> m_model;
    ov::InferRequest m_request;
};

}  // namespace ov::genai
