// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS LTX2TextConnectors {
public:
    struct OPENVINO_GENAI_EXPORTS Output {
        ov::Tensor video_text_embedding;
        ov::Tensor audio_text_embedding;
        ov::Tensor connector_attention_mask;
    };

    explicit LTX2TextConnectors(const std::filesystem::path& root_dir);

    LTX2TextConnectors(const std::filesystem::path& root_dir,
                       const std::string& device,
                       const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    LTX2TextConnectors(const std::filesystem::path& root_dir,
                       const std::string& device,
                       Properties&&... properties)
        : LTX2TextConnectors(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    LTX2TextConnectors(const LTX2TextConnectors&);

    LTX2TextConnectors clone();

    LTX2TextConnectors& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<LTX2TextConnectors&, Properties...> compile(const std::string& device,
                                                                               Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    LTX2TextConnectors& reshape(int batch_size);

    /// @brief Projects text encoder hidden states into per-modality embeddings for the transformer
    Output infer(const ov::Tensor& text_encoder_hidden_states, const ov::Tensor& attention_mask);

private:
    std::shared_ptr<ov::Model> m_model;
    ov::InferRequest m_request;
};

}  // namespace ov::genai
