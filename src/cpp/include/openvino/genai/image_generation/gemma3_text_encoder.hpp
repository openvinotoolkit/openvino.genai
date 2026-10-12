// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS Gemma3TextEncoder {
public:
    explicit Gemma3TextEncoder(const std::filesystem::path& root_dir);

    Gemma3TextEncoder(const std::filesystem::path& root_dir,
                      const std::string& device,
                      const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Gemma3TextEncoder(const std::filesystem::path& root_dir,
                      const std::string& device,
                      Properties&&... properties)
        : Gemma3TextEncoder(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    Gemma3TextEncoder(const Gemma3TextEncoder&);

    std::shared_ptr<Gemma3TextEncoder> clone();

    Gemma3TextEncoder& reshape(int batch_size, int max_sequence_length);

    Gemma3TextEncoder& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<Gemma3TextEncoder&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /// Returns all hidden states stacked along the channel dimension: [batch, seq_len, hidden_size * num_hidden_states]
    ov::Tensor infer(const std::string& pos_prompt,
                     const std::string& neg_prompt,
                     bool do_classifier_free_guidance,
                     int max_sequence_length);

    ov::Tensor get_prompt_attention_mask() const;

private:
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    ov::Tensor m_prompt_attention_mask;
    std::vector<std::string> m_hidden_state_names;

    Tokenizer m_tokenizer;
};

} // namespace genai
} // namespace ov
