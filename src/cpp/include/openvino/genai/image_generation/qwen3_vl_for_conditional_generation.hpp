// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace genai {

/// @brief Qwen3-VL language model producing the prompt embeddings consumed by Qwen-Image 2.1.
class OPENVINO_GENAI_EXPORTS Qwen3VLForConditionalGeneration {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t hidden_size = 4096;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir);

    Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                    const std::string& device,
                                    const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                    const std::string& device,
                                    Properties&&... properties)
        : Qwen3VLForConditionalGeneration(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    Qwen3VLForConditionalGeneration(const Qwen3VLForConditionalGeneration&);

    std::shared_ptr<Qwen3VLForConditionalGeneration> clone();

    Qwen3VLForConditionalGeneration& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<Qwen3VLForConditionalGeneration&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief Embeds a single prompt.
    /// @return Prompt embeddings of shape (1, prompt_sequence_length, hidden_size). The chat template prefix
    /// occupied by the system message is dropped, so the sequence length depends on the prompt.
    ov::Tensor infer(const std::string& prompt, int max_sequence_length);

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    const Config& get_config() const;

private:
    static const std::string SYSTEM_PREFIX;
    static const std::string PROMPT_TEMPLATE;

    Config m_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    Tokenizer m_tokenizer;
    size_t m_system_prefix_length;
};

}  // namespace genai
}  // namespace ov
