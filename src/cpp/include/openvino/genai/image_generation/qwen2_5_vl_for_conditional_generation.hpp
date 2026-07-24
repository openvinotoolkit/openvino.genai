// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS Qwen2_5_VLForConditionalGeneration {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t hidden_size = 3584;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit Qwen2_5_VLForConditionalGeneration(const std::filesystem::path& root_dir);

    Qwen2_5_VLForConditionalGeneration(const std::filesystem::path& root_dir,
                         const std::string& device,
                         const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Qwen2_5_VLForConditionalGeneration(const std::filesystem::path& root_dir,
                         const std::string& device,
                         Properties&&... properties)
        : Qwen2_5_VLForConditionalGeneration(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    Qwen2_5_VLForConditionalGeneration(const Qwen2_5_VLForConditionalGeneration&);

    std::shared_ptr<Qwen2_5_VLForConditionalGeneration> clone();

    Qwen2_5_VLForConditionalGeneration& reshape(const int batch_size, const int max_sequence_length);

    Qwen2_5_VLForConditionalGeneration& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<Qwen2_5_VLForConditionalGeneration&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    // Returns pair of (prompt_embeds, encoder_attention_mask)
    std::pair<ov::Tensor, ov::Tensor> infer(const std::string& prompt, const int max_sequence_length);

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    const Config& get_config() const;

private:
    static constexpr size_t PROMPT_TEMPLATE_PREFIX_LENGTH = 34;
    static constexpr size_t TOKENIZER_MAX_LENGTH = 1024;
    static const std::string PROMPT_TEMPLATE;

    Config m_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    Tokenizer m_tokenizer;
};

}  // namespace genai
}  // namespace ov
