// Copyright (C) 2023-2026 Intel Corporation
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

class OPENVINO_GENAI_EXPORTS Qwen3TextEncoder {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t hidden_size = 2560;
        size_t num_hidden_layers = 36;
        std::vector<size_t> hidden_states_layers = {9, 18, 27};

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit Qwen3TextEncoder(const std::filesystem::path& root_dir);

    Qwen3TextEncoder(const std::filesystem::path& root_dir,
                     const std::string& device,
                     const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Qwen3TextEncoder(const std::filesystem::path& root_dir,
                     const std::string& device,
                     Properties&&... properties)
        : Qwen3TextEncoder(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    Qwen3TextEncoder(const Qwen3TextEncoder&);

    std::shared_ptr<Qwen3TextEncoder> clone();

    Qwen3TextEncoder& reshape(const int batch_size, const int max_sequence_length);

    Qwen3TextEncoder& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<Qwen3TextEncoder&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor infer(const std::string& pos_prompt, const std::string& neg_prompt, const bool do_classifier_free_guidance, const int& max_sequence_length);

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    const Config& get_config() const;

private:
    Config m_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    Tokenizer m_tokenizer;
};

}  // namespace genai
}  // namespace ov
