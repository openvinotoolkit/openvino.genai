// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
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

class OPENVINO_GENAI_EXPORTS T5EncoderModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t max_sequence_length = 512;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit T5EncoderModel(const std::filesystem::path& root_dir);

    T5EncoderModel(const std::filesystem::path& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    T5EncoderModel(const std::filesystem::path& root_dir,
                  const std::string& device,
                  Properties&&... properties)
        : T5EncoderModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    T5EncoderModel(const T5EncoderModel&);

    const Config& get_config() const;

    T5EncoderModel& reshape(int batch_size);

    T5EncoderModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<T5EncoderModel&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor infer(const std::string& pos_prompt);

    ov::Tensor get_output_tensor(const size_t idx);

private:
    Config m_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;

    Tokenizer m_clip_tokenizer;
};

} // namespace genai
} // namespace ov
