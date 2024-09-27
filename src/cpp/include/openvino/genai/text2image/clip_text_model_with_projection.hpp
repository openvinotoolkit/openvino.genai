// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS CLIPTextModelWithProjection {
public:
    struct Config {
        size_t max_position_embeddings = 77;
        size_t hidden_size = 512;

        explicit Config(const std::string& config_path);
    };

    explicit CLIPTextModelWithProjection(const std::string root_dir);

    CLIPTextModelWithProjection(const std::string& root_dir,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    CLIPTextModelWithProjection(const std::string& root_dir,
                  const std::string& device,
                  Properties&&... properties)
        : CLIPTextModelWithProjection(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    CLIPTextModelWithProjection(const CLIPTextModelWithProjection&);

    const Config& get_config() const;

    CLIPTextModelWithProjection& reshape(int batch_size);

    CLIPTextModelWithProjection& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<CLIPTextModelWithProjection&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor infer(const std::string& pos_prompt, const std::string& neg_prompt, bool do_classifier_free_guidance);

    ov::Tensor get_output_tensor(const size_t idx);

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;

    Tokenizer m_clip_tokenizer;
};

} // namespace genai
} // namespace ov
