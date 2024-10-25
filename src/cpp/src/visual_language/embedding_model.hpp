// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

#include "visual_language/vlm_config.hpp"

#include <openvino/openvino.hpp>
#include "visual_language/processor_config.hpp"
#include "visual_language/vlm_model_type.hpp"

namespace ov {
namespace genai {

class EmbeddingsModel {
public:
    explicit EmbeddingsModel(const std::filesystem::path& root_dir, const VLMConfig& vlm_config);

    EmbeddingsModel(const std::filesystem::path& root_dir,
                    const VLMConfig& vlm_config,
                    const std::string& device,
                    const ov::AnyMap& properties);

    template <typename... Properties>
    EmbeddingsModel(const std::filesystem::path& root_dir,
                    const VLMConfig& vlm_config,
                    const std::string& device,
                    Properties&&... properties)
                  
        : EmbeddingsModel(root_dir, vlm_config, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    EmbeddingsModel(const EmbeddingsModel&);

    EmbeddingsModel& compile(const std::string& device, const ov::AnyMap& properties);

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<EmbeddingsModel&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ov::Tensor infer(ov::Tensor input_idx);

private:
    void merge_postprocess() const;

    VLMConfig m_vlm_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

} // namespace genai
} // namespace ov