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
    EmbeddingsModel(const std::filesystem::path& model_dir,
                    const float scale_emb,
                    const std::string& device,
                    const ov::AnyMap& properties);

    EmbeddingsModel(const std::string& model,
                    const ov::Tensor& weights,
                    const float scale_emb,
                    const std::string& device,
                    const ov::AnyMap& properties);

    EmbeddingsModel() = default;

    ov::Tensor infer(ov::Tensor input_idx);

private:
    void merge_postprocess(std::shared_ptr<ov::Model> model, float scale_emb) const;

    ov::InferRequest m_request;
};

} // namespace genai
} // namespace ov