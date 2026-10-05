// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>

#include "openvino/runtime/tensor.hpp"
#include "visual_language/vision_encoder.hpp"

namespace ov::genai {

class AudioEncoderGemma4 {
public:
    virtual ~AudioEncoderGemma4() = default;
    virtual ov::Tensor encode(const ov::Tensor& audio) = 0;

    static std::unique_ptr<AudioEncoderGemma4> create(const std::filesystem::path& model_dir,
                                                      VLMModelType model_type,
                                                      const std::string& device,
                                                      const ov::AnyMap& properties);

    static std::unique_ptr<AudioEncoderGemma4> create(const ModelsMap& models_map,
                                                      VLMModelType model_type,
                                                      const std::filesystem::path& config_dir_path,
                                                      const std::string& device,
                                                      const ov::AnyMap& properties);
};

}  // namespace ov::genai
