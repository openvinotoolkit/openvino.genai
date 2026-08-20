// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>

#include "circular_buffer_queue.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/gemma4/audio_feature_extractor.hpp"
#include "visual_language/vision_encoder.hpp"

namespace ov::genai {

class AudioEncoderGemma4 {
public:
    AudioEncoderGemma4(const std::filesystem::path& model_dir,
                       VLMModelType model_type,
                       const std::string& device,
                       const ov::AnyMap& properties);
    AudioEncoderGemma4(const ModelsMap& models_map,
                       VLMModelType model_type,
                       const std::filesystem::path& config_dir_path,
                       const std::string& device,
                       const ov::AnyMap& properties);

    bool is_available() const {
        return m_ireq_queue != nullptr;
    }

    ov::Tensor encode(const ov::Tensor& audio);

private:
    VLMModelType m_model_type;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue;
    std::optional<Gemma4AudioFeatureExtractor> m_feature_extractor;
    size_t m_unified_feature_size = 0;

    ov::Tensor prepare_unified_input(const ov::Tensor& audio) const;
    ov::Tensor encode_unified(const ov::Tensor& audio);
    ov::Tensor encode_e_models(const ov::Tensor& audio);
};

}  // namespace ov::genai
