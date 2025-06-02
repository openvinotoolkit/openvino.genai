// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen2vl/classes.hpp"

namespace ov::genai {

class VisionEncoderQwen2_5_VL : public VisionEncoderQwen2VL {
public:
    using VisionEncoderQwen2VL::VisionEncoderQwen2VL;
};

class InputsEmbedderQwen2_5_VL : public InputsEmbedderQwen2VL {
public:
    InputsEmbedderQwen2_5_VL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderQwen2_5_VL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer, 
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

protected:
    ov::Tensor run_image_embeddings_merger(
        const std::vector<EncodedImage>& images, 
        const std::vector<size_t>& images_sequence) override;
};

} // namespace ov::genai
