// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"

namespace ov::genai {

class VisionEncoderQwen2_5_VL_CustomVIT : public VisionEncoderQwen2_5_VL {
public:
    using VisionEncoderQwen2_5_VL::VisionEncoderQwen2_5_VL;
};

class InputsEmbedderQwen2_5_VL_CustomVIT : public InputsEmbedderQwen2_5_VL {
public:
    InputsEmbedderQwen2_5_VL_CustomVIT(const VLMConfig& vlm_config,
                                       const std::filesystem::path& model_dir,
                                       const std::string& device,
                                       const ov::AnyMap device_config);

    InputsEmbedderQwen2_5_VL_CustomVIT(const VLMConfig& vlm_config,
                                       const ModelsMap& models_map,
                                       const Tokenizer& tokenizer,
                                       const std::filesystem::path& config_dir_path,
                                       const std::string& device,
                                       const ov::AnyMap device_config);

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

protected:
    void encode_with_imagepreprocess_cpp(const std::vector<ov::Tensor>& images,
                                         const ov::AnyMap& config_map,
                                         ov::Tensor& out_tensor,
                                         ov::genai::ImageSize& out_rsz_size,
                                         size_t frame_num,
                                         size_t frame_id) override;

    std::pair<ov::Tensor, ov::Tensor> run_video_image_embeddings_merger(
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& images_sequence,
        const std::vector<EncodedVideo>& videos,
        const std::vector<size_t>& videos_sequence) override;
};

} // namespace ov::genai
