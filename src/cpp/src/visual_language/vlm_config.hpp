// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include <openvino/runtime/properties.hpp>
#include <filesystem>

namespace ov::genai {

enum class VLMModelType {
    MINICPM,
    LLAVA,
    NANOLLAVA,
    LLAVA_NEXT,
    LLAVA_NEXT_VIDEO,
    INTERNVL_CHAT,
    PHI3_V,
    PHI4MM,
    QWEN2_VL,
    QWEN2_5_VL,
    GEMMA3,
};

/// @brief A Configuration class passed to VLMPipeline and used to
/// change VLMPipeline's behavior. Corresponds to config.json.
class VLMConfig {
public:
    /// @brief A enum denoting model type.
    VLMModelType model_type;
    /// @brief A size of a single embedding returned by a resampler.
    /// Used to initialize positional embeddings for resampler input.
    size_t hidden_size = 3584;
    /// @brief Multiply embeddings by this value.
    float scale_emb = 1.0f;
    /// @brief A number of embedding vectors representing an image
    /// slice.
    size_t query_num = 64;
    /// @brief A string token denoting start of image embeddings for an
    /// LLM.
    std::string im_start = "<image>";
    /// @brief A string token denoting end of image embeddings for an
    /// LLM.
    std::string im_end = "</image>";
    /// @brief A string token denoting start of image slices row
    /// embeddings for an LLM.
    std::string slice_start = "<slice>";
    /// @brief A string token denoting end of image slices row
    /// embeddings for LLM.
    std::string slice_end = "</slice>";
    /// @brief Start each image (not a slice) with
    /// <image_id>i</image_id>. i is a number.
    bool use_image_id = true;
    /// @brief A string token denoting start of image number region.
    std::string im_id_start = "<image_id>";
    /// @brief A string token denoting end of image number region.
    std::string im_id_end = "</image_id>";
    /// @brief A placeholder for image embeddings in text.
    std::string unk = "<unk>";

    // llava_next specific config params
    std::vector<float> image_newline;
    size_t vision_config_patch_size = 14;

    /// @brief A string token denoting start of image embeddings for InternVL2 model.
    std::string image_start_token = "<img>";
    /// @brief A placeholder for image embeddings in text for InternVL2 model.
    std::string image_context_token = "<IMG_CONTEXT>";
    /// @brief A string token denoting end of image embeddings for InternVL2 model.
    std::string image_end_token = "</img>";
    /// @brief phi3_v and phi4mm new line token embedding to separate images.
    std::vector<float> sub_GN = std::vector(4096, 0.0f);
    std::vector<float> glb_GN = std::vector(4096, 0.0f);
    
    /// @brief A string token denoting start of vision embeddings for Qwen2VL model.
    std::string vision_start_token = "<|vision_start|>";
    /// @brief A placeholder for image embeddings in text for Qwen2VL model.
    std::string image_pad_token = "<|image_pad|>";
    /// @brief A string token denoting end of vision embeddings for Qwen2VL model.
    std::string vision_end_token = "<|vision_end|>";
    
    /// @brief A size of a window for Qwen2.5VL model, used in window attention.
    size_t vision_config_window_size = 112;

    /// @brief A string token denoting start of vision embeddings for gemma3-4b-it model.
    std::string start_of_image = "<start_of_image>";
    /// @brief A placeholder for image embeddings in text for gemma3-4b-it model.
    std::string image_soft_token = "<image_soft_token>";
    /// @brief A string token denoting end of vision embeddings for gemma3-4b-it model.
    std::string end_of_image = "<end_of_image>";

    /// @brief A string token denoting start of video embeddings 
    std::string video_start = "<video>";

    /// @brief Default constructor.
    VLMConfig() = default;
    /// @brief Construct VLMConfig from values in json_path.
    /// Keys in the file must match the VLMConfig's members.
    /// @param json_path A path to a file to extract the values from.
    explicit VLMConfig(const std::filesystem::path& config_path);
    /// @brief Default copy constructor.
    /// @param A config to copy from.
    VLMConfig(const VLMConfig&) = default;
};
}  // namespace ov::genai
