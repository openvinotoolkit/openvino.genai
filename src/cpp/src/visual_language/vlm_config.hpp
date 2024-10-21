// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include "visual_language/vlm_model_type.hpp"
#include <openvino/runtime/properties.hpp>

namespace ov::genai {
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
    
    /// @brief Default constructor.
    VLMConfig() = default;
    /// @brief Construct VLMConfig from values in json_path.
    /// Keys in the file must match the VLMConfig's members.
    /// @param json_path A path to a file to extract the values from.
    explicit VLMConfig(const std::string& config_path);
    /// @brief Default copy constructor.
    /// @param A config to copy from.
    VLMConfig(const VLMConfig&) = default;
};
}  // namespace ov::genai
