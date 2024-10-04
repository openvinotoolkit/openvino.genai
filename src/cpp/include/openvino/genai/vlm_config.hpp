// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include <openvino/runtime/properties.hpp>
#include <filesystem>

namespace ov::genai {
/// @brief A Configuration class passed to VLMPipeline and used to
/// change VLMPipeline's behavior. Corresponds to config.json.
class OPENVINO_GENAI_EXPORTS VLMConfig {
public:
    /// @brief The size of a single embedding returned by a resampler.
    /// Used to initialize positional embeddings for resampler input.
    size_t hidden_size = 2304;
    /// @brief multiply embeddings by this value.
    float scale_emb = 1.0f;
    /// @brief the number of <unk> to insert into the prompt per image
    /// slice.
    size_t query_num = 64;
    /// @brief A string denoting start of image embeddings for LLM.
    std::string im_start = "<image>";
    /// @brief A string denoting end of image embeddings for LLM.
    std::string im_end = "</image>";
    /// @brief A string denoting start of image slices row embeddings
    /// for LLM.
    std::string slice_start = "<slice>";
    /// @brief A string denoting end of image slices row embeddings
    /// for LLM.
    std::string slice_end = "</slice>";
    /// @brief Start each image (not a slice) with <image_id>i</image_id>.
    /// i is a number.
    bool use_image_id = true;
    std::string im_id_start = "<image_id>",
        im_id_end = "</image_id>",
        unk = "<unk>";  // A placeholder for image embeddings in text.
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
