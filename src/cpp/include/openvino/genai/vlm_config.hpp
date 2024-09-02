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
    /// @brief The size of a single embedding returned by VisionEncoder.
    /// Used to initialize m_pos_embed_cache.
    size_t hidden_size = 2304;
    /// @brief multiply embeddings by this value.
    float scale_emb = 12.0f;
    /// @brief the number of <unk> to insert into the prompt per image
    /// slice.
    size_t query_num = 64;
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
/*
 * Utils that allow to use generate() and operator()() in the following way:
 * pipe.generate(input_ids, ov::genai::scale_resolution(448), ...)
 * pipe(input_ids, ov::genai::scale_resolution(448), ...)
*/
static constexpr ov::Property<VLMConfig> vlm_config{"vlm_config"};
static constexpr ov::Property<size_t> hidden_size{"hidden_size"};
static constexpr ov::Property<size_t> scale_emb{"scale_emb"};
static constexpr ov::Property<size_t> query_num{"query_num"};
}  // namespace ov::genai
