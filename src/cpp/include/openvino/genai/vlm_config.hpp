// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/processor_config.hpp"

// how to merge it with tokenizer config
// and ProcessorConfig
// No much sense to allow encoding prompt outside
// Sould I always apply chat template
namespace ov::genai {
class OPENVINO_GENAI_EXPORTS VLMConfig {
public:
    /// @brief Even though it's the size of embeddings returned by VisionEncoder, this value is in config.json and this it's in VLMConfig.
    size_t hidden_size = 2304;
    /// @brief multiply embeddings by it. Hardcoded throughout this impl
    float scale_emb = 12.0f;
    /// @brief the number of <unk> to insert into the prompt per image slice.
    size_t query_num = 64;
    VLMConfig() = default;
    explicit VLMConfig(const std::filesystem::path& config_path);
    VLMConfig(const VLMConfig&) = default;
};
/*
 * Utils that allow to use generate() and operator()() in the following way:
 * pipe.generate(input_ids, ov::genai::scale_resolution(448), ...)
 * pipe(input_ids, ov::genai::scale_resolution(448), ...)
*/
static constexpr ov::Property<size_t> hidden_size{"hidden_size"};
static constexpr ov::Property<size_t> scale_emb{"scale_emb"};
static constexpr ov::Property<size_t> query_num{"query_num"};
}  // namespace ov::genai
