// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace ov {
namespace genai {

/**
 * @brief Structure to keep speech generation config parameters.
 */
class OPENVINO_GENAI_EXPORTS SpeechGenerationConfig : public GenerationConfig {
public:
    SpeechGenerationConfig();
    explicit SpeechGenerationConfig(const std::filesystem::path& json_path);

    // Padding token id.
    //int64_t pad_token_id = 1;

    // Reduction factor
    int64_t reduction_factor = 2;

    // Number of Mel filterbank channels
    uint64_t num_mel_bins = 80;

    // Minimum ratio of output length to input text length; prevents output that's too short
    float minlenratio = 0.0;

    // Maximum ratio of output length to input text length; prevents excessively long outputs
    float maxlenratio = 20.0;

    // Probability threshold for stopping decoding; when output probability exceeds above this, generation will stop
    float threshold = 0.5;

    void update_generation_config(const ov::AnyMap& config_map = {});

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief checks that are no conflicting parameters.
    /// @throws Exception if config is invalid.
    void validate() const;
};

//static constexpr ov::Property<int64_t> pad_token_id{"pad_token_id"};
static constexpr ov::Property<int64_t> reduction_factor{"reduction_factor"};
static constexpr ov::Property<uint64_t> num_mel_bins{"num_mel_bins"};
static constexpr ov::Property<float> minlenratio{"minlenratio"};
static constexpr ov::Property<float> maxlenratio{"maxlenratio"};
static constexpr ov::Property<float> threshold{"threshold"};

}  // namespace genai
}  // namespace ov
