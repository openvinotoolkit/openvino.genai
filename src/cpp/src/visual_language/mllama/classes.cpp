// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/mllama/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

EncodedImage VisionEncoderMLlamma::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    return {};
}

ov::Tensor InputsEmbedderMLlamma::get_inputs_embeds(const std::string& unified_prompt,
                                                  const std::vector<ov::genai::EncodedImage>& images,
                                                  ov::genai::VLMPerfMetrics& metrics,
                                                  bool recalculate_merged_embeddings,
                                                  const std::vector<size_t>& images_sequence) {
    return {};
}

NormalizedPrompt InputsEmbedderMLlamma::normalize_prompt(const std::string& prompt,
                                                       size_t base_id,
                                                       const std::vector<EncodedImage>& images) const {
    return {};
}

InputsEmbedderMLlamma::InputsEmbedderMLlamma(const VLMConfig& vlm_config,
                                             const std::filesystem::path& model_dir,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}

InputsEmbedderMLlamma::InputsEmbedderMLlamma(const VLMConfig& vlm_config,
                                             const ModelsMap& models_map,
                                             const Tokenizer& tokenizer,
                                             const std::filesystem::path& config_dir_path,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

}  // namespace ov::genai