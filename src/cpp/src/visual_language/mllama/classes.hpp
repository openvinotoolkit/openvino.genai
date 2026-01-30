// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderMLlamma : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderMLlamma : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderMLlamma(const VLMConfig& vlm_config,
                        const std::filesystem::path& model_dir,
                        const std::string& device,
                        const ov::AnyMap device_config);

    InputsEmbedderMLlamma(const VLMConfig& vlm_config,
                        const ModelsMap& models_map,
                        const Tokenizer& tokenizer,
                        const std::filesystem::path& config_dir_path,
                        const std::string& device,
                        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {}) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;

};

}  // namespace ov::genai