// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline.hpp"

namespace ov::genai {

Qwen3ASRNPUVLMPipelineAdapter::Qwen3ASRNPUVLMPipelineAdapter(const std::filesystem::path& models_path,
                                                             const ov::AnyMap& properties)
    : Qwen3ASRVLMPipelineAdapter(models_path, "NPU", properties) {
    m_max_asr_input_seconds = 60;
}

void Qwen3ASRNPUVLMPipelineAdapter::validate_generation_config(const ASRGenerationConfig& config) const {
    OPENVINO_ASSERT(!config.is_beam_search(), "Qwen3-ASR NPU decoder does not support beam search.");
}

}  // namespace ov::genai
