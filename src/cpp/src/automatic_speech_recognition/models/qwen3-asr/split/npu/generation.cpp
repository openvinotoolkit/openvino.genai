// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "generation.hpp"

#include <algorithm>

#include "automatic_speech_recognition/models/qwen3-asr/split/inputs_embedder.hpp"
#include "logger.hpp"

namespace ov::genai {

InputsEmbedder::GenerationPolicy InputsEmbedderQwen3ASR::prepare_generation(size_t prompt_len,
                                                                            GenerationConfig& config,
                                                                            size_t max_prompt_len,
                                                                            size_t kv_cache_capacity) const {
    OPENVINO_ASSERT(kv_cache_capacity > 0, "Qwen3-ASR NPU KV-cache capacity must be positive.");
    // The shared VLM loop reserves one slot at the KV-cache boundary.
    return {true, limit_qwen3_asr_npu_generation(prompt_len, config, max_prompt_len, kv_cache_capacity - 1)};
}

void InputsEmbedderQwen3ASR::finish_generation(const InputsEmbedder::GenerationPolicy& policy,
                                               const std::vector<GenerationFinishReason>& finish_reasons) const {
    if (policy.capacity_limited &&
        std::find(finish_reasons.begin(), finish_reasons.end(), GenerationFinishReason::LENGTH) !=
            finish_reasons.end()) {
        GENAI_ERR("Qwen3-ASR NPU KV-cache capacity was reached; the returned transcription may be truncated. "
                  "Increase MAX_PROMPT_LEN or MIN_RESPONSE_LEN to allow longer output.");
    }
}

}  // namespace ov::genai
