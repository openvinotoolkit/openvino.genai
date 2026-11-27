// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "npu_generation.hpp"

#include <algorithm>

#include "automatic_speech_recognition/models/qwen3-asr/split/npu/generation.hpp"
#include "logger.hpp"

namespace ov::genai {

bool prepare_npu_vlm_generation(VLMModelType model_type,
                                size_t prompt_len,
                                GenerationConfig& config,
                                size_t max_prompt_len,
                                size_t kv_cache_capacity) {
    if (model_type != VLMModelType::QWEN3_ASR) {
        return false;
    }
    OPENVINO_ASSERT(kv_cache_capacity > 0, "Qwen3-ASR NPU KV-cache capacity must be positive.");
    return limit_qwen3_asr_npu_generation(prompt_len, config, max_prompt_len, kv_cache_capacity - 1);
}

void report_npu_vlm_generation_limit(bool capacity_limited, const std::vector<GenerationFinishReason>& finish_reasons) {
    if (capacity_limited && std::find(finish_reasons.begin(), finish_reasons.end(), GenerationFinishReason::LENGTH) !=
                                finish_reasons.end()) {
        GENAI_ERR("Qwen3-ASR NPU KV-cache capacity was reached; the returned transcription may be truncated. "
                  "Increase MAX_PROMPT_LEN or MIN_RESPONSE_LEN to allow longer output.");
    }
}

}  // namespace ov::genai
