// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "openvino/genai/generation_config.hpp"

namespace ov::genai {

inline bool limit_qwen3_asr_npu_generation(size_t prompt_len,
                                           GenerationConfig& config,
                                           size_t max_prompt_len,
                                           size_t kv_cache_capacity) {
    OPENVINO_ASSERT(!config.is_beam_search(), "Qwen3-ASR NPU decoder does not support beam search.");
    OPENVINO_ASSERT(prompt_len > 0 && prompt_len <= max_prompt_len,
                    "Qwen3-ASR NPU prompt length ",
                    prompt_len,
                    " must be between 1 and MAX_PROMPT_LEN ",
                    max_prompt_len,
                    ".");
    OPENVINO_ASSERT(prompt_len <= kv_cache_capacity + 1,
                    "Qwen3-ASR NPU prompt exceeds the compiled KV-cache capacity.");
    const size_t available_new_tokens = kv_cache_capacity + 2 - prompt_len;
    OPENVINO_ASSERT(config.min_new_tokens <= available_new_tokens,
                    "Qwen3-ASR NPU min_new_tokens is ",
                    config.min_new_tokens,
                    ", but the compiled KV-cache can accommodate only ",
                    available_new_tokens,
                    " after a prompt of ",
                    prompt_len,
                    " tokens. Reduce min_new_tokens or increase MAX_PROMPT_LEN or MIN_RESPONSE_LEN.");
    const bool finite_limit = config.max_new_tokens != SIZE_MAX || config.max_length != SIZE_MAX;
    if (finite_limit) {
        const auto requested_new_tokens = config.get_max_new_tokens(prompt_len);
        OPENVINO_ASSERT(requested_new_tokens <= available_new_tokens,
                        "Qwen3-ASR NPU requested ",
                        requested_new_tokens,
                        " new tokens, but the compiled KV-cache can accommodate only ",
                        available_new_tokens,
                        " after a prompt of ",
                        prompt_len,
                        " tokens. Reduce max_new_tokens/max_length or increase MAX_PROMPT_LEN or MIN_RESPONSE_LEN.");
    } else {
        config.max_new_tokens = available_new_tokens;
    }
    return !finite_limit;
}

}  // namespace ov::genai
