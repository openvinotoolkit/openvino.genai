// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "openvino/genai/generation_config.hpp"
#include "sampling/structured_output/structured_output_controller.hpp"

namespace ov::genai {
namespace jump_forward_validation {

inline bool is_enabled(const std::optional<StructuredOutputConfig>& config) {
    return config && config->enable_jump_forward;
}

inline bool backend_supports(const std::optional<StructuredOutputConfig>& config) {
    return StructuredOutputController::supports_jump_forward(config);
}

inline void validate_continuous_config(const GenerationConfig& config,
                                       bool token_input,
                                       bool backend_has_capability,
                                       bool speculative_pipeline = false) {
    OPENVINO_ASSERT(token_input,
                    "Jump-forward decoding is supported only for token-input continuous batching; "
                    "embedding and VLM inputs are not supported.");
    OPENVINO_ASSERT(config.logprobs == 0,
                    "Jump-forward decoding does not support logprobs > 0.");
    OPENVINO_ASSERT(config.num_return_sequences == 1,
                    "Jump-forward decoding requires num_return_sequences == 1.");
    OPENVINO_ASSERT(!config.is_beam_search(),
                    "Jump-forward decoding does not support beam search.");
    OPENVINO_ASSERT(!config.is_prompt_lookup(),
                    "Jump-forward decoding does not support prompt lookup.");
    OPENVINO_ASSERT(!config.is_tree_search(),
                    "Jump-forward decoding does not support tree search or EAGLE generation.");
    OPENVINO_ASSERT(!speculative_pipeline,
                    "Jump-forward decoding does not support assisting or speculative generation.");
    OPENVINO_ASSERT(!config.is_assisting_generation(),
                    "Jump-forward decoding does not support assisting or speculative generation.");
    OPENVINO_ASSERT(config.stop_strings.empty(),
                    "Jump-forward decoding does not support non-empty stop_strings.");
    OPENVINO_ASSERT(backend_has_capability,
                    "The selected structured-output backend does not support jump-forward decoding.");
}

inline void validate_continuous(const GenerationConfig& config,
                                bool token_input,
                                bool speculative_pipeline = false) {
    if (!is_enabled(config.structured_output_config)) {
        return;
    }
    validate_continuous_config(
        config,
        token_input,
        backend_supports(config.structured_output_config),
        speculative_pipeline);
}

inline void validate_unsupported_pipeline(const GenerationConfig& config, const char* pipeline_name) {
    OPENVINO_ASSERT(!is_enabled(config.structured_output_config),
                    "Jump-forward decoding is not supported by the ",
                    pipeline_name,
                    " pipeline; use continuous batching.");
}

}  // namespace jump_forward_validation
}  // namespace ov::genai
