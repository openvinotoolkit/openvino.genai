// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <openvino/runtime/tensor.hpp>

namespace ov::genai {

// Describes one sequence that produced a newly sampled token in the current ContinuousBatching step.
struct StepSampledSequence {
    uint64_t request_id = 0;
    uint64_t sequence_id = 0;
    int64_t sampled_token_id = -1;
    // 0-based index of this sampled token within the sequence's generated tokens.
    size_t generated_index = 0;
    size_t prompt_len = 0;
    // Last token row of the model's exported `last_hidden_state`, shaped [1, 1, H] (f32).
    // Empty if hidden state export is disabled or unavailable.
    ov::Tensor last_hidden_state;
    // True if the sampler already marked this sequence finished (e.g. EOS or max_new_tokens).
    bool is_finished = false;
};

// Next-step embedding feedback produced by a PostSampleEmbedCallback for one sampled sequence.
struct StepNextEmbed {
    // Next talker/LM input embedding, shaped [1, 1, H] (f32). Ignored when the sequence finished
    // or when `stop_sequence` is true.
    ov::Tensor next_embed;
    // Optional next position_ids tensor (e.g. [3, 1, 1] for 3-row M-RoPE or [1, 1] for 1D RoPE).
    // When empty, ContinuousBatchingImpl synthesizes the sequential position id automatically.
    ov::Tensor next_position_ids;
    // Request early termination of this sequence (e.g. AudioStreamer returned STOP/CANCEL).
    bool stop_sequence = false;
};

// Batched per-step callback invoked by ContinuousBatchingImpl after sampling when running in
// ModelInputType::EMBEDDINGS mode without a VLM InputsEmbedder (e.g. Qwen3-TTS talker).
using PostSampleEmbedCallback =
    std::function<std::vector<StepNextEmbed>(const std::vector<StepSampledSequence>&)>;

}  // namespace ov::genai
