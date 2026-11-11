// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Model-specific work inside a continuous batching step, for models whose next input is not always the
/// embedding of a sampled token: LFM3-Omni feeds back speech frames.
class IStepHook {
public:
    /// @brief A ready-made next input [1, 1, hidden_size] for the last generated token of sequence.
    struct GeneratedEmbedding {
        SequenceGroup::Ptr sequence_group;
        Sequence::Ptr sequence;
        ov::Tensor embedding;
    };

    virtual ~IStepHook() = default;

    /// @brief Called by add_request() for embedding inputs, before position ids are computed. May append prompt rows.
    virtual void on_add_request(uint64_t request_id,
                                ov::Tensor& inputs_embeds,
                                std::optional<ov::Tensor>& prompt_ids,
                                InputsEmbedder& inputs_embedder) = 0;

    /// @brief After forward, before sampling. May force the next token of sequences that need sampling.
    /// @param logits The forward output, laid out as the sampler reads it.
    virtual void before_sampling(const std::vector<SequenceGroup::Ptr>& requests, const ov::Tensor& logits) = 0;

    /// @brief After sampling, before handles are notified. May finish sequences: their ids go to
    /// sampler_output.m_dropped_sequences.
    /// @return Embeddings to use instead of the embedding of the sampled token.
    virtual std::vector<GeneratedEmbedding> after_sampling(const std::vector<SequenceGroup::Ptr>& requests,
                                                           SamplerOutput& sampler_output) = 0;

    /// @brief The request finished, was stopped or was cancelled, and is freed.
    virtual void on_request_freed(const SequenceGroup::Ptr& request) = 0;

    /// @brief The request is dropped after a failure; no callback may run.
    virtual void on_request_dropped(uint64_t request_id) = 0;
};

}  // namespace ov::genai
