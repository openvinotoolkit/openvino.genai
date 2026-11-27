// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "sampling/sampling_overrides.hpp"
#include "sequence_group.hpp"

namespace ov::genai {

/// @brief Per-request input of a step hook, passed through generate() to IStepHook::before_add_request();
/// opaque to the pipeline.
class StepRequestPayload {
public:
    virtual ~StepRequestPayload() = default;
};

/// @brief Model-specific work inside a continuous batching step, for models whose next input is not always the
/// embedding of a sampled token: LFM3-Omni feeds back speech frames.
class IStepHook {
public:
    /// @brief A model's end of a request; the pipeline resolves it against the sampler's end.
    struct Ending {
        /// @brief FINISHED, STOP or CANCEL.
        GenerationStatus status;
        GenerationFinishReason reason;
    };

    /// @brief What a step hook did with a request after sampling; all fields empty to go on with the sampled token.
    struct RequestOutcome {
        /// @brief Next input [1, 1, hidden_size] instead of the sampled token's embedding; required after a forced
        /// token unless the request ends.
        std::optional<ov::Tensor> next_embedding;
        /// @brief Position ids of that input, copied; computed by the inputs embedder when absent.
        std::optional<ov::Tensor> position_ids;
        /// @brief Ends the request in this step; must come without a next input.
        std::optional<Ending> ending;
        /// @brief Fails only this request; the other fields must be empty.
        std::exception_ptr error;
    };

    virtual ~IStepHook() = default;

    /// @brief Called by add_request() on the caller's thread, before the request is built. inputs are embeddings
    /// [1, n, hidden_size] or token ids; appended prompt rows must also extend prompt_ids.
    /// @param sampling_overrides Output: sampling rules of this request only.
    /// @param payload May be null.
    /// @return The state of a request the hook takes, else null.
    virtual std::unique_ptr<StepRequestState> before_add_request(
        ov::Tensor& inputs,
        std::optional<ov::Tensor>& prompt_ids,
        SamplingOverrides& sampling_overrides,
        const std::shared_ptr<StepRequestPayload>& payload) = 0;

    /// @brief Called once per removed request with state, on the step thread or under the queue lock; for FAILED no
    /// model or user callback may run. A throw fails this request, or the step for FAILED or a FINISHED/IGNORED handle.
    /// @param status FINISHED, IGNORED, STOP, CANCEL or FAILED.
    virtual void after_remove_request(StepRequestState& state, GenerationStatus status) = 0;

    /// @brief After forward, before sampling: a forced token per request, or none to sample it. requests have state,
    /// sample in this step and have exactly one sequence, get_sequences().front(), which may finish in sampling.
    virtual std::vector<std::optional<int64_t>> before_sampling(const std::vector<SequenceGroup::Ptr>& requests) = 0;

    /// @brief After sampling, before handles are notified: one outcome per request, in order.
    virtual std::vector<RequestOutcome> after_sampling(const std::vector<SequenceGroup::Ptr>& requests) = 0;
};

}  // namespace ov::genai
