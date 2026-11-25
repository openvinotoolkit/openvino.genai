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
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Per-request input of a step hook, passed through generate() to IStepHook::admit(); opaque to the pipeline.
class StepRequestPayload {
public:
    virtual ~StepRequestPayload() = default;
};

/// @brief Model-specific work inside a continuous batching step, for models whose next input is not always the
/// embedding of a sampled token: LFM3-Omni feeds back speech frames.
class IStepHook {
public:
    /// @brief A request as add_request() is about to queue it. Prompt rows appended to inputs must also extend
    /// prompt_ids; sampling_overrides is an output.
    struct Admission {
        const GenerationConfig& config;
        /// @brief Embeddings [1, n, hidden_size], or token ids for token inputs.
        ov::Tensor& inputs;
        std::optional<ov::Tensor>& prompt_ids;
        /// @brief Null for token inputs.
        InputsEmbedder* inputs_embedder;
        SamplingOverrides& sampling_overrides;
        /// @brief May be null.
        const std::shared_ptr<StepRequestPayload>& payload;
    };

    /// @brief How a request ended. error is set for a failure and null for a dropped request.
    struct Terminal {
        /// @brief FINISHED, IGNORED, STOP, CANCEL or FAILED.
        GenerationStatus status;
        std::exception_ptr error;
    };

    /// @brief A model's end of a request; the pipeline resolves it against the sampler's end.
    struct Ending {
        /// @brief FINISHED, STOP or CANCEL.
        GenerationStatus status;
        GenerationFinishReason reason;
    };

    /// @brief A request that samples in this step; the pipeline gives its one running sequence.
    struct Row {
        SequenceGroup::Ptr sequence_group;
        Sequence::Ptr sequence;
        StepRequestState& state;
    };

    /// @brief What a step hook did with a row after sampling; all fields empty to go on with the sampled token.
    struct RowOutcome {
        /// @brief Next input [1, 1, hidden_size] instead of the sampled token's embedding; required after a forced
        /// token unless the row ends.
        std::optional<ov::Tensor> next_embedding;
        /// @brief Position ids of that input, copied; computed by the inputs embedder when absent.
        std::optional<ov::Tensor> position_ids;
        /// @brief Ends the request in this step; must come without a next input.
        std::optional<Ending> ending;
        /// @brief Fails only this request; the other fields must be empty.
        std::exception_ptr error;
    };

    virtual ~IStepHook() = default;

    /// @brief Called by add_request() on the caller's thread.
    /// @return The state of a request the hook takes, else null.
    virtual std::unique_ptr<StepRequestState> admit(const Admission& admission) = 0;

    /// @brief Called once per queued admitted request, on the step thread or under the queue lock; for FAILED no model
    /// or user callback may run. An exception fails this request, or the step for FAILED or a FINISHED/IGNORED handle.
    virtual void on_terminal(StepRequestState& state, const Terminal& terminal) = 0;

    /// @brief After forward, before sampling: a forced token per row, or none to sample it.
    virtual std::vector<std::optional<int64_t>> select_tokens(const std::vector<Row>& rows) = 0;

    /// @brief After sampling, before handles are notified: one outcome per row, in row order.
    virtual std::vector<RowOutcome> after_sampling(const std::vector<Row>& rows) = 0;
};

}  // namespace ov::genai
