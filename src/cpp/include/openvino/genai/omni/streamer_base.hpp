// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * @brief Write end of the thinker -> talker bridge. The VLM calls write() once per decode step
 * with that step's tokens and thinker hidden states, so the talker can start speaking before
 * text generation finishes.
 *
 * Inherit and implement write() and end(). A typical implementation is a bounded queue that also
 * inherits OmniTextSourceBase, so the same object is the VLM's sink and the talker's source; pass
 * it to OmniPipeline, which hands it to both stages.
 *
 * The payload is an ov::AnyMap rather than a fixed signature so producers can add data without
 * breaking implementations. Keys are the contract; see the `omni_stream` namespace below for the
 * typed constants that name them. Implementations MUST ignore keys they don't recognize — a newer
 * VLM may pass more than an older streamer reads.
 *
 * Keys written by the Qwen3-Omni VLM stage:
 *   - `omni_stream::tokens` (std::vector<int64_t>) — token ids produced by this step. The prefill
 *     step carries every prompt token; each decode step carries exactly one.
 *   - `omni_stream::hidden_states` (std::vector<ov::Tensor>) — f32 thinker hidden states, each of
 *     shape [1, 1, hidden_size]. This is the same per-token layout as
 *     VLMDecodedResults::intermediate_hidden_states, so a step payload is literally a slice of what
 *     the batch path hands the talker. Present only when the backend collects hidden states
 *     (continuous batching with GenerationConfig::return_omni_outputs); check before reading.
 *     Tensors are ref-counted handles, so keeping them past the call is fine.
 *
 * The two keys are separate streams that a reader concatenates independently; align them by
 * absolute position across the whole stream, not within one payload. They line up position for
 * position, but a single write() may carry a different number of each, and the token stream ends
 * one entry longer: a hidden state is what predicted the *next* token, so the last generated token
 * (typically EOS) is sampled from the previous position's state and never gets one of its own.
 *
 * Writes are final: there is no retraction, and a reader may act on a token as soon as it arrives.
 * A token is written at the step it is sampled, which is before a stop string spanning it can be
 * matched, so a stop string rewound out of the returned text has already been written here. Readers
 * that must agree exactly with the text cannot use this interface; see
 * GenerationConfig::text2audio_stream.
 *
 * Thread safety: write() and end() are called sequentially from the thread running the VLM decode
 * loop; no concurrent calls are made from that side. When the talker consumes the stream on
 * another thread, the implementation owns the synchronization between write() and read().
 *
 * Lifecycle:
 *   1. write() is called zero or more times.
 *   2. end() is always called exactly once after the last write(), even if generation stopped or
 *      was cancelled early, or if an error occurred. Readers blocked in read() must be released by
 *      end(), otherwise the talker deadlocks on a failed VLM stage.
 *
 * Return values from write():
 *   - RUNNING -- continue generating.
 *   - STOP    -- stop generation gracefully; already-generated text is kept. end() is still called.
 *   - CANCEL  -- cancel generation; generated text may be dropped. end() is still called.
 *
 * @note This is a preview API and is subject to change.
 */
class OPENVINO_GENAI_EXPORTS OmniStreamerBase {
public:
    /// @brief Called once per VLM decode step with that step's data.
    /// @param data Step payload keyed as documented above. Unknown keys must be ignored.
    /// @return StreamingStatus to continue (RUNNING), stop (STOP), or cancel (CANCEL) the VLM.
    virtual StreamingStatus write(const ov::AnyMap& data) = 0;

    /// @brief Called exactly once when the VLM stage ends. Always called, even on early
    ///        stop/cancel or error. Must release any reader blocked in OmniTextSourceBase::read().
    virtual void end() = 0;

    virtual ~OmniStreamerBase();
};

/// @brief Typed names for the keys of the OmniStreamerBase::write() payload. Kept in their own
/// namespace because `tokens` / `hidden_states` are too generic to sit directly in ov::genai.
/// Use them on both ends so the two sides can't drift:
///     write(ov::AnyMap{omni_stream::tokens(ids), omni_stream::hidden_states(hs)});
///     const auto ids = data.at(omni_stream::tokens.name()).as<std::vector<int64_t>>();
namespace omni_stream {

/// @brief Token ids produced by this step: all prompt tokens on prefill, one token per decode step.
static constexpr ov::Property<std::vector<int64_t>> tokens{"tokens"};

/// @brief Thinker hidden states for this step: f32 [1, 1, hidden_size] tensors, co-indexed with
/// `tokens` by absolute position across the stream rather than within a single payload.
static constexpr ov::Property<std::vector<ov::Tensor>> hidden_states{"hidden_states"};

}  // namespace omni_stream

}  // namespace ov::genai
