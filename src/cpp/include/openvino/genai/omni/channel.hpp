// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include "openvino/core/any.hpp"
#include "openvino/genai/omni/streamer_base.hpp"
#include "openvino/genai/omni/text_source_base.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

/**
 * @brief Built-in thinker -> talker bridge: the VLM writes decode steps in, the talker reads them
 * out, so speech generation runs while text generation is still going.
 *
 * OmniPipeline creates one per generate() call when `GenerationConfig::text2audio_stream` is set
 * and hands the same object to both stages — as an OmniStreamerBase to the VLM, and as an
 * OmniTextSourceBase to the talker. Construct one directly only when driving the two stages
 * yourself.
 *
 * Steps are queued in arrival order and are not dropped or coalesced: what the talker reads is
 * exactly what the thinker produced, in order. The queue is unbounded, so a talker slower than the
 * thinker costs memory rather than stalling text generation — the same memory the non-streaming
 * path spends accumulating VLMDecodedResults::intermediate_hidden_states.
 *
 * write() always returns RUNNING: a channel never asks the thinker to stop. Cancellation stays
 * with the caller's own StreamerVariant.
 *
 * Thread safety: unlike bare OmniStreamerBase implementations, this class is safe to use from two
 * threads — one writing, one reading — which is the point of the bridge. Multiple concurrent
 * writers are not supported (the VLM decode loop is single-threaded).
 *
 * @note This is a preview API and is subject to change.
 */
class OPENVINO_GENAI_EXPORTS OmniChannel : public OmniStreamerBase, public OmniTextSourceBase {
public:
    OmniChannel();

    ~OmniChannel() override;

    /// @brief Queue one VLM decode step. Keys are documented on OmniStreamerBase; the payload is
    ///        stored as-is, so ov::Tensor values are kept as ref-counted handles, not deep-copied.
    /// @return Always RUNNING.
    StreamingStatus write(const ov::AnyMap& data) override;

    /// @brief Close the write end. Idempotent. Wakes any reader waiting on the channel so a VLM
    ///        stage that stopped early — or threw — can't leave the talker blocked forever.
    void end() override;

    /// @brief Take the oldest queued step, blocking until one is queued or the write end closes.
    /// @return The step, or nullopt once end() has been called and the queue is drained.
    std::optional<ov::AnyMap> read() override;

    /// @brief Take the oldest queued step if one is already there, without blocking.
    /// @return The step, or nullopt when the queue is momentarily empty — which, unlike read()'s
    ///         nullopt, does not mean the stream ended. Use for draining after the writer is known
    ///         to be done, or for polling alongside other work.
    std::optional<ov::AnyMap> try_read();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
