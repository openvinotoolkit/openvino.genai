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
 * The reader closes its own end with detach() when it stops listening and with abort() when it
 * fails. Both drop the queue, which is what keeps a reader that is gone from turning the rest of
 * the generation into unbounded growth. They differ in what write() then tells the thinker:
 * detach() leaves it RUNNING so the text still finishes, while abort() returns STOP because
 * nobody will hear the rest. Until one of them is called write() returns RUNNING — a healthy
 * channel never asks the thinker to stop, and caller-driven cancellation stays with the caller's
 * own StreamerVariant.
 *
 * The writer closes with end() or abandon(), and the difference travels to the reader as
 * truncated(). abandon() also drops the queue: a talker that keeps consuming a backlog the thinker
 * never finished would spend a full inference on speech nobody will hear, and delay the thinker's
 * exception by exactly that long.
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
    ///        Once the reader has closed its end the payload is dropped instead of queued.
    /// @return RUNNING, or STOP once the reader has called abort().
    StreamingStatus write(const ov::AnyMap& data) override;

    /// @brief Close the write end. Idempotent. Wakes any reader waiting on the channel so a VLM
    ///        stage that stopped early — or threw — can't leave the talker blocked forever.
    void end() override;

    /// @brief Close the write end after the VLM stage threw, dropping whatever is still queued.
    ///        The next read() returns nullopt and truncated() then reports true.
    void abandon() override;

    /// @brief Take the oldest queued step, blocking until one is queued or the write end closes.
    /// @return The step, or nullopt once end() has been called and the queue is drained, or once
    ///         this end was closed by detach() or abort().
    std::optional<ov::AnyMap> read() override;

    /// @brief Whether the write end closed with abandon() rather than end().
    bool truncated() const override;

    /// @brief Close the read end: no more steps are wanted, and that is not an error. The thinker
    ///        keeps generating, so a talker that finished early — codec EOS, its own token budget,
    ///        a speech streamer that asked to stop — still leaves the caller with the full text.
    ///        Idempotent, and callable after abort(), which it does not undo.
    void detach();

    /// @brief Close the read end after a failure: the talker gave up, so the rest of the response
    ///        is pointless and write() starts returning STOP. Idempotent, and takes precedence
    ///        over detach(). The failure itself does not travel on the channel — OmniPipeline
    ///        carries it on the talker's std::future and rethrows it on the caller's thread.
    void abort();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
