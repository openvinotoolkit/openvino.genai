// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

#include "openvino/core/any.hpp"
#include "openvino/genai/omni/streamer_base.hpp"
#include "openvino/genai/omni/text_source_base.hpp"
#include "openvino/genai/streamer_base.hpp"

namespace ov::genai {

/**
 * @brief Thinker -> talker bridge OmniPipeline creates for each generate() call with
 * `GenerationConfig::stream_text2speech`: the VLM writes decode steps in as an OmniStreamerBase,
 * the talker reads them out as an OmniTextSourceBase on its own thread.
 *
 * Steps are queued in arrival order and are not dropped or coalesced. The queue is unbounded, so a
 * talker slower than the thinker costs memory rather than stalling text generation — the same
 * memory the non-streaming path spends accumulating VLMDecodedResults::intermediate_hidden_states.
 *
 * The reader closes its own end with finish_reading() when it stops listening normally and with
 * fail_reading() when it fails. Both drop the queue and are idempotent, with fail_reading() taking
 * precedence. After finish_reading(), write() keeps returning RUNNING so text generation can finish;
 * after fail_reading(), it returns STOP. The failure itself never travels on the channel —
 * OmniPipeline carries it on the talker's std::future and rethrows it on the caller's thread.
 *
 * The writer closes with end() or abandon(), and the difference travels to the reader as
 * truncated(). abandon() also drops the queue: a talker that keeps consuming a backlog the thinker
 * never finished would spend a full inference on speech nobody will hear.
 *
 * Safe to use from two threads, one writing and one reading.
 */
class OmniChannel : public OmniStreamerBase, public OmniTextSourceBase {
public:
    /// @brief Queue one VLM decode step, or drop it once the reader has closed its end.
    /// @return RUNNING, or STOP once the reader has called fail_reading().
    StreamingStatus write(const ov::AnyMap& data) override;

    void end() override;

    void abandon() override;

    /// @return The oldest queued step, or nullopt once the write end closed and the queue is
    ///         drained, or once the read end was closed by finish_reading() or fail_reading().
    std::optional<ov::AnyMap> read() override;

    bool truncated() const override;

    /// @brief Close the read end after the talker finished normally; text generation continues.
    void finish_reading();

    /// @brief Close the read end after the talker failed; text generation stops at the next step.
    void fail_reading();

private:
    /// A normal finish cannot soften a prior failure.
    enum class ReaderState { OPEN, FINISHED, FAILED };

    void close_read(ReaderState state);

    std::deque<ov::AnyMap> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_finished = false;
    bool m_truncated = false;
    ReaderState m_reader_state = ReaderState::OPEN;
};

}  // namespace ov::genai
