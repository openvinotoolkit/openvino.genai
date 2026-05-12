// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <variant>

#include "openvino/genai/streamer_base.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * @brief Base class for audio streamers. Inherit and implement write() and end()
 * to receive audio chunks during speech generation.
 *
 * Thread safety: write() and end() are called sequentially on the thread that
 * runs generate(). No concurrent calls are made.
 *
 * Lifecycle:
 *   1. write() is called zero or more times with audio chunks.
 *   2. end() is always called exactly once after the last write(), even if
 *      generation was stopped or cancelled early, or if an error occurred.
 *
 * Return values from write():
 *   - RUNNING  -- continue generating and streaming audio chunks.
 *   - STOP     -- stop generation gracefully. end() is still called.
 *                  Already-generated tokens are kept in the output.
 *   - CANCEL   -- cancel generation. end() is still called.
 *                  Generated tokens may be discarded depending on the pipeline.
 */
class OPENVINO_GENAI_EXPORTS AudioStreamerBase {
public:
    /// @brief Called with each audio chunk during speech generation.
    /// @param audio_chunk Waveform tensor [1, 1, N_samples] float32 PCM at 24kHz.
    ///        The tensor is valid only for the duration of this call; copy if needed.
    /// @return StreamingStatus to continue (RUNNING), stop (STOP), or cancel (CANCEL).
    virtual StreamingStatus write(ov::Tensor audio_chunk) = 0;

    /// @brief Called exactly once when speech generation ends.
    ///        Always called, even on early stop/cancel or error.
    virtual void end() = 0;

    virtual ~AudioStreamerBase() = default;
};

/// @brief Variant type for audio streaming callbacks.
/// - Lambda: std::function<StreamingStatus(ov::Tensor)> — called with each audio chunk
/// - AudioStreamerBase subclass via shared_ptr
/// - monostate: no streaming (batch mode, default)
using AudioStreamerVariant = std::variant<
    std::function<StreamingStatus(ov::Tensor)>,
    std::shared_ptr<AudioStreamerBase>,
    std::monostate>;

}  // namespace ov::genai
