// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include "openvino/core/any.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

/**
 * @brief Read end of the thinker -> talker bridge. The talker calls read() in a loop to pull the
 * VLM's decode steps as they are produced, instead of waiting for a finished VLMDecodedResults.
 *
 * This is the mirror of OmniStreamerBase: what a producer passed to write() is what read() returns,
 * with the same keys (see the `omni_stream` namespace in omni/streamer_base.hpp) and in the same
 * order. A class that inherits both interfaces is the bridge itself — OmniChannel is the built-in
 * one; OmniPipeline hands the same object to the VLM as a sink and to the talker as a source.
 *
 * read() blocks until a step is available or the stream ends, so the talker thread parks instead of
 * spinning. It must not block forever on a VLM that stopped early or threw: OmniStreamerBase::end()
 * is always called and must release the reader.
 *
 * @note This is a preview API and is subject to change.
 */
class OPENVINO_GENAI_EXPORTS OmniTextSourceBase {
public:
    /// @brief Take the next VLM step, blocking until one arrives or the stream ends.
    /// @return The step payload, or std::nullopt once the write end has called end() and every
    ///         queued step has been read. nullopt is final: further reads keep returning nullopt.
    virtual std::optional<ov::AnyMap> read() = 0;

    virtual ~OmniTextSourceBase();
};

}  // namespace ov::genai
