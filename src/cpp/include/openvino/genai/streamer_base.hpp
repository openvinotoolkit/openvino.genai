// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/generation_handle.hpp"
#include <variant>

namespace ov {
namespace genai {

enum class StreamerRunningStatus {
    RUNNING = 0, // Continue to run of inference
    STOP = 1, // Stop generation, keep history as is, KV cache includes last request and generated tokens
    CANCEL = 2 // Stop generate, drop last prompt and all generated tokens from history, KV cache include history but last step
};

using CallbackTypeVariant = std::variant<bool, StreamerRunningStatus, std::monostate>;

/**
 * @brief base class for streamers. In order to use inherit from from this class and implement put, and methods
 *
 * @param m_tokenizer tokenizer
 */
class OPENVINO_GENAI_EXPORTS StreamerBase {
public:
    StreamerRunningStatus m_streaming_finish_status = StreamerRunningStatus::RUNNING;
    /// @brief put is called every time new token is decoded,
    /// @return bool flag to indicate whether generation should be stopped, if return true generation stops
    virtual bool put(int64_t token) = 0;

    /// @brief end is called at the end of generation. It can be used to flush cache if your own streamer has one
    virtual void end() = 0;

    StreamerRunningStatus get_streaming_status() {
        return m_streaming_finish_status;
    }

    virtual ~StreamerBase();
};

}  // namespace genai
}  // namespace ov
