// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"
#include <variant>

namespace ov {
namespace genai {

enum class StreamingStatus {
    RUNNING = 0, // Continue to run of inference
    STOP = 1, // Stop generation, keep history as is, KV cache includes last request and generated tokens
    CANCEL = 2 // Stop generate, drop last prompt and all generated tokens from history, KV cache includes history but last step
};

/**
 * @brief base class for streamers. In order to use inherit from from this class and implement put, and methods
 */
class OPENVINO_GENAI_EXPORTS StreamerBase {
public:
    /// @brief put is called every time new token is decoded. Deprecated. Please, use write instead.
    /// @return bool flag to indicate whether generation should be stopped, if return true generation stops
    OPENVINO_DEPRECATED("Please, use `write()` instead of `put()`. Support will be removed in 2026.0.0 release.")
    virtual bool put(int64_t token) {
        OPENVINO_THROW("This method is deprecated and will be removed in 2026.0.0 release. Please, override write() instead.");
        return true;
    };

    /// @brief write is called every time new token is decoded
    /// @return StreamingStatus flag to indicate whether generation should be countinue to run or stopped or cancelled
    virtual StreamingStatus write(int64_t token) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return put(token) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
        OPENVINO_SUPPRESS_DEPRECATED_END
    };

    /// @brief write is called every time new vector of tokens is decoded, in case of assisting or prompt lookup decoding
    /// @return StreamingStatus flag to indicate whether generation should be countinue to run or stopped or cancelled
    virtual StreamingStatus write(const std::vector<int64_t>& tokens) {
        for (const auto token : tokens) {
            const auto status = write(token);
            if (status != StreamingStatus::RUNNING) {
                return status;
            }
        }

        return StreamingStatus::RUNNING;
    };

    /// @brief end is called at the end of generation. It can be used to flush cache if your own streamer has one
    virtual void end() = 0;

    virtual ~StreamerBase();
};

}  // namespace genai
}  // namespace ov
