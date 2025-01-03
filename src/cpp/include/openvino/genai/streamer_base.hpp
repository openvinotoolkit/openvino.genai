// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"
#include <variant>

namespace ov {
namespace genai {

enum class CallbacWorkStatus {
    UNDEF = 0, // Streaming is not run
    RUNNING = 1, // Continue to run of inference
    STOP = 2, // Stop generate, keep hitory as is, KV state include last prompt and generated tokens at the end
    CANCEL = 3 // Stop generate, drop last prompt and all generated tokens from history, KV state include history exept last step
};

using CallbackTypeVariant = std::variant<bool, CallbacWorkStatus>;

/**
 * @brief base class for streamers. In order to use inherit from from this class and implement put, and methods
 *
 * @param m_tokenizer tokenizer
 */
class OPENVINO_GENAI_EXPORTS StreamerBase {
protected:
    CallbacWorkStatus streaming_finish_status = CallbacWorkStatus::UNDEF;
public:
    /// @brief put is called every time new token is decoded,
    /// @return bool flag to indicate whether generation should be stopped, if return true generation stops
    virtual bool put(int64_t token) = 0;

    /// @brief end is called at the end of generation. It can be used to flush cache if your own streamer has one
    virtual void end() = 0;

    virtual CallbacWorkStatus get_finish_streaming_reason() {
        return streaming_finish_status;
    }

    virtual ~StreamerBase();
};

}  // namespace genai
}  // namespace ov
