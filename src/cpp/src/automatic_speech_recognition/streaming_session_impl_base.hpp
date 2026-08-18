// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

namespace ov::genai {

// Abstract base for ASRStreamingSession's PIMPL — one concrete subclass per model family.
class ASRStreamingSession::Impl {
public:
    virtual std::optional<ASRPartialResult> push_chunk(const std::vector<float>& pcm16k) = 0;
    virtual ASRPartialResult finish() = 0;
    virtual ~Impl() = default;
};

}  // namespace ov::genai
