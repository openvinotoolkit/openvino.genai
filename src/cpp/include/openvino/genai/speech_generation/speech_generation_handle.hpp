// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

// One incremental unit of audio produced for a single add_request() session.
struct SpeechGenerationOutput {
    // PCM float32 waveform samples produced since the previous read() call for this request.
    ov::Tensor waveform;
    GenerationFinishReason finish_reason = GenerationFinishReason::NONE;
};

class SpeechGenerationStream;

class OPENVINO_GENAI_EXPORTS SpeechGenerationHandleImpl {
    std::shared_ptr<SpeechGenerationStream> m_generation_stream;

public:
    explicit SpeechGenerationHandleImpl(std::shared_ptr<SpeechGenerationStream> generation_stream)
        : m_generation_stream(std::move(generation_stream)) {}

    ~SpeechGenerationHandleImpl();

    // There can be only one handle for a request.
    SpeechGenerationHandleImpl(const SpeechGenerationHandleImpl&) = delete;
    SpeechGenerationHandleImpl& operator=(const SpeechGenerationHandleImpl&) = delete;

    GenerationStatus get_status();

    bool can_read();

    bool is_stopped();

    bool is_cancelled();

    void stop(GenerationFinishReason finish_reason = GenerationFinishReason::STOP);

    void cancel();

    // Reads audio produced since the previous read() call for this request.
    SpeechGenerationOutput read();

    // Blocks until generation finishes, returning the fully concatenated waveform.
    SpeechGenerationOutput read_all();

    SpeechGenerationPerfMetrics get_perf_metrics() const;
};

using SpeechGenerationHandle = std::shared_ptr<SpeechGenerationHandleImpl>;

}  // namespace genai
}  // namespace ov
