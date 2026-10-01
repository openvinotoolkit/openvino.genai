// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/speech_generation_handle.hpp"

#include "speech_generation/speech_generation_stream.hpp"

using namespace ov::genai;

SpeechGenerationHandleImpl::~SpeechGenerationHandleImpl() {
    stop();
}

GenerationStatus SpeechGenerationHandleImpl::get_status() {
    return m_generation_stream->get_status();
}

bool SpeechGenerationHandleImpl::can_read() {
    return !is_cancelled() && !is_stopped() && m_generation_stream->can_read();
}

bool SpeechGenerationHandleImpl::is_stopped() {
    return get_status() == GenerationStatus::STOP;
}

bool SpeechGenerationHandleImpl::is_cancelled() {
    return get_status() == GenerationStatus::CANCEL;
}

void SpeechGenerationHandleImpl::stop(GenerationFinishReason finish_reason) {
    m_generation_stream->stop(finish_reason);
}

void SpeechGenerationHandleImpl::cancel() {
    m_generation_stream->cancel();
}

SpeechGenerationOutput SpeechGenerationHandleImpl::read() {
    OPENVINO_ASSERT(!(is_stopped() || is_cancelled()) || m_generation_stream->can_read(),
                    "SpeechGenerationHandle cannot be used after it is stopped / cancelled.");
    return m_generation_stream->read();
}

SpeechGenerationOutput SpeechGenerationHandleImpl::read_all() {
    OPENVINO_ASSERT(!(is_stopped() || is_cancelled()) || m_generation_stream->can_read(),
                    "SpeechGenerationHandle cannot be used after it is stopped / cancelled.");
    std::vector<float> samples;
    GenerationFinishReason finish_reason = GenerationFinishReason::NONE;
    while (get_status() == GenerationStatus::RUNNING || m_generation_stream->can_read()) {
        SpeechGenerationOutput chunk = read();
        const float* chunk_data = chunk.waveform.data<const float>();
        samples.insert(samples.end(), chunk_data, chunk_data + chunk.waveform.get_size());
        finish_reason = chunk.finish_reason;
    }

    SpeechGenerationOutput result;
    result.waveform = ov::Tensor(ov::element::f32, ov::Shape{samples.size()});
    std::copy(samples.begin(), samples.end(), result.waveform.data<float>());
    result.finish_reason = finish_reason;
    return result;
}

SpeechGenerationPerfMetrics SpeechGenerationHandleImpl::get_perf_metrics() const {
    return m_generation_stream->get_perf_metrics();
}
