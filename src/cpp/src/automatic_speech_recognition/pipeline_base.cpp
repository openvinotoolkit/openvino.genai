// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/pipeline_base.hpp"

#include <optional>
#include <set>
#include <string>

#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

namespace {

const std::set<std::string> allowed_asr_ctor_properties = {
    ov::genai::word_timestamps.name(),
    ov::genai::forced_aligner.name(),
};

}  // namespace

namespace ov::genai {

void erase_allowed_asr_ctor_properties(ov::AnyMap& properties) {
    for (const std::string& property_name : allowed_asr_ctor_properties) {
        properties.erase(property_name);
    }
}

ASRStreamImplBase::ASRStreamImplBase(const size_t max_queued_frames) : m_max_queued_frames{max_queued_frames} {
    OPENVINO_ASSERT(m_max_queued_frames > 0, "ASR stream queue capacity must be positive.");
}

void ASRStreamImplBase::push_audio(const std::vector<float>& audio_frame) {
    std::vector<float> frame = audio_frame;
    {
        std::unique_lock<std::mutex> lock{m_mutex};
        m_frame_popped.wait(lock, [this] {
            return m_queue.size() < m_max_queued_frames || m_audio_ended || m_state == GenerationState::STOPPED;
        });
        OPENVINO_ASSERT(!m_audio_ended, "Cannot push audio after ASRStream::end_audio().");
        OPENVINO_ASSERT(m_state != GenerationState::STOPPED, "Cannot push audio after ASRStream::generate() has stopped.");
        m_queue.push_back(std::move(frame));
    }
    m_frame_pushed.notify_one();
}

void ASRStreamImplBase::end_audio() {
    {
        const std::lock_guard<std::mutex> lock{m_mutex};
        OPENVINO_ASSERT(!m_audio_ended, "ASRStream::end_audio() may only be called once.");
        m_audio_ended = true;
    }
    m_frame_pushed.notify_all();
    m_frame_popped.notify_all();
}

ASRDecodedResults ASRStreamImplBase::generate(const ASRTextCallback& text_callback) {
    {
        const std::lock_guard<std::mutex> lock{m_mutex};
        OPENVINO_ASSERT(m_state == GenerationState::NOT_STARTED, "ASRStream::generate() may only be called once.");
        m_state = GenerationState::RUNNING;
    }
    std::string transcript;
    const auto emit = [&transcript, &text_callback](std::string delta) {
        transcript += delta;
        if (delta.empty() || !text_callback) {
            return StreamingStatus::RUNNING;
        }
        return text_callback(std::move(delta));
    };
    try {
        StreamingStatus status = StreamingStatus::RUNNING;
        while (status == StreamingStatus::RUNNING) {
            const std::optional<std::vector<float>> frame = pop_audio_frame();
            if (!frame) {
                emit(flush());
                break;
            }
            status = emit(process_audio(*frame));
        }
    } catch (...) {
        stop_generation();
        throw;
    }
    stop_generation();

    ASRDecodedResults results;
    results.texts.push_back(std::move(transcript));
    results.scores.push_back(0.0f);
    results.languages.emplace_back();
    return results;
}

std::optional<std::vector<float>> ASRStreamImplBase::pop_audio_frame() {
    std::vector<float> frame;
    {
        std::unique_lock<std::mutex> lock{m_mutex};
        m_frame_pushed.wait(lock, [this] {
            return !m_queue.empty() || m_audio_ended;
        });
        if (m_queue.empty()) {
            return std::nullopt;
        }
        frame = std::move(m_queue.front());
        m_queue.pop_front();
    }
    m_frame_popped.notify_one();
    return frame;
}

void ASRStreamImplBase::stop_generation() {
    {
        const std::lock_guard<std::mutex> lock{m_mutex};
        m_state = GenerationState::STOPPED;
        m_queue.clear();
    }
    m_frame_popped.notify_all();
}

}  // namespace ov::genai
