// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/voice_activity_detector.hpp"

#include "automatic_speech_recognition/vad/silero_vad.hpp"

namespace ov::genai {

class VoiceActivityDetector::Impl {
public:
    Impl(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
        : vad(models_path, device, properties) {}

    silero_vad::SileroVAD vad;
};

VoiceActivityDetector::VoiceActivityDetector(const std::filesystem::path& models_path,
                                             const std::string& device,
                                             const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(models_path, device, properties)) {}

VoiceActivityDetector::~VoiceActivityDetector() = default;

std::vector<SpeechTimestamp> VoiceActivityDetector::get_speech_timestamps(const std::vector<float>& audio,
                                                                         const ov::AnyMap& properties) {
    const std::vector<SpeechSegment> segments = m_impl->vad.get_speech_segments(audio, properties);
    std::vector<SpeechTimestamp> timestamps;
    timestamps.reserve(segments.size());
    for (const SpeechSegment& segment : segments) {
        timestamps.push_back({static_cast<double>(segment.start_sample) / silero_vad::SAMPLE_RATE,
                              static_cast<double>(segment.end_sample) / silero_vad::SAMPLE_RATE});
    }
    return timestamps;
}

}  // namespace ov::genai
