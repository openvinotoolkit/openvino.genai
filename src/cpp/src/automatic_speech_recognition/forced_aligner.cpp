// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/forced_aligner.hpp"

#include "automatic_speech_recognition/models/qwen3-asr/qwen3_forced_aligner.hpp"

namespace ov {
namespace genai {

class ASRForcedAligner::Impl {
public:
    Impl(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
        : m_aligner(models_path, device, properties) {}

    std::vector<ASRDecodedResultChunk> align(const std::vector<float>& audio,
                                             const std::string& transcript,
                                             const ov::AnyMap& properties) {
        return m_aligner.align(audio, transcript, properties);
    }

private:
    Qwen3ForcedAligner m_aligner;
};

ASRForcedAligner::ASRForcedAligner(const std::filesystem::path& models_path,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(models_path, device, properties)) {}

ASRForcedAligner::~ASRForcedAligner() = default;

std::vector<ASRDecodedResultChunk> ASRForcedAligner::align(const std::vector<float>& audio,
                                                           const std::string& transcript,
                                                           const ov::AnyMap& properties) {
    return m_impl->align(audio, transcript, properties);
}

}  // namespace genai
}  // namespace ov
