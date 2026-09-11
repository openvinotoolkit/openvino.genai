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

    Qwen3ForcedAligner m_aligner;
};

ASRForcedAligner::ASRForcedAligner(const std::filesystem::path& models_path,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(models_path, device, properties)) {}

ASRForcedAligner::~ASRForcedAligner() = default;

std::vector<ASRDecodedResultChunk> ASRForcedAligner::align(const std::vector<float>& audio,
                                                           const std::string& transcript,
                                                           std::optional<std::string> language) {
    return m_impl->m_aligner.align(audio, transcript, std::move(language));
}

}  // namespace genai
}  // namespace ov
