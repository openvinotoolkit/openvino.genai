// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "automatic_speech_recognition/models/qwen3-asr/split/audio_encoder.hpp"

namespace ov::genai {

class AudioEncoderQwen3ASRNPU final : public AudioEncoderQwen3ASR {
public:
    AudioEncoderQwen3ASRNPU(const VLMConfig& config,
                            const std::shared_ptr<ov::Model>& audio_model,
                            const std::filesystem::path& config_dir,
                            const ov::AnyMap& properties);

private:
    EncodedAudio encode_audio(const ov::Tensor& audio) override;
};

}  // namespace ov::genai
