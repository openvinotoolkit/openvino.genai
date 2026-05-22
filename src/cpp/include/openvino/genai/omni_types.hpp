// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ov {
namespace genai {

class AudioStreamerBase {
public:
    virtual ov::genai::StreamingStatus write(const ov::Tensor& audio_chunk) = 0;
    virtual void end() = 0;
    virtual ~AudioStreamerBase() = default;
};

using AudioStreamerCallback = std::function<ov::genai::StreamingStatus(const ov::Tensor&)>;
using AudioStreamerVariant = std::variant<std::monostate, AudioStreamerCallback, std::shared_ptr<AudioStreamerBase>>;

struct AudioSegment {
    RawSpeechInput data;
    uint32_t sample_rate = 16000u;
};

struct OmniInput {
    std::optional<std::string> text;
    std::vector<ov::Tensor> images;
    std::vector<AudioSegment> audios;
    std::vector<ov::Tensor> video_frames;
    float video_fps = 1.0f;

    void validate() const {
        OPENVINO_ASSERT(text.has_value() || !images.empty() || !audios.empty() || !video_frames.empty(),
                        "OmniInput: at least one modality (text, images, audios, video_frames) must be provided.");
    }
};

enum class OmniOutputModality {
    TEXT,
    AUDIO,
    TEXT_AUDIO,
};

enum class AudioSynthesisMode {
    SEQUENTIAL,
    STREAMING_SENTENCE,
    STREAMING_TOKEN,
};

class OPENVINO_GENAI_EXPORTS OmniGenerationConfig : public GenerationConfig {
public:
    OmniGenerationConfig() = default;

    OmniOutputModality output_modality = OmniOutputModality::TEXT;
    AudioSynthesisMode audio_synthesis_mode = AudioSynthesisMode::SEQUENTIAL;

    std::optional<ov::Tensor> speaker_embedding;
    float speech_rate = 1.0f;

    bool return_timestamps = false;
    std::optional<std::string> language;
    std::optional<std::string> task;

    void validate() const;
};

struct OmniDecodedResults : public DecodedResults {
    std::optional<std::vector<ov::Tensor>> speeches;
};

}  // namespace genai
}  // namespace ov
