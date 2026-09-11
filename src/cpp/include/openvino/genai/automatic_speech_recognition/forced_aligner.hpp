// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

/**
 * @brief Standalone forced aligner producing word-level timestamps for a transcript and its audio.
 *
 * Loaded from a dedicated forced-aligner model directory, independently of ASRPipeline. Useful for
 * generating timestamps for transcripts obtained elsewhere, including a continuous-batching ASR path.
 */
class OPENVINO_GENAI_EXPORTS ASRForcedAligner {
public:
    /**
     * @brief Constructs the forced aligner from a model directory.
     * @param models_path Forced-aligner model directory.
     * @param device Device to run inference on (e.g. "CPU", "GPU").
     * @param properties OpenVINO compile properties.
     */
    ASRForcedAligner(const std::filesystem::path& models_path,
                     const std::string& device,
                     const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ASRForcedAligner(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : ASRForcedAligner(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~ASRForcedAligner();

    /**
     * @brief Aligns a transcript to audio and returns word-level chunks with timestamps relative to the audio start.
     * @param audio Raw mono float PCM at the model's sampling rate.
     * @param transcript Transcript text to align to the audio.
     * @param language Language of the transcript. Optional at this generic boundary; the Qwen3 backend
     *                 requires it and throws if absent.
     */
    std::vector<ASRDecodedResultChunk> align(const std::vector<float>& audio,
                                             const std::string& transcript,
                                             std::optional<std::string> language = std::nullopt);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace genai
}  // namespace ov
