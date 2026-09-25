// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

/**
 * @brief Standalone forced aligner producing word-level timestamps for a transcript and its audio.
 *
 * Loaded from a dedicated forced-aligner model directory, independently of ASRPipeline.
 * Useful for generating timestamps for transcripts produced externally.
 *
 * @note A single instance may be reused for sequential align() calls, but is not thread-safe:
 * it must not be called concurrently from multiple threads, including when the same instance is
 * shared with an ASRPipeline. Use a separate instance per thread for concurrent alignment.
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
    ASRForcedAligner(const std::filesystem::path& models_path,
                     const std::string& device,
                     Properties&&... properties)
        : ASRForcedAligner(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~ASRForcedAligner();

    /**
     * @brief Aligns a transcript to audio and returns word-level chunks with timestamps relative to the audio start.
     * @param audio Raw mono float PCM at the model's sampling rate.
     * @param transcript Transcript text to align to the audio.
     * @param properties Backend-specific alignment properties, e.g. ov::genai::language("English"). Optional at the
     *                   generic API boundary; the current Qwen3 forced-aligner backend requires a language property.
     * @return Word-level chunks with timestamps in seconds relative to the audio start. Each returned
     *         ASRDecodedResultChunk has an empty token_ids: alignment units are normalized independently
     *         of ASR tokenization, so there is no reliable per-unit token mapping.
     */
    std::vector<ASRDecodedResultChunk> align(const std::vector<float>& audio,
                                             const std::string& transcript,
                                             const ov::AnyMap& properties = {});

    template <typename... Properties>
    util::EnableIfAllStringAny<std::vector<ASRDecodedResultChunk>, Properties...> align(
        const std::vector<float>& audio,
        const std::string& transcript,
        Properties&&... properties) {
        return align(audio, transcript, ov::AnyMap{std::forward<Properties>(properties)...});
    }

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
