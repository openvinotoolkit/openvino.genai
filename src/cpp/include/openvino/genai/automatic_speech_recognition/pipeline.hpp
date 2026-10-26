// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace ov::genai {

using AudioInputs = std::variant<std::vector<float>>;

/// Time-aligned text chunk — used for both segment-level and word-level timestamps.
/// For segments: text is the decoded segment, token_ids are the tokens in that segment.
/// For words: text is the individual word, token_ids are the tokens composing that word.
struct ASRDecodedResultChunk {
    float start_ts;  // start of chunk in seconds
    float end_ts;    // end of chunk in seconds (-1.0f if not predicted by model)
    std::string text;
    std::vector<int64_t> token_ids;
};

struct OPENVINO_GENAI_EXPORTS ASRDecodedResults {
    std::vector<std::string> texts;
    std::vector<float> scores;
    /**
     * @brief Language associated with each transcription.
     *
     * Each entry contains the detected language when the model supports language identification, or the requested
     * language when generation forces one. An entry is empty when no language was requested and the model does not
     * support language identification, as with Fun-ASR.
     */
    std::vector<std::string> languages;
    ASRPerfMetrics perf_metrics;

    std::optional<std::vector<std::vector<ASRDecodedResultChunk>>> chunks = std::nullopt;
    std::optional<std::vector<std::vector<ASRDecodedResultChunk>>> words = std::nullopt;

    operator std::string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    };

    operator std::vector<std::string>() const {
        return texts;
    };

    friend std::ostream& operator<<(std::ostream& os, const ASRDecodedResults& dr) {
        OPENVINO_ASSERT(dr.scores.size() == dr.texts.size(), "The number of scores and texts doesn't match.");
        if (dr.texts.empty()) {
            return os;
        }
        if (dr.texts.size() == 1) {
            os << dr.texts[0];
            return os;
        }
        for (size_t i = 0; i < dr.texts.size() - 1; ++i) {
            os << std::to_string(dr.scores[i]) << ": " << dr.texts[i] << '\n';
        }
        return os << std::to_string(dr.scores.back()) << ": " << dr.texts.back();
    }
};

class ASRPipelineImplBase;
class ASRForcedAligner;
class ASRStreamImplBase;

/// Receives transcript deltas. Returning STOP or CANCEL stops generation and discards not yet processed audio.
using ASRTextCallback = std::function<StreamingStatus(std::string)>;

/**
 * @brief Streaming audio transcription session.
 *
 * generate() runs inference once for the whole stream and consumes audio frames submitted by push_audio()
 * until end_audio() is called. A producer thread may call push_audio() and end_audio() while generate() runs
 * in a different thread. All methods are thread-safe. Frames are processed in the order push_audio() calls
 * complete. Every thread must return from push_audio(), end_audio() and generate() before the stream is
 * destroyed or moved.
 */
class OPENVINO_GENAI_EXPORTS ASRStream {
public:
    ASRStream(ASRStream&&) noexcept;
    ASRStream& operator=(ASRStream&&) noexcept;
    ~ASRStream();

    /**
     * @brief Enqueue normalized mono 16 kHz audio samples for generate() without running inference.
     *
     * Blocks while the bounded input queue is full until generate() consumes a frame. Throws after end_audio() or
     * once generate() has stopped: after completion, after the callback requested a stop, or after an error.
     */
    void push_audio(const std::vector<float>& audio_frame);

    /// Signal the end of audio input. generate() processes the queued frames and returns. May be called once.
    void end_audio();

    /**
     * @brief Run inference over all audio pushed to the stream and return the complete transcription.
     *
     * May be called once. Blocks until end_audio() is called and the queue is drained, or the callback requests
     * a stop. The callback is invoked on the calling thread with each non-empty transcript delta and must not wait
     * for push_audio() because a full queue isn't drained while the callback runs. Exceptions thrown by inference
     * or the callback propagate from generate() and stop the stream.
     */
    ASRDecodedResults generate(const ASRTextCallback& text_callback = nullptr);

private:
    explicit ASRStream(std::unique_ptr<ASRStreamImplBase> impl);

    std::unique_ptr<ASRStreamImplBase> m_impl;

    friend class ASRPipeline;
};

class OPENVINO_GENAI_EXPORTS ASRPipeline {
    std::unique_ptr<ASRPipelineImplBase> m_impl;

public:
    ASRPipeline(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ASRPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : ASRPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~ASRPipeline();

    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config = std::nullopt,
                               StreamerVariant streamer = std::monostate());

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ASRDecodedResults, Properties...> generate(const AudioInputs& audio_inputs,
                                                                              Properties&&... properties) {
        return generate(audio_inputs, AnyMap{std::forward<Properties>(properties)...});
    }

    ASRDecodedResults generate(const AudioInputs& audio_inputs, const ov::AnyMap& config_map);

    /// Create a stateful input stream when the selected ASR model supports incremental audio.
    ASRStream create_stream(const std::optional<ASRGenerationConfig>& generation_config = std::nullopt);

    Tokenizer get_tokenizer();
    ASRGenerationConfig get_generation_config() const;
    void set_generation_config(const ASRGenerationConfig& config);
};

/**
 * @brief Pre-constructed forced aligner used for word-level timestamps.
 *
 * Constructor-only property carrying an already-built ASRForcedAligner instance, used when word-level
 * timestamps are requested. Currently supported only for Qwen3-ASR.
 */
static constexpr ov::Property<std::shared_ptr<ASRForcedAligner>> forced_aligner{"forced_aligner"};

}  // namespace ov::genai
