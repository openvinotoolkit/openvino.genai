// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

class Text2SpeechStreamingSessionImpl;

/// Parameters of a Text2SpeechStreamingSession.
struct OPENVINO_GENAI_EXPORTS Text2SpeechStreamingConfig {
    /// Pushed text is held back until it is certain to tokenize the same as in the complete text: a trailing partial
    /// word is kept until whitespace follows it, or end() is called. A held-back run longer than this many characters
    /// (text without spaces, e.g. Chinese) is committed anyway, and the tokens at that boundary may then differ from
    /// the complete text.
    size_t max_holdback_chars = 32;
};

/// Audio produced by one Text2SpeechStreamingSession::read().
struct OPENVINO_GENAI_EXPORTS Text2SpeechStreamingResult {
    /// Float32 PCM, shape [N], at `output_sample_rate`. Append-only across results: the audio of all results
    /// concatenated is the audio of the whole utterance.
    ov::Tensor speech;
    uint32_t output_sample_rate = 0;
    /// Index of the first sample of `speech` in the utterance.
    size_t start_sample = 0;
    /// True for the last result, after end(). `speech` may be empty.
    bool is_final = false;
    /// Set on the final result: STOP when the model ended the utterance, LENGTH when max_new_tokens was reached.
    GenerationFinishReason finish_reason = GenerationFinishReason::NONE;
    /// Metrics of this read().
    SpeechGenerationPerfMetrics perf_metrics;
};

/**
 * @brief Speech synthesis of text that arrives in pieces, for example an LLM reply while it is being generated.
 *
 * push() only buffers text. read() returns the next audio that is ready and runs the inference that produces it on
 * the calling thread; it never waits for text. end() closes the input; the following read() calls drain the rest, and
 * the last result has is_final == true.
 *
 * With the same seed and the Qwen3-TTS streaming prompt layout (SpeechGenerationConfig::non_streaming_mode = false,
 * which a session always uses), the audio equals what Text2SpeechPipeline::generate() produces for the complete text,
 * however the text is split into push() calls, as long as the pieces end at word boundaries.
 *
 * Threading: push() and end() may be called from one producer thread while read() is called from one consumer thread.
 * The session co-owns the pipeline implementation and may outlive the Text2SpeechPipeline. The constructor throws if
 * the backend cannot serve another session: Qwen3-TTS supports one live session per pipeline, only on the SDPA
 * backend (ATTENTION_BACKEND = SDPA), and generate() cannot be called while a session is alive.
 */
class OPENVINO_GENAI_EXPORTS Text2SpeechStreamingSession {
public:
    /// @param pipeline the pipeline to speak with
    /// @param speaker_embedding optional speaker embedding, as for Text2SpeechPipeline::generate()
    /// @param generation_config optional configuration; the pipeline's configuration when not given.
    ///        `non_streaming_mode` is ignored: the text arrives over time, so a session always uses the streaming
    ///        prompt layout.
    /// @param streaming_config session parameters
    Text2SpeechStreamingSession(Text2SpeechPipeline& pipeline,
                                const ov::Tensor& speaker_embedding = ov::Tensor(),
                                const std::optional<SpeechGenerationConfig>& generation_config = std::nullopt,
                                const Text2SpeechStreamingConfig& streaming_config = {});

    ~Text2SpeechStreamingSession();
    /// Move-only: a session is one live utterance. A moved-from session throws on use.
    Text2SpeechStreamingSession(Text2SpeechStreamingSession&&) noexcept;
    Text2SpeechStreamingSession& operator=(Text2SpeechStreamingSession&&) noexcept;

    /// Append text. Never runs inference. Throws after end().
    void push(const std::string& text);

    /// Close the text input. May be called once.
    void end();

    /// Return the next audio, or std::nullopt if there is nothing to return yet (more text is needed and end() has not
    /// been called) or the final result was already returned. Blocks while the inference runs.
    std::optional<Text2SpeechStreamingResult> read();

private:
    std::unique_ptr<Text2SpeechStreamingSessionImpl> m_impl;
};

}  // namespace ov::genai
