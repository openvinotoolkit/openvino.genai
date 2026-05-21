// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <vector>
#include <map>

#include "openvino/core/any.hpp"
#include "openvino/genai/perf_metrics.hpp"

namespace ov {
namespace genai {

/**
 * @brief Timestamp chunk for word-level or segment-level timestamps.
 */
struct ASRTimestampChunk {
    float start_ts;   ///< Start time in seconds
    float end_ts;     ///< End time in seconds
    std::string text; ///< Transcribed text for this chunk
};

/**
 * @brief Generic result type for ASR pipeline outputs.
 * 
 * This replaces Whisper-specific WhisperDecodedResults with a more
 * generic structure that works for all ASR models (Whisper, Paraformer, etc.)
 */
struct ASRDecodedResults {
    /// Transcribed text(s) - one per batch item
    std::vector<std::string> texts;
    
    /// Optional confidence scores (if supported by the model)
    std::vector<float> scores;
    
    /// Optional word/segment-level timestamps
    std::optional<std::vector<std::vector<ASRTimestampChunk>>> timestamps;
    
    /// Performance metrics for the inference
    PerfMetrics perf_metrics;
    
    // Convenience accessors
    bool has_timestamps() const { return timestamps.has_value() && !timestamps->empty(); }
    bool has_scores() const { return !scores.empty(); }
    
    // Single-result convenience (for batch_size=1)
    const std::string& text() const { 
        static const std::string empty;
        return texts.empty() ? empty : texts[0]; 
    }
};

/**
 * @brief Generic generation configuration for ASR pipelines.
 * 
 * This provides common configuration options that work across different
 * ASR model architectures (Whisper, Paraformer, Wav2Vec2, etc.)
 */
struct ASRGenerationConfig {
    /// Maximum number of tokens to generate
    size_t max_new_tokens = 448;
    
    /// Language hint (ISO 639-1 code, e.g., "en", "zh", "de")
    /// Empty string means auto-detect (if supported by model)
    std::string language;
    
    /// Task: "transcribe" (default) or "translate" (to English)
    /// Note: Not all models support translation
    std::string task = "transcribe";
    
    /// Whether to return word/segment-level timestamps
    bool return_timestamps = false;
    
    // Sampling parameters (for autoregressive models like Whisper)
    float temperature = 1.0f;
    int top_k = 0;        ///< 0 = disabled
    float top_p = 1.0f;   ///< 1.0 = disabled
    
    // Beam search parameters
    size_t num_beams = 1;   ///< 1 = greedy decoding
    
    /// Whether to suppress blank tokens at the beginning
    bool suppress_blank = true;
    
    /// Decoder start token IDs (model-specific)
    std::vector<int64_t> decoder_start_token_ids;
    
    /// Tokens to suppress during generation
    std::vector<int64_t> suppress_tokens;
    
    /// Initial prompt/prefix for generation (if supported)
    std::optional<std::string> initial_prompt;
    
    /**
     * @brief Update config from a map of key-value pairs.
     * @param config_map Map of configuration options
     */
    void update(const ov::AnyMap& config_map);
};

}  // namespace genai
}  // namespace ov
