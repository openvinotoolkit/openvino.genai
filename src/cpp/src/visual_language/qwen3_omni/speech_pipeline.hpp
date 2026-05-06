// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/audio_streamer_base.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

/// @brief Configuration for Qwen3-Omni speech generation.
struct Qwen3OmniSpeechConfig {
    int64_t codec_bos_id = 2149;
    int64_t codec_eos_token_id = 2150;
    int64_t codec_pad_id = 2148;
    int64_t codec_nothink_id = 2155;
    int64_t codec_think_bos_id = 2156;
    int64_t codec_think_eos_id = 2157;
    int64_t tts_bos_token_id = -1;
    int64_t tts_eos_token_id = -1;
    int64_t tts_pad_token_id = -1;
    int64_t im_start_token_id = -1;
    int64_t im_end_token_id = -1;
    int64_t system_token_id = -1;
    int64_t user_token_id = -1;
    int64_t assistant_token_id = -1;
    int64_t audio_token_id = -1;
    int64_t image_token_id = -1;
    int64_t video_token_id = -1;
    size_t num_code_groups = 16;
    size_t talker_hidden_size = 1024;
    size_t thinker_hidden_size = 2560;
    std::map<std::string, int64_t> speaker_ids;

    // Talker generation parameters (loaded from generation_config.json)
    float talker_temperature = 0.9f;
    size_t talker_top_k = 50;
    float talker_repetition_penalty = 1.0f;
    size_t talker_max_new_tokens = 4096;
    size_t talker_vocab_size = 3072;
    // Token IDs in [vocab_size-1024, vocab_size) except codec_eos are suppressed
    std::vector<int64_t> talker_suppress_tokens;

    /// @brief Initialize from VLMConfig.
    static Qwen3OmniSpeechConfig from_vlm_config(const VLMConfig& config);
};

/// @brief Speech generation pipeline for Qwen3-Omni.
/// Runs Talker + CodePredictor + Code2Wav after the Thinker completes text generation.
class Qwen3OmniSpeechPipeline {
public:
    Qwen3OmniSpeechPipeline(const std::filesystem::path& model_dir,
                            const VLMConfig& config,
                            const std::string& device,
                            const ov::AnyMap& properties);

    /// @brief Check if all speech models were loaded successfully.
    bool is_available() const {
        return m_talker_available;
    }

    /// @brief Generate speech from thinker generation results.
    /// @param full_token_ids All token IDs from the full sequence (prompt + generated).
    /// @param all_hidden_states Accumulated final hidden states [one tensor per step].
    /// @param all_intermediate_hidden_states Accumulated layer-14 hidden states [one tensor per step].
    /// @param audio_streamer Callback or AudioStreamerBase for streaming (monostate = batch mode).
    /// @param chunk_frames Number of codec frames per streaming chunk (0 = no streaming).
    /// @param speaker Speaker name (e.g., "f245", "m02").
    /// @param max_new_tokens Maximum number of codec tokens to generate.
    /// @return Waveform tensor [1, 1, audio_samples] or empty tensor on failure.
    ov::Tensor generate_speech(const std::vector<int64_t>& full_token_ids,
                               const std::vector<ov::Tensor>& all_hidden_states,
                               const std::vector<ov::Tensor>& all_intermediate_hidden_states,
                               const AudioStreamerVariant& audio_streamer = std::monostate{},
                               size_t chunk_frames = 0,
                               const std::string& speaker = "",
                               size_t max_new_tokens = 4096);

private:
    Qwen3OmniSpeechConfig m_config;
    bool m_talker_available = false;

    // Thinker text embeddings (for embedding TTS special tokens)
    ov::InferRequest m_thinker_text_embeddings;

    // Sub-models
    ov::InferRequest m_talker;
    ov::InferRequest m_talker_text_embeddings;
    ov::InferRequest m_talker_projections;  // dual-output: text_projection + hidden_projection
    ov::InferRequest m_code_predictor;
    ov::InferRequest m_code2wav;

    // LRU cache for talker token embeddings to avoid redundant inference.
    // Bounded to kMaxEmbeddingCacheSize entries; evicts least recently used on overflow.
    static constexpr size_t kMaxEmbeddingCacheSize = 256;
    std::list<std::pair<int64_t, ov::Tensor>> m_embedding_lru_list;
    std::unordered_map<int64_t, std::list<std::pair<int64_t, ov::Tensor>>::iterator> m_embedding_lru_map;

    // CodePredictor codec_embedding weights [num_groups-1, vocab_size, hidden_size]
    // Loaded from code_predictor_codec_embedding.npy (exported by optimum-intel)
    ov::Tensor m_cp_codec_embeddings;
    size_t m_cp_vocab_size = 0;
    size_t m_cp_hidden_size = 0;
    bool m_has_cp_embeds = false;

    // Pre-allocated scratch buffers reused across generate_speech() calls
    std::vector<float> m_talker_buf;
    ov::Tensor m_cp_embed_sum;
    ov::Tensor m_stack_codes_buf;

    /// @brief Resolve speaker name to codec token ID.
    int64_t resolve_speaker_id(const std::string& speaker) const;

    /// @brief Embed a token via thinker word embeddings (for TTS special tokens).
    /// @return Tensor [1, 1, thinker_hidden_size].
    ov::Tensor embed_thinker_token(int64_t token_id);

    /// @brief Embed a single token via talker text embeddings.
    /// @return Tensor [1, 1, talker_hidden_size].
    ov::Tensor embed_talker_token(int64_t token_id);

    /// @brief Get raw pointer to CodePredictor codec_embedding weights for a step/code.
    /// @param step Step index 0..num_code_groups-2
    /// @param code Token ID (0..cp_vocab_size-1)
    /// @return Pointer to [cp_hidden_size] floats within m_cp_codec_embeddings.
    const float* cp_token_weights(size_t step, int64_t code) const;

    /// @brief Project thinker hidden state through text projection.
    /// @param hidden_state [1, seq_len, thinker_hidden_size]
    /// @return [1, seq_len, talker_hidden_size]
    ov::Tensor project_text(const ov::Tensor& hidden_state);

    /// @brief Project thinker intermediate hidden state through hidden projection.
    /// @param hidden_state [1, seq_len, thinker_hidden_size]
    /// @return [1, seq_len, talker_hidden_size]
    ov::Tensor project_hidden(const ov::Tensor& hidden_state);

    /// @brief Build the talker input embeddings from thinker outputs.
    /// @return Pair of (talker_input_embeds, trailing_text_hidden).
    std::pair<ov::Tensor, ov::Tensor> build_talker_input(const std::vector<int64_t>& full_token_ids,
                                                         const std::vector<ov::Tensor>& all_hidden_states,
                                                         const std::vector<ov::Tensor>& all_intermediate_hidden_states,
                                                         int64_t speaker_codec_id);

    /// @brief Run the CodePredictor mini-loop for one talker step.
    /// @param talker_hidden_state The last hidden state from talker [1, 1, talker_hidden_size].
    /// @param first_code The first codec code from talker sampling.
    /// @return Pair of (additional_codes, codec_embeddings_sum) where codec_embeddings_sum
    ///         is the sum of all 15 step-specific codec embeddings [1, 1, hidden_size].
    std::pair<std::vector<int64_t>, ov::Tensor> predict_codes(const ov::Tensor& talker_hidden_state,
                                                              int64_t first_code);

    /// @brief Convert codec codes to waveform.
    ov::Tensor codes_to_wav(const ov::Tensor& codes);

    /// @brief Reset talker KV cache state.
    void reset_talker();

    /// @brief Reset code predictor KV cache state.
    void reset_code_predictor();

    /// @brief Sample from logits with temperature, top-k, repetition penalty, and token suppression.
    static int64_t sample_top_k(const float* logits,
                                size_t vocab_size,
                                float temperature,
                                size_t top_k,
                                float repetition_penalty,
                                const std::vector<int64_t>& generated_tokens,
                                const std::vector<int64_t>& suppress_tokens = {});
};

}  // namespace ov::genai
