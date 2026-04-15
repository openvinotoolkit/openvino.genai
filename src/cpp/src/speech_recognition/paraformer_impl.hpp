// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>
#include <unordered_map>

#include "asr_pipeline_impl_base.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace genai {

/**
 * @brief FBANK Feature Extractor for Paraformer models.
 * 
 * Extracts 80-dimensional log-mel filterbank features with LFR (Low Frame Rate) stacking.
 * This matches the preprocessing expected by Paraformer models.
 */
class ParaformerFeatureExtractor {
public:
    ParaformerFeatureExtractor(int sample_rate = 16000, 
                                int n_mels = 80,
                                int n_fft = 400,
                                int hop_length = 160,
                                int win_length = 400,
                                int lfr_m = 7,
                                int lfr_n = 6);
    
    /**
     * @brief Extract FBANK features from raw audio with LFR stacking.
     * @param audio Raw audio samples (single channel, float)
     * @param sample_rate Input sample rate (will resample if needed)
     * @return Tensor of shape [1, num_lfr_frames, n_mels * lfr_m] (e.g., [1, ?, 560])
     */
    ov::Tensor extract(const std::vector<float>& audio, int sample_rate = 16000);

private:
    int m_sample_rate;
    int m_n_mels;
    int m_n_fft;
    int m_hop_length;
    int m_win_length;
    int m_lfr_m;  // LFR stacking: number of frames to stack
    int m_lfr_n;  // LFR stacking: skip between stacks
    
    // Mel filterbank matrix
    std::vector<std::vector<float>> m_mel_filters;
    
    void init_mel_filters();
    std::vector<float> apply_preemphasis(const std::vector<float>& audio, float coeff = 0.97f);
    std::vector<std::vector<float>> compute_stft(const std::vector<float>& audio);
    std::vector<float> hann_window(int length);
};

/**
 * @brief Detokenizer for Paraformer models.
 * 
 * Handles token ID to text conversion with proper Chinese text handling.
 */
class ParaformerDetokenizer {
public:
    ParaformerDetokenizer() = default;
    
    /**
     * @brief Load vocabulary from tokens.json file.
     */
    bool load(const std::filesystem::path& tokens_path);
    
    /**
     * @brief Decode token IDs to text.
     */
    std::string decode(const std::vector<int64_t>& token_ids) const;
    
    /**
     * @brief Check if vocabulary is loaded.
     */
    bool is_loaded() const { return m_loaded; }

private:
    std::unordered_map<int64_t, std::string> m_vocab;
    bool m_loaded = false;
    
    // Special token IDs
    int64_t m_blank_id = 0;
    int64_t m_sos_id = 1;
    int64_t m_eos_id = 2;
};

/**
 * @brief ASR back-end for Paraformer (FunASR) models.
 *
 * Loads openvino_model.xml + tokens.json and performs CTC-style decoding
 * of the model output. Includes FBANK feature extraction for raw audio input.
 */
class ParaformerImpl : public ASRPipelineImplBase {
public:
    explicit ParaformerImpl(const std::filesystem::path& model_dir,
                            const std::string& device,
                            const ov::AnyMap& properties = {});
    ~ParaformerImpl() override;

    // ── Core inference (Generic API) ────────────────────────────────────

    ASRDecodedResults generate(
        const RawSpeechInput& raw_speech_input,
        const ASRGenerationConfig& config,
        const std::shared_ptr<StreamerBase> streamer) override;

    // ── Accessors ───────────────────────────────────────────────────────

    Tokenizer get_tokenizer() override;
    
    ASRGenerationConfig get_generation_config() const override;
    void set_generation_config(const ASRGenerationConfig& config) override;

private:
    ov::CompiledModel m_compiled_model;
    ASRGenerationConfig m_generation_config;
    
    // Feature extraction
    std::unique_ptr<ParaformerFeatureExtractor> m_feature_extractor;
    
    // Detokenizer
    ParaformerDetokenizer m_detokenizer;
    
    // Legacy token map (for backward compatibility)
    std::vector<std::string> m_tokens;
    bool m_has_tokens = false;

    void load_tokens(const std::filesystem::path& tokens_path);
    std::string decode_ids(const std::vector<int64_t>& ids) const;
    
    // Perform CTC decoding on model output
    std::vector<int64_t> ctc_decode(const ov::Tensor& logits);
};

}  // namespace genai
}  // namespace ov