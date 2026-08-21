// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <random>
#include <string>
#include <unordered_map>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "text2speech_pipeline_impl.hpp"

namespace ov {
namespace genai {

class Qwen3TTSImpl : public Text2SpeechPipelineImpl {
public:
    // Internal voice-clone prompt used by the generate() dispatch path (ICL mode only).
    struct Qwen3VoiceClonePrompt {
        ov::Tensor ref_spk_embedding;
        ov::Tensor ref_code;
        std::string ref_text;
    };
    Qwen3TTSImpl(const std::filesystem::path& models_path,
                 const std::string& device,
                 const ov::AnyMap& properties,
                 const Tokenizer& tokenizer);

    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config) override;

    ov::Shape get_speaker_embedding_shape() const override;

private:
    struct QwenIds {
        int64_t tts_bos_token_id = -1;
        int64_t tts_eos_token_id = -1;
        int64_t tts_pad_token_id = -1;

        int64_t codec_bos_id = -1;
        int64_t codec_pad_id = -1;
        int64_t codec_eos_token_id = -1;
        int64_t codec_think_id = -1;
        int64_t codec_nothink_id = -1;
        int64_t codec_think_bos_id = -1;
        int64_t codec_think_eos_id = -1;

        size_t num_code_groups = 16;
        size_t talker_vocab_size = 3072;

        std::unordered_map<std::string, int64_t> codec_language_id;
        std::unordered_map<std::string, int64_t> spk_id;
        std::unordered_map<std::string, std::string> spk_is_dialect;
    };

    void init_config(const std::filesystem::path& models_path);

    ov::Tensor infer_embedding(ov::InferRequest& request, int64_t token_id);
    ov::Tensor infer_embedding_seq(ov::InferRequest& request, const std::vector<int64_t>& token_ids);
    ov::Tensor infer_text_projection(const ov::Tensor& hidden_states);
    ov::Tensor infer_talker(const ov::Tensor& inputs_embeds,
                            const ov::Tensor& attention_mask,
                            const ov::Tensor& position_ids,
                            bool reset_state);
    ov::Tensor infer_talker_hidden(const ov::Tensor& inputs_embeds,
                                   const ov::Tensor& attention_mask,
                                   const ov::Tensor& position_ids,
                                   bool reset_state);
    // Single-token step against the static all-heads code predictor. Manages the
    // host-side explicit KV cache and absolute-position counter internally; pass
    // reset_state=true at the start of each per-frame code-group prediction.
    // Returns the stacked all-heads logits [num_heads, 1, 1, vocab].
    ov::Tensor infer_predictor(const ov::Tensor& inputs_embeds, bool reset_state);
    // Slice one MTP head (0-based, = code_group-1) out of stacked all-heads logits
    // into a [1, 1, vocab] tensor for sample_token_from_logits.
    ov::Tensor select_predictor_head(const ov::Tensor& all_logits, size_t head) const;
    // Read static-predictor dims from the IR and (re)allocate host KV buffers.
    void init_static_predictor_meta(const std::shared_ptr<ov::Model>& model);
    // Reshape a dynamically-exported code predictor IR to the required static
    // shapes (no-op if the IR is already static). Derives the KV window from the
    // stacked-heads `logits` output and the fixed n_kv/head_dim of past_key_values.
    void reshape_predictor_to_static(const std::shared_ptr<ov::Model>& model);
    // Zero host KV buffers and reset the running absolute-position counter.
    void reset_predictor_state();
    ov::Tensor infer_predictor_embedding(int64_t token_id, int64_t generation_step);
    ov::Tensor infer_predictor_embedding_seq(const std::vector<int64_t>& token_ids, int64_t generation_step);

    int64_t sample_token_from_logits(const ov::Tensor& logits,
                                     const SpeechGenerationConfig& generation_config,
                                     const std::vector<int64_t>& generated,
                                     const std::vector<bool>& suppressed,
                                     std::mt19937& rng) const;
    std::vector<int64_t> generate_codec_groups(const ov::Tensor& past_hidden,
                                               int64_t first_codec_token,
                                               const SpeechGenerationConfig& generation_config,
                                               std::mt19937& rng);

    Text2SpeechDecodedResults generate_voice_clone(const std::string& text,
                                                   const Qwen3VoiceClonePrompt& prompt,
                                                   const SpeechGenerationConfig& generation_config);

    Text2SpeechDecodedResults decode_from_prefill(const ov::Tensor& talker_prefill,
                                                  const ov::Tensor& tts_pad,
                                                  const ov::Tensor& trailing_text_hidden,
                                                  const SpeechGenerationConfig& generation_config,
                                                  const std::vector<bool>& suppress_tokens);

    ov::Tensor make_attention_mask(size_t length);
    ov::Tensor make_position_ids_prefill(size_t length);
    ov::Tensor make_position_ids_decode(size_t absolute_position);
    ov::Tensor make_predictor_position_ids(size_t start_position, size_t length);

    std::vector<float> decode_speech_tokenizer(const std::vector<int64_t>& codes);

    std::string normalize_text_language(const std::string& language) const;
    std::string normalize_speaker(const std::string& speaker) const;
    bool is_base_model() const;
    ov::Tensor normalize_external_speaker_embedding(const ov::Tensor& speaker_embedding,
                                                    size_t hidden_size) const;
    std::vector<float> normalize_ref_audio_waveform(const ov::Tensor& ref_audio) const;
    ov::Tensor extract_qwen3_speaker_embedding_from_audio(const ov::Tensor& ref_audio) const;
    ov::Tensor extract_qwen3_ref_code_from_audio(const ov::Tensor& ref_audio) const;

private:
    std::filesystem::path m_models_path;
    std::string m_device;
    bool m_is_npu = false;
    Tokenizer m_tokenizer;
    QwenIds m_ids;
    std::string m_tts_model_type = "custom_voice";
    size_t m_speaker_embedding_dim = 1024;
    size_t m_talker_hidden_size = 1024;  // talker hidden_size; predictor inputs_embeds width

    ov::InferRequest m_talker;
    ov::InferRequest m_talker_embedding;
    ov::InferRequest m_talker_text_embedding;
    ov::InferRequest m_talker_text_projection;
    ov::InferRequest m_talker_code_predictor;
    ov::InferRequest m_talker_code_predictor_embedding;

    // Static all-heads code predictor state. The loaded predictor IR must expose
    // explicit past_key_values inputs (a non-static model is rejected at
    // construction); the host drives the KV cache, see infer_predictor.
    size_t m_pred_num_layers = 0;
    size_t m_pred_n_kv = 0;
    size_t m_pred_head_dim = 0;
    size_t m_pred_past_len = 0;   // KV slots in past_key_values inputs
    size_t m_pred_kv_len = 0;     // KV slots in present outputs (= past_len + 1)
    size_t m_pred_num_heads = 0;  // stacked code-group heads in `logits`
    size_t m_pred_vocab = 0;
    size_t m_pred_position = 0;   // running absolute position within a frame
    std::vector<ov::Tensor> m_pred_past_k;  // per-layer host KV buffers
    std::vector<ov::Tensor> m_pred_past_v;
    // Pre-allocated reusable tensors for infer_predictor (avoid repeated allocations)
    ov::Tensor m_pred_attn;   // attention_mask, reused across calls
    ov::Tensor m_pred_pos;    // position_ids, reused across calls
    // Pre-allocated reusable tensors for infer_predictor_embedding (avoid repeated allocations)
    ov::Tensor m_pred_emb_ids;  // token ids input, reused across calls
    ov::Tensor m_pred_emb_step;  // generation_steps input, reused across calls
    ov::InferRequest m_speech_tokenizer_decoder;
    ov::InferRequest m_qwen3_mel_preprocess;
    ov::InferRequest m_speaker_encoder;
    ov::InferRequest m_speech_tokenizer_encoder;

    uint32_t m_output_sample_rate = 24000;
    mutable std::unordered_map<std::string, double>   m_perf_ms;
    mutable std::unordered_map<std::string, int64_t>  m_perf_calls;
    mutable std::unordered_map<std::string, std::string> m_perf_device;
    mutable int64_t m_talker_prefill_tokens = 0;

    void perf_print_and_reset() const;

    uint32_t m_decoder_num_quantizers = 16;
    uint32_t m_decoder_upsample = 1920;
    uint32_t m_speaker_encoder_sample_rate = 24000;
    uint32_t m_speaker_encoder_mel_dim = 128;
    uint32_t m_speech_tokenizer_input_sample_rate = 24000;

    bool m_has_speaker_encoder = false;
    bool m_has_speech_tokenizer_encoder = false;
    bool m_has_qwen3_mel_preprocess = false;
};

}  // namespace genai
}  // namespace ov
