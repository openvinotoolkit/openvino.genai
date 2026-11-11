// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "circular_buffer_queue.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/lfm3_omni/speech_core.hpp"

namespace ov::genai {

/// @brief Interleaved text and speech decode of LFM3-Omni on the stateful (SDPA) language model. The language model
/// emits text tokens and speech frames in one stream: the depth decoder samples the codebooks of a frame from the last
/// hidden state, and the frame is fed back through audio_out_embeddings. Borrows the stateful language model and text
/// embeddings of the VLM pipeline.
class LFM3OmniSpeechEngine {
public:
    LFM3OmniSpeechEngine(const std::filesystem::path& models_dir,
                         ov::InferRequest& language,
                         EmbeddingsModel::Ptr embedding,
                         const std::string& device,
                         const ov::AnyMap& properties);

    LFM3OmniSpeechEngine(const ModelsMap& models_map,
                         const std::filesystem::path& config_dir_path,
                         ov::InferRequest& language,
                         EmbeddingsModel::Ptr embedding,
                         const std::string& device,
                         const ov::AnyMap& properties);

    /// @param inputs_embeds Prompt embeddings [1, seq_len, hidden_size].
    /// @param attention_mask Ones over the cached history and the prompt, [1, history + seq_len].
    /// @param last_prompt_token Last prompt token id; a marker already in place is not appended again.
    /// @param speech_streamer Optional. When set, receives the 24 kHz waveform in chunks of request.audio_chunk_frames
    /// frames, [1, 1, samples]; without it every frame is decoded once at the end. The caller ends it.
    LFM3SpeechResult generate(const ov::Tensor& inputs_embeds,
                              const ov::Tensor& attention_mask,
                              int64_t last_prompt_token,
                              const LFM3SpeechRequest& request,
                              const std::shared_ptr<StreamerBase>& streamer,
                              const OmniSpeechStreamerVariant& speech_streamer);

private:
    LFM3OmniSpeechEngine(LFM3SpeechModels models, ov::InferRequest& language, EmbeddingsModel::Ptr embedding);

    struct StepOutput {
        std::vector<float> logits;
        ov::Tensor hidden_state;  ///< [1, hidden_size], owned copy
    };

    StepOutput language_step(const ov::Tensor& inputs_embeds, size_t attention_length);
    ov::Tensor embed_token(int64_t token_id);

    LFM3SpeechModels m_models;
    ov::InferRequest& m_language;
    EmbeddingsModel::Ptr m_embedding;
    ov::Tensor m_attention_mask{ov::element::i64, {1, 0}};
    ov::Tensor m_beam_idx{ov::element::i32, {1}};
};

}  // namespace ov::genai
