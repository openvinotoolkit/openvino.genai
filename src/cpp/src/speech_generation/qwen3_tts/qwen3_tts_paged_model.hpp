// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "continuous_batching/pipeline_impl.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_codec_engine.hpp"
#include "speech_generation/text2speech_pipeline_impl.hpp"

namespace ov {
namespace genai {

// PagedAttention + ContinuousBatching driver for the Qwen3-TTS talker.
//
// Composes `Qwen3TTSCodecEngine` (embeddings, optional fused single-dispatch subcode predictor,
// streaming codec decoder pool) with `ContinuousBatchingImpl` in `ModelInputType::EMBEDDINGS`
// mode driven by a `PostSampleEmbedCallback`:
//   - Paged KV cache (U8 default on GPU, block_size=16 on GPU / 32 on CPU)
//   - Prefix caching on shared embedding prefixes (instruct, role, speaker)
//   - Chunked prefill (`dynamic_split_fuse`)
//   - Multi-request continuous batching with batched fused subcode prediction
class Qwen3TTSPagedImpl : public Text2SpeechPipelineImpl {
public:
    Qwen3TTSPagedImpl(const std::filesystem::path& models_path,
                      const std::string& device,
                      const ov::AnyMap& properties,
                      const Tokenizer& tokenizer);

    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config) override;

    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config,
                                       const AudioStreamerVariant& streamer) override;

    ov::Shape get_speaker_embedding_shape() const override;

private:
    Qwen3TTSCodecEngine m_engine;
    std::unique_ptr<EmbeddingContinuousBatchingPipeline> m_cb_pipeline;
    std::string m_talker_device;
};

}  // namespace genai
}  // namespace ov
