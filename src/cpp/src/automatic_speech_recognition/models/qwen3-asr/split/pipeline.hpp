// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "automatic_speech_recognition/models/qwen3-asr/pipeline_base.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

namespace ov::genai {

/**
 * @brief Runs Qwen3-ASR split export (audio encoder, text embeddings and language model) through
 * ContinuousBatchingPipeline with PagedAttention backend. Audio chunks are transcribed as a single continuous batch.
 */
class Qwen3ASRContinuousBatchingAdapter : public Qwen3ASRPipelineBase {
public:
    Qwen3ASRContinuousBatchingAdapter(const std::filesystem::path& models_path,
                                      const std::string& device,
                                      const ov::AnyMap& properties);

private:
    ContinuousBatchingPipeline m_impl;

    std::vector<std::string> infer(const std::vector<AudioChunk>& chunks,
                                   const ASRGenerationConfig& config,
                                   ASRPerfMetrics& perf_metrics,
                                   const std::shared_ptr<StreamerBase>& streamer) override;
};

/**
 * @brief Runs Qwen3-ASR split export through VLMPipeline with SDPA backend. Audio chunks are transcribed sequentially.
 */
class Qwen3ASRVLMPipelineAdapter : public Qwen3ASRPipelineBase {
public:
    Qwen3ASRVLMPipelineAdapter(const std::filesystem::path& models_path,
                               const std::string& device,
                               const ov::AnyMap& properties);

private:
    // VLMPipeline is created before Qwen3ASRPipelineBase to share its tokenizer.
    Qwen3ASRVLMPipelineAdapter(const std::filesystem::path& models_path,
                               std::unique_ptr<VLMPipeline> impl,
                               const ov::AnyMap& properties);

    std::unique_ptr<VLMPipeline> m_impl;

    std::vector<std::string> infer(const std::vector<AudioChunk>& chunks,
                                   const ASRGenerationConfig& config,
                                   ASRPerfMetrics& perf_metrics,
                                   const std::shared_ptr<StreamerBase>& streamer) override;
};

/**
 * @brief Create Qwen3-ASR split export pipeline for the requested attention backend. PagedAttention backend is used
 * by default and falls back to SDPA if it isn't available, unless PagedAttention is explicitly requested.
 */
std::unique_ptr<ASRPipelineImplBase> create_qwen3_asr_split_pipeline(const std::filesystem::path& models_path,
                                                                     const std::string& device,
                                                                     const ov::AnyMap& properties);

}  // namespace ov::genai
