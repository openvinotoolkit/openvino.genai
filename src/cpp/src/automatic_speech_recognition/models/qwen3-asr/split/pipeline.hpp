// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "automatic_speech_recognition/models/qwen3-asr/pipeline_base.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

namespace ov::genai {

/**
 * @brief Runs Qwen3-ASR split export (audio encoder, text embeddings and language model) through
 * ContinuousBatchingPipeline. Audio chunks are transcribed as a single continuous batch.
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

}  // namespace ov::genai
