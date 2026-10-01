// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "speech_generation/text2speech_continuous_batching_pipeline_impl.hpp"

namespace ov {
namespace genai {

// Thin adapter from Text2SpeechContinuousBatchingPipelineImpl's interface onto the friended
// ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl engine.
class Qwen3TTSContinuousBatchingPipelineImpl : public Text2SpeechContinuousBatchingPipelineImpl {
public:
    Qwen3TTSContinuousBatchingPipelineImpl(const std::filesystem::path& models_path,
                                          const SchedulerConfig& scheduler_config,
                                          const std::string& device,
                                          const ov::AnyMap& properties);

    SpeechGenerationHandle add_request(uint64_t request_id,
                                       const std::string& text,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config) override;

    void step() override;
    bool has_non_finished_requests() const override;
    ov::Shape get_speaker_embedding_shape() const override;

private:
    std::shared_ptr<ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl> m_impl;
};

}  // namespace genai
}  // namespace ov
