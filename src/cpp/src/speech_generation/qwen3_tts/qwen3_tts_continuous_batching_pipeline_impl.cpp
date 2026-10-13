// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_continuous_batching_pipeline_impl.hpp"

#include "speech_generation/qwen3_tts/qwen3_tts_continuous_batching_impl.hpp"

namespace ov {
namespace genai {

Qwen3TTSContinuousBatchingPipelineImpl::Qwen3TTSContinuousBatchingPipelineImpl(
    const std::filesystem::path& models_path,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_impl(std::make_shared<ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl>(
          models_path,
          scheduler_config,
          device,
          properties,
          Tokenizer(models_path, properties))) {}

SpeechGenerationHandle Qwen3TTSContinuousBatchingPipelineImpl::add_request(uint64_t request_id,
                                                                          const std::string& text,
                                                                          const ov::Tensor& speaker_embedding,
                                                                          const SpeechGenerationConfig& generation_config) {
    return m_impl->add_speech_request(request_id, text, speaker_embedding, generation_config);
}

void Qwen3TTSContinuousBatchingPipelineImpl::step() {
    m_impl->step();
}

bool Qwen3TTSContinuousBatchingPipelineImpl::has_non_finished_requests() const {
    return m_impl->has_non_finished_requests();
}

ov::Shape Qwen3TTSContinuousBatchingPipelineImpl::get_speaker_embedding_shape() const {
    return m_impl->get_speaker_embedding_shape();
}

uint32_t Qwen3TTSContinuousBatchingPipelineImpl::get_output_sample_rate() const {
    return m_impl->get_output_sample_rate();
}

}  // namespace genai
}  // namespace ov
