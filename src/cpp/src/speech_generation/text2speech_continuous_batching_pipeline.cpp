// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/text2speech_continuous_batching_pipeline.hpp"

#include "speech_generation/qwen3_tts/qwen3_tts_continuous_batching_pipeline_impl.hpp"
#include "speech_generation/text2speech_continuous_batching_pipeline_impl.hpp"

namespace ov {
namespace genai {

Text2SpeechContinuousBatchingPipeline::Text2SpeechContinuousBatchingPipeline(
    const std::filesystem::path& models_path,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_impl(std::make_shared<Qwen3TTSContinuousBatchingPipelineImpl>(models_path, scheduler_config, device, properties)) {}

SpeechGenerationHandle Text2SpeechContinuousBatchingPipeline::add_request(uint64_t request_id,
                                                                          const std::string& text,
                                                                          const ov::Tensor& speaker_embedding,
                                                                          const ov::AnyMap& properties) {
    SpeechGenerationConfig request_config = m_speech_gen_config;
    request_config.update_generation_config(properties);
    request_config.validate();
    return m_impl->add_request(request_id, text, speaker_embedding, request_config);
}

void Text2SpeechContinuousBatchingPipeline::step() {
    m_impl->step();
}

bool Text2SpeechContinuousBatchingPipeline::has_non_finished_requests() const {
    return m_impl->has_non_finished_requests();
}

ov::Shape Text2SpeechContinuousBatchingPipeline::get_speaker_embedding_shape() const {
    return m_impl->get_speaker_embedding_shape();
}

}  // namespace genai
}  // namespace ov
