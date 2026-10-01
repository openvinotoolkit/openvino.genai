// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_handle.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {
class Text2SpeechContinuousBatchingPipelineImpl;

/**
 * @brief Low-level, multi-session counterpart to Text2SpeechPipeline, analogous to how
 * ContinuousBatchingPipeline is a separate, lower-level counterpart to LLMPipeline/VLMPipeline.
 *
 * Text2SpeechPipeline::generate() is a closed batch: it schedules exactly the requests passed to
 * one call and returns only once all of them finish, so it cannot merge independently-arriving
 * sessions into shared PagedAttention scheduling. This class instead exposes add_request()/step(),
 * letting a caller add new text-to-speech requests while others are still generating (run step()
 * in a loop from a dedicated thread, exactly like ContinuousBatchingPipeline's add_request()/step()
 * are used for LLM serving).
 *
 * Currently only Qwen3-TTS models are supported.
 */
class OPENVINO_GENAI_EXPORTS Text2SpeechContinuousBatchingPipeline {
public:
    Text2SpeechContinuousBatchingPipeline(const std::filesystem::path& models_path,
                                         const SchedulerConfig& scheduler_config,
                                         const std::string& device,
                                         const ov::AnyMap& properties = {});

    /**
     * Adds a text-to-speech generation request to the pipeline without blocking.
     * @param request_id Unique id used to track the request; the caller must ensure uniqueness.
     * @param text Input text for which to generate speech.
     * @param speaker_embedding Optional speaker embedding tensor, see Text2SpeechPipeline::generate().
     * @param properties Speech generation parameters specified as properties.
     * @return A handle used to read generated audio and query/control the request's status.
     */
    SpeechGenerationHandle add_request(uint64_t request_id,
                                       const std::string& text,
                                       const ov::Tensor& speaker_embedding = ov::Tensor(),
                                       const ov::AnyMap& properties = {});

    /// @brief Advances all in-flight add_request() sessions by one generation step.
    void step();

    /// @brief Whether there are requests added via add_request() that have not finished yet.
    bool has_non_finished_requests() const;

    /// @brief Get the expected speaker embedding shape for the loaded model.
    ov::Shape get_speaker_embedding_shape() const;

private:
    std::shared_ptr<Text2SpeechContinuousBatchingPipelineImpl> m_impl;
    SpeechGenerationConfig m_speech_gen_config;
};

}  // namespace genai
}  // namespace ov
