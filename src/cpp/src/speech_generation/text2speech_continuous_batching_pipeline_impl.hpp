// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_handle.hpp"

namespace ov {
namespace genai {

// Backend interface for Text2SpeechContinuousBatchingPipeline. Unlike Text2SpeechPipelineImpl
// (synchronous generate()), this interface is for backends that schedule multiple in-flight
// requests together via add_request()/step(), analogous to how ContinuousBatchingPipeline is a
// separate, lower-level counterpart to LLMPipeline/VLMPipeline.
class Text2SpeechContinuousBatchingPipelineImpl {
public:
    virtual SpeechGenerationHandle add_request(uint64_t request_id,
                                               const std::string& text,
                                               const ov::Tensor& speaker_embedding,
                                               const SpeechGenerationConfig& generation_config) = 0;

    virtual void step() = 0;
    virtual bool has_non_finished_requests() const = 0;

    /// @brief Get the expected speaker embedding shape for this backend.
    virtual ov::Shape get_speaker_embedding_shape() const = 0;

    /// @brief Get output sample rate of produced waveforms for this backend.
    virtual uint32_t get_output_sample_rate() const = 0;

    virtual ~Text2SpeechContinuousBatchingPipelineImpl() = default;
};

}  // namespace genai
}  // namespace ov
