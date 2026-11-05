// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/runtime/core.hpp>

#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "sampling/sampler.hpp"

namespace ov::genai {

class Qwen3ASRDecoder {
public:
    Qwen3ASRDecoder(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    virtual ~Qwen3ASRDecoder() = default;

    EncodedResults generate(const ov::Tensor& input_ids,
                            const ov::Tensor& encoder_hidden_state,
                            const ASRGenerationConfig& config,
                            RawPerfMetrics& raw_metrics,
                            ASRRawPerfMetrics& asr_raw_metrics,
                            const std::shared_ptr<StreamerBase>& streamer_ptr = nullptr);

    void set_seed(size_t seed);

protected:
    Qwen3ASRDecoder() = default;

    static std::shared_ptr<ov::Model> read_model(const std::filesystem::path& models_path,
                                                 const ov::AnyMap& properties);

    InferRequest m_request;

private:
    Sampler m_sampler;
};

}  // namespace ov::genai
