// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
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

    EncodedResults generate(const ov::Tensor& input_ids,
                            const ov::Tensor& encoder_hidden_state,
                            const ASRGenerationConfig& config,
                            RawPerfMetrics& raw_metrics,
                            ASRRawPerfMetrics& asr_raw_metrics,
                            const std::shared_ptr<StreamerBase>& streamer_ptr = nullptr);

    void set_seed(size_t seed);

private:
    InferRequest m_request;
    InferRequest m_text_embedding_request;
    ov::Tensor m_text_embedding_input;
    InferRequest m_text_embedding_decode_request;
    ov::Tensor m_text_embedding_decode_input;
    Sampler m_sampler;
    bool m_is_npu = false;
    size_t m_max_prompt_len = 0;
    size_t m_max_kv_cache_size = 0;
    int64_t m_audio_token_id = -1;
    size_t m_hidden_size = 0;
};

}  // namespace ov::genai
