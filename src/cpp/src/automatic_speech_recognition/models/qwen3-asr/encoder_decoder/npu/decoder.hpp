// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../decoder.hpp"

namespace ov::genai {

ASRGenerationConfig prepare_qwen3_asr_npu_generation(size_t prompt_len,
                                                     const ASRGenerationConfig& config,
                                                     size_t max_prompt_len,
                                                     size_t kv_cache_capacity);

ov::AnyMap prepare_qwen3_asr_npu_properties(const ov::AnyMap& properties, size_t max_encoder_len);

class Qwen3ASRDecoderNPU final : public Qwen3ASRDecoder {
public:
    Qwen3ASRDecoderNPU(const std::filesystem::path& models_path, const ov::AnyMap& properties, size_t max_encoder_len);

private:
    ASRGenerationConfig prepare_generation_config(size_t prompt_len, const ASRGenerationConfig& config) const override;

    size_t m_max_prompt_len = 0;
    size_t m_kv_cache_capacity = 0;
};

}  // namespace ov::genai
