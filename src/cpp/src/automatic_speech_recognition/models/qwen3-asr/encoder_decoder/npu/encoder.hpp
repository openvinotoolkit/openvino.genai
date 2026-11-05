// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../encoder.hpp"

namespace ov::genai {

class Qwen3ASREncoderNPU final : public Qwen3ASREncoder {
public:
    Qwen3ASREncoderNPU(const std::filesystem::path& models_path, const ov::AnyMap& properties);

    size_t get_max_encoder_len(size_t max_asr_input_seconds, size_t sampling_rate, size_t hop_length) const;

private:
    ov::Tensor infer(const ov::Tensor& input) override;
    void reset_input() override;

    ov::PartialShape m_last_hidden_state_shape;
};

}  // namespace ov::genai
