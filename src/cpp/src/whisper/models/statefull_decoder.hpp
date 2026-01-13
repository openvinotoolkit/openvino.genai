// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "decoder.hpp"
#include "openvino/runtime/core.hpp"

namespace ov::genai {

class WhisperStatefullDecoder : public WhisperDecoder {
public:
    WhisperStatefullDecoder(const std::filesystem::path& models_path,
                            const std::string& device,
                            const ov::AnyMap& properties,
                            const ov::PartialShape& lhs_shape,
                            const ov::genai::WhisperConfig& model_config,
                            const std::vector<std::pair<size_t, size_t>>& alignment_heads,
                            const bool decompose_cross_attention_spda_ops);

    void start_async(const Tensor& encoder_hidden_state, const Tensor& input_ids, const Tensor& beam_idx) override;

    Tensor wait() override;

    void reset_state() override;

    ov::Tensor create_host_tensor(const element::Type element_type, const Shape& shape) override;

    std::vector<Tensor> get_alignments_heads_qks() override;

private:
    ov::InferRequest m_request;
    ov::genai::WhisperConfig m_model_config;
    bool m_has_cache_position = true;
    void _set_cache_position_tensor(const size_t seq_len);

    bool m_decompose_cross_attention_spda_ops = false;
    std::vector<std::pair<size_t, size_t>> m_alignment_heads;
};
}  // namespace ov::genai
