// Copyright (C) 2024-2026 Intel Corporation
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
                            const bool decompose_cross_attention_spda_ops);

    void start_async(const Tensor& encoder_hidden_state, const Tensor& input_ids, const Tensor& beam_idx) override;

    Tensor wait() override;

    void reset_state() override;

    void start_new_round() override;

    ov::Tensor create_host_tensor(const element::Type element_type, const Shape& shape) override;

    std::vector<Tensor> get_alignments_heads_qks(
        const std::vector<std::pair<size_t, size_t>>& alignment_heads) override;

private:
    ov::InferRequest m_request;
    bool m_has_cache_position = true;
    void _set_cache_position_tensor(const size_t seq_len);

    bool m_decompose_cross_attention_spda_ops = false;

    bool m_recreate_request_per_round = true;

    // TEMPORARY DIAGNOSTIC (macOS 14 CI Concat-exception investigation, GENAI_WHISPER_SHAPE_TRACE=1).
    // Monotonically increasing id of the current m_request, bumped every time create_infer_request() runs.
    int64_t m_request_epoch = -1;
};
}  // namespace ov::genai
