// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "alignment_heads.hpp"

namespace ov::genai {

std::vector<ov::Tensor> get_whisper_alignments_heads_qks(
    ov::InferRequest& request,
    const std::vector<std::pair<size_t, size_t>>& alignment_heads) {
    // [layers] * [batch, num_heads, seq_len, frame_len] -> [layers] * [batch, seq_len, frame_len]
    std::vector<ov::Tensor> alignment_qks;
    for (const auto& [layer_idx, head_idx] : alignment_heads) {
        const ov::Tensor alignment_tensor =
            request.get_tensor("cross_attention_qk_scaled_scores_" + std::to_string(layer_idx));

        // [batch, num_heads, seq_len, frame_len]
        const ov::Shape& alignment_shape = alignment_tensor.get_shape();

        // [batch, seq_len, frame_len]
        ov::Tensor head_tensor{ov::element::f32, {alignment_shape[0], alignment_shape[2], alignment_shape[3]}};
        auto* alignment_data = alignment_tensor.data<float>();
        auto* head_data = head_tensor.data<float>();
        const size_t batch_size = alignment_shape[0];
        const size_t num_heads = alignment_shape[1];
        const size_t seq_len = alignment_shape[2];
        const size_t frame_len = alignment_shape[3];

        for (size_t batch = 0; batch < batch_size; ++batch) {
            const size_t batch_offset = batch * num_heads * seq_len * frame_len;
            const size_t head_offset = head_idx * seq_len * frame_len;
            const size_t head_batch_offset = batch * seq_len * frame_len;

            std::memcpy(head_data + head_batch_offset,
                        alignment_data + batch_offset + head_offset,
                        seq_len * frame_len * sizeof(float));
        }

        alignment_qks.push_back(head_tensor);
    }

    return alignment_qks;
}

}  // namespace ov::genai
