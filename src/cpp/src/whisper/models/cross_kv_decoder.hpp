// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "decoder.hpp"
#include "openvino/runtime/core.hpp"

namespace ov::genai {

/**
 * Stateful Whisper decoder with cross-attention K/V supplied as inputs.
 *
 * A batch-1 projector computes K/V per encoder row on the first
 * start_async() after reset. Self-attention K/V remains stateful.
 *
 * Single-row cross-KV supports beam search; per-audio cross-KV supports
 * dynamic batching and row compaction via retain_rows().
 * Beam search across multiple audios is not supported.
 */
class WhisperCrossKVDecoder : public WhisperDecoder {
public:
    // Returns nullptr if the decoder model does not have the f32 stateful cross-attention structure this class needs.
    static std::shared_ptr<WhisperCrossKVDecoder> create(const std::filesystem::path& models_path,
                                                         const std::string& device,
                                                         const ov::AnyMap& properties,
                                                         const ov::PartialShape& lhs_shape,
                                                         const bool decompose_cross_attention_sdpa_ops);

    void start_async(const Tensor& encoder_hidden_state, const Tensor& input_ids, const Tensor& beam_idx) override;

    Tensor wait() override;

    void reset_state() override;

    ov::Tensor create_host_tensor(const element::Type element_type, const Shape& shape) override;

    std::vector<Tensor> get_alignments_heads_qks(
        const std::vector<std::pair<size_t, size_t>>& alignment_heads) override;

    bool supports_dynamic_batch() const override {
        return true;
    }

    void retain_rows(const std::vector<int32_t>& source_rows) override;

private:
    WhisperCrossKVDecoder(ov::InferRequest request,
                          ov::InferRequest projector,
                          std::vector<std::string> cross_kv_names,
                          const ov::element::Type& cross_kv_type,
                          const ov::Shape& cross_kv_row_shape,
                          const bool has_cache_position,
                          const bool decompose_cross_attention_sdpa_ops);

    void compute_cross_kv(const Tensor& encoder_hidden_state);
    void bind_cross_kv();
    void set_cache_position_tensor(const size_t seq_len);
    void validate_beam_idx(const size_t rows, const Tensor& beam_idx) const;

    ov::InferRequest m_request;
    ov::InferRequest m_projector;
    ov::Tensor m_idle_projector_input;
    std::vector<std::string> m_cross_kv_names;
    std::vector<ov::Tensor> m_cross_kv;
    ov::element::Type m_cross_kv_type;
    ov::Shape m_cross_kv_row_shape;
    size_t m_cross_kv_capacity = 0;
    size_t m_cross_kv_rows = 0;
    bool m_cross_kv_valid = false;
    bool m_cross_kv_shared = false;
    std::optional<std::vector<int32_t>> m_retained_rows;
    bool m_infer_in_flight = false;
    bool m_has_cache_position = true;
    bool m_decompose_cross_attention_sdpa_ops = false;
};

}  // namespace ov::genai
