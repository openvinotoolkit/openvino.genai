// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/runtime/core.hpp"
#include "whisper/config.hpp"

namespace ov::genai {
class WhisperDecoder {
public:
    // Effective CPU inference precision; dynamic for other devices.
    static std::shared_ptr<WhisperDecoder> from_path(const std::filesystem::path& models_path,
                                                     const std::string& device,
                                                     const ov::AnyMap& properties,
                                                     const ov::PartialShape& lhs_shape,
                                                     const bool decompose_cross_attention_spda_ops,
                                                     const ov::element::Type& inference_precision);

    // Batch decoding is validated for f32 and the dynamic precision marker used for ACCURACY mode.
    static bool is_batch_validated_precision(const ov::element::Type& inference_precision) {
        return inference_precision == ov::element::f32 || inference_precision == ov::element::dynamic;
    }

    std::pair<int64_t, float> detect_language(const Tensor& encoder_hidden_state,
                                              const WhisperGenerationConfig& config);

    // Detects one language per encoder row; multiple rows require supports_dynamic_batch().
    std::pair<std::vector<int64_t>, float> detect_languages(const Tensor& encoder_hidden_state,
                                                            const WhisperGenerationConfig& config);

    // Whether this decoder supports multi-audio batching and row compaction.
    virtual bool supports_dynamic_batch() const {
        return false;
    }

    /**
     * Retains rows in source_rows order. The next start_async() must pass
     * source_rows as beam_idx to reorder self-attention KV state.
     */
    virtual void retain_rows([[maybe_unused]] const std::vector<int32_t>& source_rows) {
        OPENVINO_THROW("This Whisper decoder does not support removing decoder rows.");
    }

    virtual void start_async(const Tensor& encoder_hidden_state, const Tensor& input_ids, const Tensor& beam_idx) = 0;

    virtual Tensor wait() = 0;

    virtual void reset_state() = 0;

    virtual ov::Tensor create_host_tensor(const element::Type element_type, const Shape& shape);

    virtual std::vector<Tensor> get_alignments_heads_qks(
        const std::vector<std::pair<size_t, size_t>>& alignment_heads) {
        OPENVINO_THROW("Not implemented");
    }

    virtual ~WhisperDecoder();

private:
    std::pair<Tensor, float> infer_language_logits(const Tensor& encoder_hidden_state,
                                                   const WhisperGenerationConfig& config,
                                                   const size_t rows);
};
}  // namespace ov::genai
