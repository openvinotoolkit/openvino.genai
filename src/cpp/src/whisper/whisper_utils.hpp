// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>

#include <openvino/openvino.hpp>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {
namespace utils {

void infer_with_perf_metrics(ov::InferRequest& request,
                             ov::genai::RawPerfMetrics& raw_metrics,
                             std::vector<ov::genai::MicroSeconds>& extra_durations);

int64_t argmax(const ov::Tensor& logits, const size_t batch_idx);

ov::genai::WhisperGenerationConfig prepare_per_generate_config(
    const ov::genai::WhisperGenerationConfig& base_config,
    const ov::genai::OptionalWhisperGenerationConfig& per_generate_config);

std::string find_language_by_token_id(const std::map<std::string, int64_t>& lang_to_id, int64_t token_id);

// Resolve a Whisper language token id from a plain code ("en") or a wrapped
// token ("<|en|>"). Plain codes are normalized to the wrapped form used by
// lang_to_id. Throws if the normalized key is not present.
int64_t get_or_throw_token_id_by_language(const std::map<std::string, int64_t>& lang_to_id,
                                          const std::string& language);

// "<|en|>" -> "en"
std::string to_unescaped_language(const std::string& language);

}  // namespace utils
}  // namespace genai
}  // namespace ov
