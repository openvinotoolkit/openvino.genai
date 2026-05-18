// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "openvino/genai/llm_pipeline.hpp"
#include "sampling/sampler.hpp"
#include "visual_language/embedding_model.hpp"

namespace ov {
namespace genai {

ov::genai::utils::GenerationFinishInfo get_lm_encoded_results(
    ov::InferRequest& m_llm,
    const ov::Tensor& input_ids,
    const ov::Tensor& attention_mask,
    const std::shared_ptr<StreamerBase>& streamer_ptr,
    Sampler& sampler,
    std::vector<SequenceGroup::Ptr> sequence_groups,
    std::optional<ov::Tensor> position_ids,
    std::optional<ov::Tensor> token_type_ids,
    utils::CacheState& m_cache_state,
    EmbeddingsModel::Ptr m_embedding,
    std::optional<int64_t> rope_delta = std::nullopt,
    const size_t max_kv_cache_size = std::numeric_limits<size_t>::max(),
    const bool use_intermediate_remote_tensor = true,
    const std::unordered_map<std::string, ov::Tensor>& lm_extra_inputs = {},
    std::function<ov::Tensor(const ov::Tensor& new_input_ids)> per_layer_embeddings_callback = nullptr);

void align_cache_and_history(const ov::Tensor& new_chat_tokens, utils::CacheState& cache_state);

TokenizedInputs get_chat_encoded_input(const ov::Tensor& new_chat_tokens, utils::CacheState& cache_state);

}  // namespace genai
}  // namespace ov
