// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

namespace ov {
namespace genai {
namespace utils {

template <typename T>
bool has_token_type_ids_input(const T& inputs) {
    for (const auto& input : inputs) {
        if (input.get_any_name() == "token_type_ids") {
            return true;
        }
    }
    return false;
}

void reshape_model(std::shared_ptr<ov::Model>& model,
                   const TextEmbeddingPipeline::Config& config,
                   std::optional<size_t> max_position_embeddings);
std::shared_ptr<ov::Model> apply_postprocessing(std::shared_ptr<ov::Model> model,
                                                const TextEmbeddingPipeline::Config& config);
std::shared_ptr<ov::Model> create_post_model(std::shared_ptr<ov::Model> model,
                                             const TextEmbeddingPipeline::Config& config);

}  // namespace utils
}  // namespace genai
}  // namespace ov
