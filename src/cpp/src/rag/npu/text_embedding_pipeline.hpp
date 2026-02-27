// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov {
namespace genai {

InferRequest create_text_embedding_npu_request(std::shared_ptr<ov::Model>& model,
                                               const TextEmbeddingPipeline::Config& config,
                                               const ov::AnyMap& properties,
                                               std::optional<size_t> max_position_embeddings,
                                               const bool is_seq_len_fixed);

InferRequest create_text_embedding_npu_post_request(std::shared_ptr<ov::Model>& model,
                                                    const TextEmbeddingPipeline::Config& config);

}  // namespace genai
}  // namespace ov
