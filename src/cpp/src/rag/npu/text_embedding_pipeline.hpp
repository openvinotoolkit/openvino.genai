// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov {
namespace genai {

//
// Enable NPU dynamic prompt input support, which allows text-embedding models to handle long-context inputs more
// effectively.
//
// NPUW_LLM_PREFILL_CHUNK_SIZE:
//     Controls the chunk size for prompt prefill, which determines the granularity of input dynamism.
//     Default value: 1024.
//
// NPUW_F16IC:
//     Forces subgraph interconnect tensors to use FP16 precision when they would otherwise be FP32, provided that the
//     partitioning pipeline is enabled. Setting this property to False may improve accuracy.
//     Default value: True
//

InferRequest create_text_embedding_npu_request(std::shared_ptr<ov::Model>& model,
                                               const TextEmbeddingPipeline::Config& config,
                                               const ov::AnyMap& properties,
                                               std::optional<size_t> max_position_embeddings,
                                               const bool is_seq_len_fixed);

InferRequest create_text_embedding_npu_post_request(std::shared_ptr<ov::Model>& model,
                                                    const TextEmbeddingPipeline::Config& config);

}  // namespace genai
}  // namespace ov
