// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "modeling/builder_context.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

// KV cache append helper.
// Example shapes (before cache):
//   keys/values: [batch, num_kv_heads, seq_len, head_dim]
// After cache append (with existing cache_len):
//   outputs: [batch, num_kv_heads, cache_len + seq_len, head_dim]
std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                          const Tensor& values,
                                          const Tensor& beam_idx,
                                          int32_t num_kv_heads,
                                          int32_t head_dim,
                                          const std::string& cache_prefix,
                                          const BuilderContext& ctx);


}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
