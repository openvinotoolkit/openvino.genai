// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {
void validate_gguf_clip_language(const std::shared_ptr<ov::Model>& model);
std::shared_ptr<ov::Model> adapt_gguf_clip_encoder(const std::shared_ptr<ov::Model>& model);
ov::Tensor merge_gguf_llava_embeddings(const std::vector<bool>& image_slots,
                                     const ov::Tensor& text_embeddings,
                                     const std::vector<ov::Tensor>& image_embeddings);
}
