// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/model.hpp"

namespace ov::genai {
std::shared_ptr<ov::Model> adapt_gguf_clip_encoder(const std::shared_ptr<ov::Model>& model);
std::shared_ptr<ov::Model> adapt_gguf_packed_vision_encoder(const std::shared_ptr<ov::Model>& model,
                                                            size_t patch_size,
                                                            size_t temporal_patch_size);
}  // namespace ov::genai
