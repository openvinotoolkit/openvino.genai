// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/model.hpp"

namespace ov::genai {
void validate_gguf_clip_language(const std::shared_ptr<ov::Model>& model);
std::shared_ptr<ov::Model> adapt_gguf_clip_encoder(const std::shared_ptr<ov::Model>& model);
}
