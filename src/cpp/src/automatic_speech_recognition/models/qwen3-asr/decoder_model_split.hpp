// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <openvino/core/model.hpp>

namespace ov::genai {

struct Qwen3ASRDecoderModels {
    std::shared_ptr<ov::Model> textEmbedding = nullptr;
    std::shared_ptr<ov::Model> languageModel = nullptr;
    int64_t audioTokenId = -1;
    size_t hiddenSize = 0;
};

Qwen3ASRDecoderModels splitQwen3ASRDecoderModel(const std::shared_ptr<ov::Model>& decoderModel);

}  // namespace ov::genai
