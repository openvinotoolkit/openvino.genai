// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../decoder.hpp"

namespace ov::genai {

class Qwen3ASRDecoderNPU final : public Qwen3ASRDecoder {
public:
    Qwen3ASRDecoderNPU(const std::filesystem::path& models_path, const ov::AnyMap& properties, size_t max_encoder_len);
};

}  // namespace ov::genai
