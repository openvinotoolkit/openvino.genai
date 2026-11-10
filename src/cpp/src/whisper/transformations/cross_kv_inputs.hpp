// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov::genai {

struct WhisperCrossKVModels {
    std::shared_ptr<ov::Model> decoder;
    std::shared_ptr<ov::Model> projector;
    // Projector output and decoder input names, ordered by layer (key before value).
    std::vector<std::string> names;
};

/**
 * Splits eligible cross-attention KV states into a projector and
 * explicit decoder inputs, leaving self-attention KV stateful.
 * Returns std::nullopt if the graph cannot be safely split.
 */
std::optional<WhisperCrossKVModels> split_whisper_cross_kv(const std::shared_ptr<ov::Model>& decoder);

}  // namespace ov::genai
