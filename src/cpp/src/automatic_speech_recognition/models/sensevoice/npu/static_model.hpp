// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

/**
 * Reshapes SenseVoiceSmall to the static shapes required by NPU and compiles it.
 * The model accepts an audio window of up to `window_seconds` (30s by default), so shorter audio has to be zero padded
 * and longer audio split.
 */
ov::CompiledModel compile_sensevoice_for_npu(const std::filesystem::path& model_path,
                                             const std::string& device,
                                             std::optional<size_t> window_seconds,
                                             const ov::AnyMap& properties);

}  // namespace ov::genai
