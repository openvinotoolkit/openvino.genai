// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

/**
 * Reshapes SenseVoiceSmall to the static shapes required by NPU and compiles it.
 * The model accepts a fixed length audio window, so shorter audio has to be zero padded and longer audio split.
 */
ov::CompiledModel compile_sensevoice_for_npu(const std::filesystem::path& model_path, const ov::AnyMap& properties);

}  // namespace ov::genai
