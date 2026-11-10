// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <utility>

#include "automatic_speech_recognition/models/sensevoice/input_limits.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

/**
 * Reshapes SenseVoiceSmall to the static shapes required by NPU and compiles it.
 *
 * The model accepts a single audio of at most 30 seconds. Features of shorter audio are zero-padded
 * to the static length, so the returned limits are `max_batch == 1`, the padded frame count and
 * the maximum audio length.
 */
std::pair<ov::CompiledModel, SenseVoiceInputLimits> compile_sensevoice_for_npu(
    const std::filesystem::path& model_path,
    const ov::AnyMap& properties);

}  // namespace ov::genai
