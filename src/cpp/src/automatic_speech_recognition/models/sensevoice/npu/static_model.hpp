// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

/**
 * Reshapes SenseVoiceSmall to the static shapes required by NPU and compiles it.
 *
 * The model accepts a single audio window of 30 seconds. Features of shorter audio are zero-padded to the static
 * length. Longer audio has to be split into windows, see split_audio_into_windows().
 */
ov::CompiledModel compile_sensevoice_for_npu(const std::filesystem::path& model_path, const ov::AnyMap& properties);

}  // namespace ov::genai
