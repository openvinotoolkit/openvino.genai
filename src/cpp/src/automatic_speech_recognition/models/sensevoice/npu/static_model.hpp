// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>

#include "automatic_speech_recognition/models/sensevoice/windowing.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

/**
 * Reshapes SenseVoiceSmall to the static shapes required by NPU and compiles it.
 *
 * The model accepts a single audio window of 30 seconds. Features of shorter audio are zero-padded to the static
 * length. Longer audio is split into windows by the windowing from create_static_windowing().
 */
ov::CompiledModel compile_sensevoice_for_npu(const std::filesystem::path& model_path, const ov::AnyMap& properties);

/**
 * Windowing for a model with the static input shape [max_batch, padded_frames, features].
 *
 * Audio that does not fit the shape is split into windows at low-energy positions, which are transcribed one by one.
 * Texts of the windows are joined and language, emotion and event are chosen by the largest total window duration.
 *
 * Limitation: when streaming, tokens of consecutive windows are written without a separator, so a space between
 * windows of space-delimited languages can be missing in the streamed text compared to the resulting text.
 */
std::unique_ptr<SenseVoiceWindowing> create_static_windowing(size_t max_batch, size_t padded_frames);

}  // namespace ov::genai
