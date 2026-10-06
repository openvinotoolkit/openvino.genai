// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/video_generation/generation_config.hpp"

namespace ov::genai::utils {

static constexpr char VIDEO_GENERATION_CONFIG[] = "VIDEO_GENERATION_CONFIG";

/// Whether any modality asks for classifier-free guidance. Mirrors diffusers'
/// 'LTX2Pipeline.do_classifier_free_guidance': the audio scale can request guidance on its own, so the
/// video 'guidance_scale' alone does not decide this.
/// @note Only meaningful once a pipeline's 'replace_defaults' has run, because that is what fills
/// 'audio_guidance_scale' with the loaded model's default.
bool requests_classifier_free_guidance(const VideoGenerationConfig& config);

void validate_generation_config(const VideoGenerationConfig& config);

void update_generation_config(VideoGenerationConfig& config, const ov::AnyMap& properties);

/// Drops the negative prompt when no modality requests guidance, as there is no unconditional branch to
/// apply it to. Must run after 'replace_defaults' - see 'requests_classifier_free_guidance'.
void resolve_negative_prompt(VideoGenerationConfig& config);

std::pair<std::string, ov::Any> generation_config(const VideoGenerationConfig& generation_config);

} // namespace ov::genai::utils
