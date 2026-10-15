// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/video_generation/generation_config.hpp"

namespace ov::genai::utils {

static constexpr char VIDEO_GENERATION_CONFIG[] = "VIDEO_GENERATION_CONFIG";

/// Whether either modality requests classifier-free guidance. Call after 'replace_defaults'.
bool requests_classifier_free_guidance(const VideoGenerationConfig& config);

void validate_generation_config(const VideoGenerationConfig& config);

void update_generation_config(VideoGenerationConfig& config, const ov::AnyMap& properties);

/// Drops the negative prompt when no modality requests guidance. Call after 'replace_defaults'.
void resolve_negative_prompt(VideoGenerationConfig& config);

std::pair<std::string, ov::Any> generation_config(const VideoGenerationConfig& generation_config);

} // namespace ov::genai::utils
