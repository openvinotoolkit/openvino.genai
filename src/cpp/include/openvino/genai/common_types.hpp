// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>

#include "openvino/core/core.hpp"
#include <openvino/runtime/properties.hpp>

namespace ov {
namespace genai {

/**
 * @brief A map of models for VLMPipeline constructor.
 * Key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
 * and value is a pair of model IR as string and weights as tensor.
 */
using ModelsMap = std::map<std::string, std::pair<std::string, ov::Tensor>>;

/**
 * @brief blob_path property defines a path to a directory containing compiled blobs previously exported with
 * `pipeline.export_model` method.
 *
 * Use of compiled blobs can significantly reduce model load time, especially for large models.
 */
static constexpr ov::Property<std::filesystem::path> blob_path{"blob_path"};

/**
 * @brief device_mapping property maps sub-model role names (such as "text_embeddings_per_layer",
 * "text_embeddings", "vision_embeddings", "audio_embeddings", "resampler", "multi_modal_projector",
 * "vision_embeddings_merger", or "language_model") to target OpenVINO execution devices.
 *
 * When the pipeline runs on a GPU device, "text_embeddings_per_layer" (Per-Layer Embeddings)
 * defaults to "CPU" so its embedding table is memory-mapped in host RAM unless explicitly
 * overridden.
 *
 * Device placement precedence (highest to lowest):
 * 1. MODEL_PROPERTIES[role]["device"] (or "DEVICE")
 * 2. device_mapping[role]
 * 3. Role-specific default ("CPU" for "text_embeddings_per_layer" on GPU)
 * 4. Pipeline target device
 *
 * Compile properties for a role are merged on top of the resolved device in order:
 * global properties -> DEVICE_PROPERTIES[resolved_device] -> MODEL_PROPERTIES[role].
 */
static constexpr ov::Property<std::map<std::string, std::string>> device_mapping{"device_mapping"};

}  // namespace genai
}  // namespace ov
