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
 * @brief device_mapping property maps sub-model role names (for example, "text_embeddings_per_layer",
 * "text_embeddings", "vision_embeddings", "language_model") to target OpenVINO execution devices.
 */
static constexpr ov::Property<std::map<std::string, std::string>> device_mapping{"device_mapping"};

}  // namespace genai
}  // namespace ov
