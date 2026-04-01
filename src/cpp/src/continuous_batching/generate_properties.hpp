// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <openvino/runtime/tensor.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"

namespace ov::genai {

struct CBGenerateProperties {
    std::optional<std::vector<std::vector<ov::Tensor>>> images_batches;
    std::optional<std::vector<std::vector<ov::Tensor>>> videos_batches;
    std::optional<std::vector<std::vector<VideoMetadata>>> videos_metadata_batches;
    std::optional<std::vector<GenerationConfig>> generation_config_batches;
    StreamerVariant streamer = std::monostate();

    bool has_vision_properties() const {
        return images_batches.has_value() || videos_batches.has_value() || videos_metadata_batches.has_value();
    }

    template <typename T>
    static std::vector<T> resolve_property(const std::optional<std::vector<T>>& property, size_t batch_size) {
        const auto default_value = std::vector<T>(batch_size);
        if (!property.has_value() || property->empty()) {
            return default_value;
        }
        return property.value();
    }
};


CBGenerateProperties extract_cb_generate_properties(const ov::AnyMap& config_map);


 /**
 * @brief Wrap each inner vector<T> into ov::Any, producing ov::AnyVector.
 * Required because ov::Any cannot store std::vector<std::vector<T>> directly
 * due to the lack of T::operator== (e.g. for ov::Tensor).
 */
template <typename T>
ov::AnyVector wrap_vectors_to_any(const std::vector<std::vector<T>>& nested_vectors) {
    ov::AnyVector any_vector;
    any_vector.reserve(nested_vectors.size());
    for (const auto& inner_vector : nested_vectors) {
        any_vector.push_back(ov::Any::make<std::vector<T>>(inner_vector));
    }
    return any_vector;
}

template <typename T>
std::vector<std::vector<T>> unwrap_vectors_from_any(const ov::AnyVector& any_vector) {
    std::vector<std::vector<T>> nested_vectors;
    nested_vectors.reserve(any_vector.size());
    for (const auto& any : any_vector) {
        OPENVINO_ASSERT(any.is<std::vector<T>>(), "Each ov::Any entry should hold std::vector<T>");
        nested_vectors.push_back(any.as<std::vector<T>>());
    }
    return nested_vectors;
}

}  // namespace ov::genai
