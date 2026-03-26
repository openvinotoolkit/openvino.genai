// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/generate_properties.hpp"
#include "utils.hpp"

namespace ov::genai {

std::pair<std::string, ov::Any> images_batches(
    const std::vector<std::vector<ov::Tensor>>& images_batches
) {
    return {utils::IMAGES_BATCHES_ARG_NAME, wrap_vectors_to_any(images_batches)};
}

std::pair<std::string, ov::Any> videos_batches(
    const std::vector<std::vector<ov::Tensor>>& videos_batches
) {
    return {utils::VIDEOS_BATCHES_ARG_NAME, wrap_vectors_to_any(videos_batches)};
}

std::pair<std::string, ov::Any> videos_metadata_batches(
    const std::vector<std::vector<VideoMetadata>>& videos_metadata_batches
) {
    return {utils::VIDEOS_METADATA_BATCHES_ARG_NAME, wrap_vectors_to_any(videos_metadata_batches)};
}

CBGenerateProperties extract_cb_generate_properties(const ov::AnyMap& properties_map, size_t batch_size) {
    CBGenerateProperties properties;

    properties.streamer = utils::get_streamer_from_map(properties_map);

    auto generation_config_batches_iter = properties_map.find(ov::genai::generation_config_batches.name());
    if (generation_config_batches_iter != properties_map.end()) {
        OPENVINO_ASSERT(generation_config_batches_iter->second.is<std::vector<GenerationConfig>>(),
            "generation_config_batches property has to be of type std::vector<GenerationConfig>");
        properties.generation_config_batches = generation_config_batches_iter->second.as<std::vector<GenerationConfig>>();
    } else {
        properties.generation_config_batches = std::vector<GenerationConfig>(batch_size);
    }

    auto images_batches_iter = properties_map.find(utils::IMAGES_BATCHES_ARG_NAME);
    if (images_batches_iter != properties_map.end()) {
        properties.images_batches = unwrap_vectors_from_any<ov::Tensor>(images_batches_iter->second.as<ov::AnyVector>());
    } else {
        properties.images_batches = std::vector<std::vector<ov::Tensor>>(batch_size);
    }

    auto videos_batches_iter = properties_map.find(utils::VIDEOS_BATCHES_ARG_NAME);
    if (videos_batches_iter != properties_map.end()) {
        properties.videos_batches = unwrap_vectors_from_any<ov::Tensor>(videos_batches_iter->second.as<ov::AnyVector>());
    } else {
        properties.videos_batches = std::vector<std::vector<ov::Tensor>>(batch_size);
    }

    auto videos_metadata_batches_iter = properties_map.find(utils::VIDEOS_METADATA_BATCHES_ARG_NAME);
    if (videos_metadata_batches_iter != properties_map.end()) {
        properties.videos_metadata_batches = unwrap_vectors_from_any<VideoMetadata>(
            videos_metadata_batches_iter->second.as<ov::AnyVector>()
        );
    } else {
        properties.videos_metadata_batches = std::vector<std::vector<VideoMetadata>>(batch_size);
    }

    return properties;
}

} // namespace ov::genai
