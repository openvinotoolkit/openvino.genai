// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <openvino/runtime/tensor.hpp>

#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"

namespace ov::genai {

struct VisionProperties {
    std::optional<std::vector<ov::Tensor>> images;
    std::optional<std::vector<ov::Tensor>> videos;
    std::optional<std::vector<VideoMetadata>> videos_metadata;

    bool has_value() const {
        return images.has_value() || videos.has_value() || videos_metadata.has_value();
    }
};

inline VisionProperties extract_vision_properties(const ov::AnyMap& properties_map) {
    VisionProperties vision_properties;
    
    const auto image_it = properties_map.find(ov::genai::image.name());
    if (image_it != properties_map.end()) {
        vision_properties.images = {image_it->second.as<ov::Tensor>()};
    }

    const auto images_it = properties_map.find(ov::genai::images.name());
    if (images_it != properties_map.end()) {
        if (images_it->second.is<std::vector<ov::Tensor>>()) {
            const auto images = images_it->second.as<std::vector<ov::Tensor>>();
            vision_properties.images = images_it->second.as<std::vector<ov::Tensor>>();
        } else if (images_it->second.is<ov::Tensor>()) {
            vision_properties.images = {images_it->second.as<ov::Tensor>()};
        } else if (!images_it->second.empty()) {
            OPENVINO_THROW("Unknown images type.");
        }
    }

    const auto videos_it = properties_map.find(ov::genai::videos.name());
    if (videos_it != properties_map.end()) {
        if (videos_it->second.is<std::vector<ov::Tensor>>()) {
            vision_properties.videos = videos_it->second.as<std::vector<ov::Tensor>>();
        } else if (videos_it->second.is<ov::Tensor>()) {
            vision_properties.videos = {videos_it->second.as<ov::Tensor>()};
        } else if (!videos_it->second.empty()) {
            OPENVINO_THROW("Unknown videos type.");
        }
    }

    const auto videos_metadata_it = properties_map.find(ov::genai::videos_metadata.name());
    if (videos_metadata_it != properties_map.end()) {
        if (videos_metadata_it->second.is<std::vector<VideoMetadata>>()) {
            vision_properties.videos_metadata = videos_metadata_it->second.as<std::vector<VideoMetadata>>();
        } else if (videos_metadata_it->second.is<VideoMetadata>()) {
            vision_properties.videos_metadata = {videos_metadata_it->second.as<VideoMetadata>()};
        } else if (!videos_metadata_it->second.empty()) {
            OPENVINO_THROW("Unknown videos metadata type.");
        }
    }

    return vision_properties;
}

}  // namespace ov::genai
