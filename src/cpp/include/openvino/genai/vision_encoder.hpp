// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/processor_config.hpp"
#include <openvino/openvino.hpp>

namespace ov::genai {
/// @brief A pair describing image size.
struct HeightWidth {
    /// @brief Height of a corresponding image.
    size_t height;
    /// @brief Width of a corresponding image.
    size_t width;
};

/// @brief Embeddings of a given image. The number of slices is no
/// greater than ProcessorConfig's max_slice_nums.
struct EncodedImage {
    /// @brief Embeddings of a resized image based on ProcessorConfig's
    /// scale_resolution. The tensor's shape is
    /// [1, number_of_embeddings, embedding_size].
    ov::Tensor resized_source;
    /// @brief A size of an image used to compute embeddings for
    /// divided by ProcessorConfig's patch_size.
    HeightWidth resized_source_size;
    /// @brief Embeddings of images obtained from a source image by
    /// slicing at no more than max_slice_nums pieces and resizing.
    /// The tensor's shape is
    /// [slice_y, slice_x, number_of_embeddings, embedding_size].
    /// slices_sizes.size() == slice_y * slice_x.
    ov::Tensor slices;
    /// @brief Flattened sizes of images used to compute embeddings
    /// stored in slices member divided by ProcessorConfig's patch_size.
    std::vector<HeightWidth> slices_sizes;
};

class OPENVINO_GENAI_EXPORTS VisionEncoder {
public:
    ov::InferRequest m_encoder;
    ProcessorConfig m_processor_config;

    explicit VisionEncoder(
        const ov::InferRequest& encoder,
        const ProcessorConfig& processor_config=ProcessorConfig{}
    ) : m_encoder{encoder}, m_processor_config{processor_config} {}

    explicit VisionEncoder(
        const std::filesystem::path& model_dir,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    );

    EncodedImage encode(const ov::Tensor& image) {
        return encode(image, m_processor_config);
    }

    EncodedImage encode(
        const ov::Tensor& image, const ProcessorConfig& config
    );

    EncodedImage encode(
        const ov::Tensor& image, const ov::AnyMap& config_map
    );

    template <typename... Properties>
    util::EnableIfAllStringAny<EncodedImage, Properties...> encode(
        const ov::Tensor& image,
        Properties&&... properties
    ) {
        return encode(
            image, AnyMap{std::forward<Properties>(properties)...}
        );
    }
};
}
