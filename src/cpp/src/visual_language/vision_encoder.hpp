// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include "openvino/runtime/infer_request.hpp"

#include "visual_language/vlm_config.hpp"
#include "visual_language/processor_config.hpp"
#include "circular_buffer_queue.hpp"

namespace ov::genai {
/// @brief A pair describing image size.
struct ImageSize {
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
    /// [N, H*W, hidden_size]. [N, 1014, 1152] is a possible example for
    /// openbmb/MiniCPM-V-2. Only batch 1 is supported.
    ov::Tensor resized_source;
    /// @brief A size of an image used to compute embeddings for
    /// divided by ProcessorConfig's patch_size.
    ImageSize resized_source_size;
    /// @brief Embeddings of images obtained from a source image by
    /// slicing at no more than max_slice_nums pieces and resizing.
    /// The tensor's shape is
    /// [slice_y, slice_x, number_of_embeddings, embedding_size].
    /// slices_sizes.size() == slice_y * slice_x.
    ov::Tensor slices;
    /// @brief A size of images used to compute embeddings
    /// stored in slices member divided by ProcessorConfig's patch_size.
    ImageSize slices_size;

    /// @brief Patches grid after llava_next preprocessing.
    /// Format: [num_patches_height, num_patches_width]
    std::pair<int, int> patches_grid;
    
    /// @brief Original size of the image
    ImageSize original_image_size;
};

/// @brief A class used to infer embeddings of an image using
/// ov::InferRequest and configured by ProcessorConfig.
class VisionEncoder {
public:
    using Ptr = std::shared_ptr<VisionEncoder>;

    /// @brief Constructs the encoder from model_dir.
    /// @param model_dir A folder containing openvino_vision_embeddings_model.xml and
    /// preprocessor_config.json.
    /// @param model_type A type of VLM model.
    /// @param device A device to compile the encoder for.
    /// @param properties A config to be passed to
    /// ov::Core::compile_model().
    static VisionEncoder::Ptr create(
        const std::filesystem::path& model_dir,
        const VLMModelType model_type,
        const std::string& device,
        const ov::AnyMap properties = {});

    /// @brief Constructs the encoder from models map.
    /// @param model Model IR as string (openvino_vision_embeddings_model.xml)
    /// @param weights Model weights as tensor (openvino_vision_embeddings_model.bin)
    /// @param config_dir_path A path to directory containing preprocessor_config.json.
    /// @param model_type A type of VLM model.
    /// @param device A device to compile the encoder for.
    /// @param properties A config to be passed to
    /// ov::Core::compile_model().
    static VisionEncoder::Ptr create(
        const std::string& model,
        const ov::Tensor& weights,
        const std::filesystem::path& config_dir_path,
        const VLMModelType model_type,
        const std::string& device,
        const ov::AnyMap properties = {});

    /// @brief Compute embeddings of an image given
    /// ProcessorConfig members.
    /// @param image An image to infer embeddings for. Image shape must be
    /// [1CHW]. Only batch 1 is supported.
    /// @param config_map A config or its members values to follow
    /// instead of the config obtained in constructors.
    /// @return Resulting embeddings for the resized source image and
    /// its slices.
    virtual EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map = {}) = 0;

    /// @brief Gets processor config
    /// @return Processor config
    ProcessorConfig get_processor_config() const;

protected:
    /// @brief  Infer requests queue for image encoding model.
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_encoder;

    /// @brief A config to follow.
    ProcessorConfig m_processor_config;

public:
    VisionEncoder(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoder(
        const std::string& model,
        const ov::Tensor& weights,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);
};

} // namespace ov::genai
