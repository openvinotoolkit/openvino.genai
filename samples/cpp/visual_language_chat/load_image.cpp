
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#include "load_image.hpp"

namespace fs = std::filesystem;

std::vector<ov::Tensor> utils::load_images(const std::filesystem::path& input_path, std::optional<int32_t> target_height, std::optional<int32_t> target_width) {
    if (input_path.empty() || !fs::exists(input_path)) {
        throw std::runtime_error{"Path to images is empty or does not exist."};
    }
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_images{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> images;
        for (const fs::path& dir_entry : sorted_images) {
            images.push_back(utils::load_image(dir_entry, target_height, target_width));
        }
        return images;
    }
    return {utils::load_image(input_path, target_height, target_width)};
}

ov::Tensor utils::load_image(const std::filesystem::path& image_path, std::optional<int32_t> target_height, std::optional<int32_t> target_width) {
    OPENVINO_ASSERT(target_height.has_value() == target_width.has_value(),
                    "target_height and target_width must be provided together.");
    if (target_height.has_value()) {
        OPENVINO_ASSERT(*target_height > 0 && *target_width > 0,
                        "target_height and target_width must be positive values.");
    }
    int x = 0, y = 0, channels_in_file = 0;
    constexpr int desired_channels = 3;
    unsigned char* image = stbi_load(
        image_path.string().c_str(),
        &x, &y, &channels_in_file, desired_channels);
    if (!image) {
        std::stringstream error_message;
        error_message << "Failed to load the image '" << image_path << "'";
        throw std::runtime_error{error_message.str()};
    }
    if (target_height.has_value()) {
        unsigned char* resized = static_cast<unsigned char*>(
            malloc(size_t(*target_width) * size_t(*target_height) * desired_channels));
        if (!resized) {
            stbi_image_free(image);
            throw std::runtime_error{"Failed to allocate memory for resized image."};
        }
        stbir_resize_uint8_linear(
            image, x, y, 0,
            resized, *target_width, *target_height, 0,
            static_cast<stbir_pixel_layout>(desired_channels));
        stbi_image_free(image);
        image = resized;
        x = *target_width;
        y = *target_height;
    }
    struct SharedImageAllocator {
        unsigned char* image;
        int channels, height, width;
        void* allocate(size_t bytes, size_t) const {
            if (image && channels * height * width == bytes) {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void*, size_t, size_t) noexcept {
            stbi_image_free(image);
            image = nullptr;
        }
        bool is_equal(const SharedImageAllocator& other) const noexcept {return this == &other;}
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(y), size_t(x), size_t(desired_channels)},
        SharedImageAllocator{image, desired_channels, y, x}
    );
}
