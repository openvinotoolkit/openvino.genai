
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "load_image.hpp"

ov::Tensor utils::load_image(const std::filesystem::path& image_path) {
    int x = 0, y = 0, channels_in_file = 0;
    constexpr int desired_channels = 3;
    unsigned char* image = stbi_load(
        image_path.string().c_str(),
        &x, &y, &channels_in_file, desired_channels);
    if (!image) {
        throw std::runtime_error{"Failed to load the image."};
    }
    struct SharedImageAllocator {
        unsigned char* image;
        int channels, height, width;
        void* allocate(size_t bytes, size_t) const {
            if (channels * height * width == bytes) {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void*, size_t bytes, size_t) {
            if (channels * height * width != bytes) {
                throw std::runtime_error{"Unexpected number of bytes was requested to deallocate."};
            }
            std::free(image);
            image = nullptr;
        }
        bool is_equal(const SharedImageAllocator& other) const noexcept {return this == &other;}
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(desired_channels), size_t(y), size_t(x)},
        SharedImageAllocator{image, desired_channels, y, x}
    );
}
