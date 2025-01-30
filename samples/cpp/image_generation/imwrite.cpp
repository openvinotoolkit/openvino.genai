// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <limits>
#include <string>
#include <iostream>

#include "imwrite.hpp"

#include "openvino/core/except.hpp"

namespace {

unsigned char file[14] = {
    'B',
    'M',  // magic
    0,
    0,
    0,
    0,  // size in bytes
    0,
    0,  // app data
    0,
    0,  // app data
    40 + 14,
    0,
    0,
    0  // start of data offset
};

unsigned char info[40] = {
    40,
    0,
    0,
    0,  // info hd size
    0,
    0,
    0,
    0,  // width
    0,
    0,
    0,
    0,  // height
    1,
    0,  // number color planes
    24,
    0,  // bits per pixel
    0,
    0,
    0,
    0,  // compression is none
    0,
    0,
    0,
    0,  // image bits size
    0x13,
    0x0B,
    0,
    0,  // horz resolution in pixel / m
    0x13,
    0x0B,
    0,
    0,  // vert resolution (0x03C3 = 96 dpi, 0x0B13 = 72
        // dpi)
    0,
    0,
    0,
    0,  // #colors in palette
    0,
    0,
    0,
    0,  // #important colors
};

void imwrite_single_image(const std::string& name, ov::Tensor image, bool convert_bgr2rgb) {
    const ov::Shape shape = image.get_shape();
    const size_t width = shape[2], height = shape[1], channels = shape[3];
    OPENVINO_ASSERT(image.get_element_type() == ov::element::u8 &&
        shape.size() == 4 && shape[0] == 1 && channels == 3,
        "Image of u8 type and [1, H, W, 3] shape is expected.",
        "Given image has shape ", shape, " and element type ", image.get_element_type());

    std::ofstream output_file(name, std::ofstream::binary);
    OPENVINO_ASSERT(output_file.is_open(), "Failed to open the output BMP image path");

    int padSize = static_cast<int>(4 - (width * channels) % 4) % 4;
    int sizeData = static_cast<int>(width * height * channels + height * padSize);
    int sizeAll = sizeData + sizeof(file) + sizeof(info);

    file[2] = (unsigned char)(sizeAll);
    file[3] = (unsigned char)(sizeAll >> 8);
    file[4] = (unsigned char)(sizeAll >> 16);
    file[5] = (unsigned char)(sizeAll >> 24);

    info[4] = (unsigned char)(width);
    info[5] = (unsigned char)(width >> 8);
    info[6] = (unsigned char)(width >> 16);
    info[7] = (unsigned char)(width >> 24);

    std::int32_t negativeHeight = -(int32_t)height;
    info[8] = (unsigned char)(negativeHeight);
    info[9] = (unsigned char)(negativeHeight >> 8);
    info[10] = (unsigned char)(negativeHeight >> 16);
    info[11] = (unsigned char)(negativeHeight >> 24);

    info[20] = (unsigned char)(sizeData);
    info[21] = (unsigned char)(sizeData >> 8);
    info[22] = (unsigned char)(sizeData >> 16);
    info[23] = (unsigned char)(sizeData >> 24);

    output_file.write(reinterpret_cast<char*>(file), sizeof(file));
    output_file.write(reinterpret_cast<char*>(info), sizeof(info));

    const std::uint8_t pad[3] = {0, 0, 0};
    const std::uint8_t* data = image.data<const std::uint8_t>();

    for (size_t y = 0; y < height; y++) {
        const std::uint8_t* current_row = data + y * width * channels;
        if (convert_bgr2rgb) {
            for (size_t x = 0; x < width; ++x) {
                output_file.write(reinterpret_cast<const char*>(current_row + 2), 1);
                output_file.write(reinterpret_cast<const char*>(current_row + 1), 1);
                output_file.write(reinterpret_cast<const char*>(current_row), 1);
                current_row += channels;
            }
        } else {
            output_file.write(reinterpret_cast<const char*>(current_row), width * channels);
        }
        output_file.write(reinterpret_cast<const char*>(pad), padSize);
    }
}

} // namespace


void imwrite(const std::string& name, ov::Tensor images, bool convert_bgr2rgb) {
    const ov::Shape shape = images.get_shape();
    OPENVINO_ASSERT(images.get_element_type() == ov::element::u8 && shape.size() == 4,
        "Image of u8 type and [1, H, W, 3] shape is expected.",
        "Given image has shape ", shape, " and element type ", images.get_element_type());

    const ov::Shape img_shape = {1, shape[1], shape[2], shape[3]};
    uint8_t* img_data = images.data<uint8_t>();

    for (int img_num = 0, num_images = shape[0], img_size = ov::shape_size(img_shape); img_num < num_images; ++img_num, img_data += img_size) {
        char img_name[25];
        sprintf(img_name, name.c_str(), img_num);

        ov::Tensor image(images.get_element_type(), img_shape, img_data);
        imwrite_single_image(img_name, image, true);
    }
}
