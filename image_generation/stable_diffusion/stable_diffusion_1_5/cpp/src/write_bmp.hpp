// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief a header file with writeOutputBmp func from
 * samples/common.hpp
 * @file write_bmp.hpp
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <openvino/openvino.hpp>
// clang-format on

/**
 * @brief Writes output data to BMP image
 * @param name - image name
 * @param data - output data
 * @param height - height of the target image
 * @param width - width of the target image
 * @return false if error else true
 */

#ifndef UNUSED
#    if defined(_MSC_VER) && !defined(__clang__)
#        define UNUSED
#    else
#        define UNUSED __attribute__((unused))
#    endif
#endif

static UNUSED void writeOutputBmp(std::string name, unsigned char* data, size_t height, size_t width) {
    std::ofstream outFile;
    outFile.open(name, std::ofstream::binary);
    if (!outFile.is_open()) {
        std::cout << "fail to open the output BMP image path\n";
    }

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

    OPENVINO_ASSERT(
        height < (size_t)std::numeric_limits<int32_t>::max && width < (size_t)std::numeric_limits<int32_t>::max,
        "File size is too big: ",
        height,
        " X ",
        width);

    int padSize = static_cast<int>(4 - (width * 3) % 4) % 4;
    int sizeData = static_cast<int>(width * height * 3 + height * padSize);
    int sizeAll = sizeData + sizeof(file) + sizeof(info);

    file[2] = (unsigned char)(sizeAll);
    file[3] = (unsigned char)(sizeAll >> 8);
    file[4] = (unsigned char)(sizeAll >> 16);
    file[5] = (unsigned char)(sizeAll >> 24);

    info[4] = (unsigned char)(width);
    info[5] = (unsigned char)(width >> 8);
    info[6] = (unsigned char)(width >> 16);
    info[7] = (unsigned char)(width >> 24);

    int32_t negativeHeight = -(int32_t)height;
    info[8] = (unsigned char)(negativeHeight);
    info[9] = (unsigned char)(negativeHeight >> 8);
    info[10] = (unsigned char)(negativeHeight >> 16);
    info[11] = (unsigned char)(negativeHeight >> 24);

    info[20] = (unsigned char)(sizeData);
    info[21] = (unsigned char)(sizeData >> 8);
    info[22] = (unsigned char)(sizeData >> 16);
    info[23] = (unsigned char)(sizeData >> 24);

    outFile.write(reinterpret_cast<char*>(file), sizeof(file));
    outFile.write(reinterpret_cast<char*>(info), sizeof(info));

    unsigned char pad[3] = {0, 0, 0};

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            unsigned char pixel[3];
            pixel[0] = data[y * width * 3 + x * 3];
            pixel[1] = data[y * width * 3 + x * 3 + 1];
            pixel[2] = data[y * width * 3 + x * 3 + 2];

            outFile.write(reinterpret_cast<char*>(pixel), 3);
        }
        outFile.write(reinterpret_cast<char*>(pad), padSize);
    }
}