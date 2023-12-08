// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief a header file for ov extension tokenizer
 * @file utils.hpp
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstring>
#include <iostream>
#include <sstream>

std::string ByteArrayToString(const uint8_t* arr, int size) {
    std::ostringstream convert;

    for (int a = 0; a < size; a++) {
        convert << arr[a];
    }

    return convert.str();
}

void StringToByteArray_vec(std::string s, std::vector<uint8_t>& bytes, int batch_size, int offset, int length) {
    bytes.resize(4 + 4 + 4 + s.length());

    std::memcpy(bytes.data() + 4 + 4 + 4, s.data(), s.length());

    bytes[0] = static_cast<uint8_t>(batch_size & 0x000000ff);
    bytes[1] = static_cast<uint8_t>((batch_size & 0x0000ff00) >> 8);
    bytes[2] = static_cast<uint8_t>((batch_size & 0x00ff0000) >> 16);
    bytes[3] = static_cast<uint8_t>((batch_size & 0xff000000) >> 24);

    bytes[4] = static_cast<uint8_t>(offset & 0x000000ff);
    bytes[5] = static_cast<uint8_t>((offset & 0x0000ff00) >> 8);
    bytes[6] = static_cast<uint8_t>((offset & 0x00ff0000) >> 16);
    bytes[7] = static_cast<uint8_t>((offset & 0xff000000) >> 24);

    bytes[8] = static_cast<uint8_t>(length & 0x000000ff);
    bytes[9] = static_cast<uint8_t>((length & 0x0000ff00) >> 8);
    bytes[10] = static_cast<uint8_t>((length & 0x00ff0000) >> 16);
    bytes[11] = static_cast<uint8_t>((length & 0xff000000) >> 24);
}

#endif