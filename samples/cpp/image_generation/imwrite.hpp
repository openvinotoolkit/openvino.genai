// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "openvino/runtime/tensor.hpp"

/**
 * @brief Writes multiple images (depending on `image` tensor batch size) to BPM file(s)
 * @param name File name or pattern to use to write images
 * @param image Image(s) tensor
 * @param convert_bgr2rgb Convert BGR to RGB
 */
void imwrite(const std::string& name, ov::Tensor images, bool convert_bgr2rgb);
