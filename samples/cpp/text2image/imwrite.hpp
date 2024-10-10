// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "openvino/runtime/tensor.hpp"

/**
 * @brief Writes image to file
 * @param name File name
 * @param image Image tensor
 * @param convert_bgr2rgb Convert BGR to RGB
 */
void imwrite(const std::string& name, ov::Tensor image, bool convert_bgr2rgb);
