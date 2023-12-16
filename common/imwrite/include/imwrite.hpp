// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "openvino/runtime/tensor.hpp"

void imwrite(const std::string& name, ov::Tensor image);
