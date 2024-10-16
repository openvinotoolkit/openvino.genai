// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/core.hpp"

namespace ov {
namespace genai {
namespace txt2img_core {

ov::Core singleton_core();
    
} // namespace genai
} // namespace ov
} // namespace txt2img_core
