// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/txt2img_core.hpp"

namespace ov {
namespace genai {
namespace txt2img_core {

ov::Core singleton_core() {
    static ov::Core core;
    return core;
}
    
} // namespace genai
} // namespace ov
} // namespace txt2img_core
