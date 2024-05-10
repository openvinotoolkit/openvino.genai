// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "openvino/genai/tokenizer.hpp"

namespace ov {

class StreamerBase {
public:
    virtual void put(int64_t token) = 0;

    virtual void end() = 0;
};

} // namespace ov
