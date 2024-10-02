// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

namespace ov {
namespace genai {
namespace numpy_utils {

// https://gist.github.com/lorenzoriano/5414671
template <typename T, typename U>
std::vector<T> linspace(U start, U end, size_t num, bool endpoint = false) {
    std::vector<T> indices;
    if (num != 0) {
        if (num == 1)
            indices.push_back(static_cast<T>(start));
        else {
            if (endpoint)
                --num;

            U delta = (end - start) / static_cast<U>(num);
            for (size_t i = 0; i < num; i++)
                indices.push_back(static_cast<T>(start + delta * i));

            if (endpoint)
                indices.push_back(static_cast<T>(end));
        }
    }
    return indices;
}

}// namespace ov
}// namespace genai
}// namespace txt2img_utils
