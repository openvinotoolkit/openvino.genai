// Copyright © 2023-2024 Apple Inc.
#pragma once

#include "openvino/openvino.hpp"

extern "C" {
#include <gguflib.h>
}

ov::Shape get_shape(const gguf_tensor& tensor);
void gguf_load_quantized(
    std::unordered_map<std::string, ov::Tensor>& a,
    const gguf_tensor& tensor);

