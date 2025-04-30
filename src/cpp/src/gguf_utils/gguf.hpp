// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <variant>
#include <cstdarg>
#include <unordered_map>
#include <string>
#include <stdio.h>
#include <assert.h>

#include "openvino/openvino.hpp"

extern "C" {
#include <gguflib.h>
}

using GGUFMetaData =
    std::variant<std::monostate, float, int, ov::Tensor, std::string, std::vector<std::string>>;

template<typename... Args>
std::string format(std::string fmt, Args... args);

ov::Shape get_shape(const gguf_tensor& tensor);

void gguf_load_quantized(std::unordered_map<std::string, ov::Tensor>& a,
    std::unordered_map<std::string, gguf_tensor_type>& qtype_map,
    const gguf_tensor& tensor);

std::tuple<std::map<std::string, GGUFMetaData>, std::unordered_map<std::string, ov::Tensor>, std::unordered_map<std::string, gguf_tensor_type>> load_gguf(const std::string& file);
