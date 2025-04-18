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

enum class QType { FP16 = 0, INT8 = 1, INT4 = 2 };

template<typename... Args>
std::string format(std::string fmt, Args... args);

ov::Shape get_shape(const gguf_tensor& tensor);

void gguf_load_quantized(std::unordered_map<std::string, ov::Tensor>& a,
    const gguf_tensor& tensor);

std::pair<std::map<std::string, GGUFMetaData>, std::unordered_map<std::string, ov::Tensor>> load_gguf(const std::string& file);
