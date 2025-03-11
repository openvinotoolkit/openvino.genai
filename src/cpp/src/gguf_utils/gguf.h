#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <variant>

#include "openvino/openvino.hpp"

extern "C" {
#include <gguflib.h>
}

using GGUFMetaData =
    std::variant<std::monostate, ov::Tensor, std::string, std::vector<std::string>>;
using GGUFLoad = std::pair<
    std::unordered_map<std::string, ov::Tensor>,
    std::unordered_map<std::string, GGUFMetaData>>;

ov::Shape get_shape(const gguf_tensor& tensor);
void gguf_load_quantized(
    std::unordered_map<std::string, ov::Tensor>& a,
    const gguf_tensor& tensor);

GGUFLoad load_gguf(const std::string& file);