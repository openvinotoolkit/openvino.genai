// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <assert.h>
#include <stdio.h>

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <variant>

#include "openvino/openvino.hpp"
#include "openvino/genai/llm_pipeline.hpp"

extern "C" {
#include <gguflib.h>
}

using GGUFMetaData =
    std::variant<std::monostate, float, int, ov::Tensor, std::string, std::vector<std::string>, std::vector<int32_t>>;

using GGUFLoad = std::tuple<std::unordered_map<std::string, GGUFMetaData>,
                            std::unordered_map<std::string, ov::Tensor>,
                            std::unordered_map<std::string, gguf_tensor_type>>;

template <typename... Args>
std::string format(std::string fmt, Args... args);

ov::Shape get_shape(const gguf_tensor& tensor);

void gguf_load_quantized(std::unordered_map<std::string, ov::Tensor>& a,
                         std::unordered_map<std::string, gguf_tensor_type>& qtype_map,
                         const gguf_tensor& tensor);

// Quantization functions for requantization during GGUF load
void quantize_q4_0(const float* src,
                   ov::Tensor& weights_out,
                   ov::Tensor& scales_out,
                   ov::Tensor& biases_out,
                   int64_t n_elements,
                   int64_t block_size);

void quantize_q8_0(const float* src,
                   ov::Tensor& weights_out,
                   ov::Tensor& scales_out,
                   ov::Tensor& biases_out,
                   int64_t n_elements,
                   int64_t block_size);

std::tuple<std::map<std::string, GGUFMetaData>,
           std::unordered_map<std::string, ov::Tensor>,
           std::unordered_map<std::string, gguf_tensor_type>>
load_gguf(const std::string& file, const ov::genai::OVModelQuantizeMode& quantize_mode = ov::genai::OVModelQuantizeMode::ORIGINAL);

GGUFLoad get_gguf_data(const std::string& file, const ov::genai::OVModelQuantizeMode& quantize_mode = ov::genai::OVModelQuantizeMode::ORIGINAL);
