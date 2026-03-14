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

extern "C" {
#ifdef HAS_LLAMA_CPP
    #include "llama.h"
    #include "ggml.h"
    #include "gguf.h" 
#else
    #include "gguflib.h"
#endif
}

#ifdef HAS_LLAMA_CPP
    using gguf_tensor_type = ggml_type;
    using gguf_tensor = ggml_tensor;
    using gguf_ctx = struct gguf_context;

    inline gguf_ctx* gguf_open(const char* fname) {
        struct gguf_init_params params = {true, nullptr};
        return gguf_init_from_file(fname, params);
    }

    inline void gguf_close(gguf_ctx* ctx) {
        if (ctx) gguf_free(ctx);
    }

    inline void load_arrays(gguf_ctx*, std::unordered_map<std::string, ov::Tensor>&, std::unordered_map<std::string, gguf_tensor_type>&) {}

    #define GGUF_TYPE_F32 GGML_TYPE_F32
    #define GGUF_TYPE_F16 GGML_TYPE_F16

#endif

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

std::tuple<std::map<std::string, GGUFMetaData>,
           std::unordered_map<std::string, ov::Tensor>,
           std::unordered_map<std::string, gguf_tensor_type>>
load_gguf(const std::string& file);

GGUFLoad get_gguf_data(const std::string& file);
