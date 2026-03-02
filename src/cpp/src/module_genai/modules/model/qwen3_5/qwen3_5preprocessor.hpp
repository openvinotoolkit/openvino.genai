// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <filesystem>
#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include "openvino/runtime/tensor.hpp"
#include "qwen3_5config.hpp"

namespace ov::genai::module {

struct Qwen3_5PreprocessorOutput {
    ov::Tensor pixel_values;
    ov::Tensor grid_thw;
    ov::Tensor pos_embeds;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;
};

class Qwen3_5Preprocessor {
public:
    explicit Qwen3_5Preprocessor(const std::filesystem::path& model_path);

    Qwen3_5PreprocessorOutput preprocess(const ov::Tensor &images);
private:
    Qwen3_5VisionPreprocessConfig m_preprocess_config;
    Qwen3_5VisionConfig m_vision_config;
    ov::Tensor m_pos_embed_weight;

    void load_pos_embed_weight(const std::filesystem::path& model_path);

    static ov::element::Type parse_ov_dtype(const std::string& s);

    std::pair<size_t, size_t> smart_resize(size_t height,
                                                  size_t width,
                                                  size_t factor);
    
    void resize_bilinear_to_chw(const uint8_t *src,
                                       size_t src_h,
                                       size_t src_w,
                                       size_t channels,
                                       bool nchw,
                                       size_t dst_h,
                                       size_t dst_w,
                                       std::vector<float> &dst_chw);

    ov::Tensor build_pos_embeds(const ov::Tensor &grid_thw);

    static ov::Tensor to_f32(const ov::Tensor& src);

    std::pair<ov::Tensor, ov::Tensor> build_rotary_cos_sin(const ov::Tensor &grid_thw);
};

}
