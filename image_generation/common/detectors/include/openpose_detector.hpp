// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/runtime/tensor.hpp>

#include "openvino/runtime/compiled_model.hpp"

class OpenposeDetector {
public:
    OpenposeDetector() = default;

    void load(const std::string&);
    std::pair<ov::Tensor, ov::Tensor> inference(const ov::Tensor&);

    void forward(const ov::Tensor&,
                 std::vector<std::vector<float>>& subset,
                 std::vector<std::vector<float>>& candidate);

private:
    ov::CompiledModel body_model;
    static const std::vector<std::vector<int>> limbSeq;
    static const std::vector<std::vector<int>> mapIdx;

    // find the peaks from heatmap, returns a vector of tuple
    // (x, y, score, id)
    void find_heatmap_peaks(const ov::Tensor& heatmap_avg /* f32 */,
                            float thre1,
                            std::vector<std::vector<std::tuple<int, int, float, int>>>& all_peaks);

    void calculate_connections(const ov::Tensor& paf_avg,
                               const std::vector<std::vector<std::tuple<int, int, float, int>>>& all_peaks,
                               const ov::Tensor& oriImg,
                               const float thre2,
                               std::vector<std::vector<std::tuple<int, int, float, int, int>>>& connection_all,
                               std::vector<int>& special_k);

    void process_connections(const std::vector<std::vector<std::tuple<int, int, float, int>>>& all_peaks,
                             const std::vector<std::vector<std::tuple<int, int, float, int, int>>>& connection_all,
                             const std::vector<int>& special_k,
                             std::vector<std::vector<float>>& subset,
                             std::vector<std::vector<float>>& candidate);
};
