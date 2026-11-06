// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "classification/activation.hpp"

#include <algorithm>
#include <cmath>

#include "openvino/core/except.hpp"

namespace {

void apply_softmax(std::vector<float>& scores) {
    const float max_logit = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& score : scores) {
        score = std::exp(score - max_logit);
        sum += score;
    }
    for (float& score : scores) {
        score /= sum;
    }
}

void apply_sigmoid(std::vector<float>& scores) {
    for (float& score : scores) {
        score = 1.0f / (1.0f + std::exp(-score));
    }
}

void apply_activation_inplace(std::vector<float>& scores, ov::genai::ClassificationActivation activation_type) {
    switch (activation_type) {
    case ov::genai::ClassificationActivation::SOFTMAX:
        apply_softmax(scores);
        return;
    case ov::genai::ClassificationActivation::SIGMOID:
        apply_sigmoid(scores);
        return;
    case ov::genai::ClassificationActivation::NONE:
        return;
    }
    OPENVINO_THROW("Unsupported classification activation");
}

}  // namespace

namespace ov::genai {

/// Mirrors transformers' TextClassificationPipeline default.
ClassificationActivation default_activation(const std::string& problem_type, size_t num_labels) {
    if (problem_type == "regression") {
        return ClassificationActivation::NONE;
    }
    if (problem_type == "multi_label_classification" || num_labels == 1) {
        return ClassificationActivation::SIGMOID;
    }
    return ClassificationActivation::SOFTMAX;
}

std::vector<std::vector<float>> apply_activation(const ov::Tensor& logits, ClassificationActivation activation_type) {
    const ov::Shape& shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "Expected logits of shape [batch, num_labels], got ", shape);
    const size_t num_labels = shape[1];
    const float* data = logits.data<const float>();

    std::vector<std::vector<float>> scores;
    scores.reserve(shape[0]);
    for (size_t row = 0; row < shape[0]; ++row) {
        apply_activation_inplace(scores.emplace_back(data + row * num_labels, data + (row + 1) * num_labels),
                                 activation_type);
    }
    return scores;
}

}  // namespace ov::genai
