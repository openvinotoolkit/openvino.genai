// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"

namespace {

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::as_type_ptr<T>(op))
            count++;
    }
    return count;
}

template <typename T>
bool has_op_before_matmul(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op)) {
            if (ov::as_type_ptr<T>(matmul->input_value(0).get_node_shared_ptr()))
                return true;
        }
    }
    return false;
}

// Build: Parameter[input_shape] x Constant[hidden, vocab] -> MatMul -> Result
std::shared_ptr<ov::Model> make_matmul_model(const ov::PartialShape& input_shape, size_t hidden, size_t vocab) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{hidden, vocab}, 0.0f);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

// Build: Parameter[input_shape] x Constant[hidden, vocab] -> MatMul -> Add(bias) -> Result
std::shared_ptr<ov::Model> make_matmul_add_model(const ov::PartialShape& input_shape, size_t hidden, size_t vocab) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{hidden, vocab}, 0.0f);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{vocab}, 0.0f);
    auto add = std::make_shared<ov::op::v1::Add>(matmul, bias);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

// Build: Parameter -> MatMul -> Transpose -> Result
std::shared_ptr<ov::Model> make_matmul_transpose_model(const ov::PartialShape& input_shape, size_t hidden, size_t vocab) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{hidden, vocab}, 0.0f);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);
    auto order = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, order);
    auto result = std::make_shared<ov::op::v0::Result>(transpose);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

// Build: Parameter -> MatMul -> Divide -> Tanh -> Multiply -> Result
std::shared_ptr<ov::Model> make_matmul_div_tanh_mul_model(const ov::PartialShape& input_shape, size_t hidden, size_t vocab) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{hidden, vocab}, 0.0f);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);
    auto divisor = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 2.0f);
    auto divide = std::make_shared<ov::op::v1::Divide>(matmul, divisor);
    auto tanh = std::make_shared<ov::op::v0::Tanh>(divide);
    auto multiplier = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1.0f);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(tanh, multiplier);
    auto result = std::make_shared<ov::op::v0::Result>(multiply);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

// Build model with specific weight values for correctness tests
std::shared_ptr<ov::Model> make_matmul_model_with_weights(const ov::PartialShape& input_shape,
                                                          size_t hidden, size_t vocab,
                                                          const std::vector<float>& weights_data) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    input->output(0).get_tensor().set_names({"input"});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{hidden, vocab}, weights_data.data());
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

} // namespace

// ===================== apply_slice_before_matmul_transformation =====================

// Graph-level: all 4 supported patterns
TEST(SliceBeforeMatmul, AllPatterns) {
    // MatMul -> Result
    {
        auto model = make_matmul_model(ov::PartialShape{1, -1, 64}, 64, 128);
        ASSERT_EQ(count_ops<ov::op::v8::Slice>(model), 0u);
        ov::genai::utils::apply_slice_before_matmul_transformation(model);
        ASSERT_EQ(count_ops<ov::op::v8::Slice>(model), 1u);
        ASSERT_TRUE((has_op_before_matmul<ov::op::v8::Slice>(model)));
    }
    // MatMul -> Add -> Result
    {
        auto model = make_matmul_add_model(ov::PartialShape{1, -1, 64}, 64, 128);
        ov::genai::utils::apply_slice_before_matmul_transformation(model);
        ASSERT_EQ(count_ops<ov::op::v8::Slice>(model), 1u);
        ASSERT_TRUE((has_op_before_matmul<ov::op::v8::Slice>(model)));
    }
    // MatMul -> Transpose -> Result
    {
        auto model = make_matmul_transpose_model(ov::PartialShape{1, -1, 64}, 64, 128);
        ov::genai::utils::apply_slice_before_matmul_transformation(model);
        ASSERT_EQ(count_ops<ov::op::v8::Slice>(model), 1u);
        ASSERT_TRUE((has_op_before_matmul<ov::op::v8::Slice>(model)));
    }
    // MatMul -> Divide -> Tanh -> Multiply -> Result
    {
        auto model = make_matmul_div_tanh_mul_model(ov::PartialShape{1, -1, 64}, 64, 128);
        ov::genai::utils::apply_slice_before_matmul_transformation(model);
        ASSERT_EQ(count_ops<ov::op::v8::Slice>(model), 1u);
        ASSERT_TRUE((has_op_before_matmul<ov::op::v8::Slice>(model)));
    }
}

// Inference: sliced output matches the last token of non-transformed output
TEST(SliceBeforeMatmul, InferenceLastTokenMatchesOriginal) {
    const size_t batch = 1, seq_len = 5, hidden = 16, vocab = 32;

    std::vector<float> input_data(batch * seq_len * hidden);
    for (size_t i = 0; i < input_data.size(); ++i)
        input_data[i] = static_cast<float>(i) * 0.01f;

    std::vector<float> weights_data(hidden * vocab);
    for (size_t i = 0; i < weights_data.size(); ++i)
        weights_data[i] = static_cast<float>(i % 7) * 0.1f;

    ov::Core core;

    // Full model (no transformation)
    auto model_full = make_matmul_model_with_weights(ov::PartialShape{1, -1, hidden}, hidden, vocab, weights_data);
    auto req_full = core.compile_model(model_full, "CPU").create_infer_request();
    req_full.set_input_tensor(ov::Tensor(ov::element::f32, ov::Shape{batch, seq_len, hidden}, input_data.data()));
    req_full.infer();
    const float* full_data = req_full.get_output_tensor().data<float>();

    // Sliced model
    auto model_sliced = make_matmul_model_with_weights(ov::PartialShape{1, -1, hidden}, hidden, vocab, weights_data);
    ov::genai::utils::apply_slice_before_matmul_transformation(model_sliced);
    auto req_sliced = core.compile_model(model_sliced, "CPU").create_infer_request();
    req_sliced.set_input_tensor(ov::Tensor(ov::element::f32, ov::Shape{batch, seq_len, hidden}, input_data.data()));
    req_sliced.infer();
    ASSERT_EQ(req_sliced.get_output_tensor().get_shape()[1], 1u);
    const float* sliced_data = req_sliced.get_output_tensor().data<float>();

    auto output_shape = req_sliced.get_output_tensor().get_shape();
    ASSERT_EQ(output_shape, (ov::Shape{batch, 1, vocab}));

    // Sliced output must equal the last token of the full output
    const size_t last_token_offset = (seq_len - 1) * vocab;
    for (size_t i = 0; i < vocab; ++i) {
        ASSERT_FLOAT_EQ(sliced_data[i], full_data[last_token_offset + i])
            << "Mismatch at vocab index " << i;
    }
}

// ===================== apply_gather_before_matmul_transformation =====================

// Graph-level: all 4 supported patterns
TEST(GatherBeforeMatmul, AllPatterns) {
    // MatMul -> Result
    {
        auto model = make_matmul_model(ov::PartialShape{1, -1, 64}, 64, 128);
        ASSERT_EQ(count_ops<ov::op::v8::Gather>(model), 0u);
        ov::genai::utils::apply_gather_before_matmul_transformation(model);
        ASSERT_EQ(count_ops<ov::op::v8::Gather>(model), 1u);
        ASSERT_TRUE((has_op_before_matmul<ov::op::v8::Gather>(model)));
    }
    // MatMul -> Add -> Result
    {
        auto model = make_matmul_add_model(ov::PartialShape{1, -1, 64}, 64, 128);
        ov::genai::utils::apply_gather_before_matmul_transformation(model);
        ASSERT_EQ(count_ops<ov::op::v8::Gather>(model), 1u);
        ASSERT_TRUE((has_op_before_matmul<ov::op::v8::Gather>(model)));
    }
    // MatMul -> Transpose -> Result
    {
        auto model = make_matmul_transpose_model(ov::PartialShape{1, -1, 64}, 64, 128);
        ov::genai::utils::apply_gather_before_matmul_transformation(model);
        ASSERT_EQ(count_ops<ov::op::v8::Gather>(model), 1u);
        ASSERT_TRUE((has_op_before_matmul<ov::op::v8::Gather>(model)));
    }
    // MatMul -> Divide -> Tanh -> Multiply -> Result
    {
        auto model = make_matmul_div_tanh_mul_model(ov::PartialShape{1, -1, 64}, 64, 128);
        ov::genai::utils::apply_gather_before_matmul_transformation(model);
        ASSERT_EQ(count_ops<ov::op::v8::Gather>(model), 1u);
        ASSERT_TRUE((has_op_before_matmul<ov::op::v8::Gather>(model)));
    }
}

// Inference: gathered output for specific indices matches corresponding tokens in full output
TEST(GatherBeforeMatmul, InferenceGatheredTokensMatchOriginal) {
    const size_t batch = 1, seq_len = 5, hidden = 16, vocab = 32;

    std::vector<float> input_data(batch * seq_len * hidden);
    for (size_t i = 0; i < input_data.size(); ++i)
        input_data[i] = static_cast<float>(i) * 0.01f;

    std::vector<float> weights_data(hidden * vocab);
    for (size_t i = 0; i < weights_data.size(); ++i)
        weights_data[i] = static_cast<float>(i % 7) * 0.1f;

    ov::Core core;

    // Full model (no transformation)
    auto model_full = make_matmul_model_with_weights(ov::PartialShape{1, -1, hidden}, hidden, vocab, weights_data);
    auto req_full = core.compile_model(model_full, "CPU").create_infer_request();
    req_full.set_input_tensor(ov::Tensor(ov::element::f32, ov::Shape{batch, seq_len, hidden}, input_data.data()));
    req_full.infer();
    const float* full_data = req_full.get_output_tensor().data<float>();

    // Gathered model — select tokens at indices {1, 3}
    auto model_gather = make_matmul_model_with_weights(ov::PartialShape{1, -1, hidden}, hidden, vocab, weights_data);
    ov::genai::utils::apply_gather_before_matmul_transformation(model_gather);
    auto req_gather = core.compile_model(model_gather, "CPU").create_infer_request();
    req_gather.set_tensor("input", ov::Tensor(ov::element::f32, ov::Shape{batch, seq_len, hidden}, input_data.data()));

    std::vector<int64_t> indices = {1, 3};
    auto indices_tensor = ov::Tensor(ov::element::i64, ov::Shape{indices.size()}, indices.data());
    req_gather.set_tensor("sampled_tokens_indices", indices_tensor);

    req_gather.infer();
    auto out_shape = req_gather.get_output_tensor().get_shape();
    ASSERT_EQ(out_shape, (ov::Shape{batch, indices.size(), vocab}));

    const float* gathered_data = req_gather.get_output_tensor().data<float>();
    for (size_t idx = 0; idx < indices.size(); ++idx) {
        size_t token_pos = static_cast<size_t>(indices[idx]);
        for (size_t v = 0; v < vocab; ++v) {
            ASSERT_FLOAT_EQ(gathered_data[idx * vocab + v], full_data[token_pos * vocab + v])
                << "Mismatch at gathered index " << idx << " (token " << token_pos << "), vocab " << v;
        }
    }
}
