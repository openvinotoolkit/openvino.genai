// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "llm_matmul_transformations.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

bool insert_slice_before_matmul(const ov::Output<ov::Node>& matched_matmul_out, int64_t slice_gather_dim) {
    auto matmul = matched_matmul_out.get_node_shared_ptr();
    if (matmul->input(0).get_partial_shape().rank().get_length() != 3) {
        return false;
    }
    auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-2});
    auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
    auto slice = std::make_shared<ov::op::v8::Slice>(matmul->input_value(0), start, stop, step, axis);
    matmul->input(0).replace_source_output(slice);
    return true;
}

bool insert_gather_before_matmul(const ov::Output<ov::Node>& matched_matmul_out, int64_t slice_gather_dim,
                                const std::shared_ptr<ov::Model>& model) {
    auto matmul = matched_matmul_out.get_node_shared_ptr();
    if (matmul->input(0).get_partial_shape().rank().get_length() != 3) {
        return false;
    }
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    indices->set_friendly_name("sampled_tokens_indices");
    indices->output(0).get_tensor().set_names({"sampled_tokens_indices"});
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
    auto gather = std::make_shared<ov::op::v8::Gather>(matmul->input_value(0), indices, axis);
    matmul->input(0).replace_source_output(gather);
    model->add_parameters({indices});
    return true;
}

}  // namespace

namespace ov {
namespace genai {

// ======================== Slice transformations ========================

SliceLastMatmul::SliceLastMatmul(bool pa_based_model) {
    int64_t dim = pa_based_model ? 0 : 1;
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
    auto res = opp::wrap_type<ov::op::v0::Result>({matmul});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        return insert_slice_before_matmul(node_to_output.at(matmul), dim);
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "SliceLastMatmul"), std::move(callback));
}

SliceLastMatmulAdd::SliceLastMatmulAdd(bool pa_based_model) {
    int64_t dim = pa_based_model ? 0 : 1;
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
    auto add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
    auto res = opp::wrap_type<ov::op::v0::Result>({add});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        return insert_slice_before_matmul(node_to_output.at(matmul), dim);
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "SliceLastMatmulAdd"), std::move(callback));
}

SliceLastMatmulTranspose::SliceLastMatmulTranspose(bool pa_based_model) {
    int64_t base_dim = pa_based_model ? 0 : 1;
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto res = opp::wrap_type<ov::op::v0::Result>({transpose});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_transpose = node_to_output.at(transpose).get_node_shared_ptr();
        auto order_const = ov::as_type_ptr<ov::op::v0::Constant>(matched_transpose->input_value(1).get_node_shared_ptr());
        if (!order_const) return false;
        auto order = order_const->get_axis_vector_val();
        int64_t dim = static_cast<int64_t>(order[base_dim]);
        return insert_slice_before_matmul(node_to_output.at(matmul), dim);
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "SliceLastMatmulTranspose"), std::move(callback));
}

SliceLastMatmulMultiply::SliceLastMatmulMultiply(bool pa_based_model) {
    int64_t dim = pa_based_model ? 0 : 1;
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
    auto div = opp::wrap_type<ov::op::v1::Divide>({matmul, opp::any_input()});
    auto tanh = opp::wrap_type<ov::op::v0::Tanh>({div});
    auto multiply = opp::wrap_type<ov::op::v1::Multiply>({tanh, opp::any_input()});
    auto res = opp::wrap_type<ov::op::v0::Result>({multiply});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        return insert_slice_before_matmul(node_to_output.at(matmul), dim);
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "SliceLastMatmulMultiply"), std::move(callback));
}

// ======================== Gather transformations ========================

GatherLastMatmul::GatherLastMatmul(std::shared_ptr<ov::Model> model, bool pa_based_model) {
    int64_t dim = pa_based_model ? 0 : 1;
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
    auto res = opp::wrap_type<ov::op::v0::Result>({matmul});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        return insert_gather_before_matmul(node_to_output.at(matmul), dim, model);
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "GatherLastMatmul"), std::move(callback));
}

GatherLastMatmulAdd::GatherLastMatmulAdd(std::shared_ptr<ov::Model> model, bool pa_based_model) {
    int64_t dim = pa_based_model ? 0 : 1;
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
    auto add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
    auto res = opp::wrap_type<ov::op::v0::Result>({add});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        return insert_gather_before_matmul(node_to_output.at(matmul), dim, model);
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "GatherLastMatmulAdd"), std::move(callback));
}

GatherLastMatmulTranspose::GatherLastMatmulTranspose(std::shared_ptr<ov::Model> model, bool pa_based_model) {
    int64_t base_dim = pa_based_model ? 0 : 1;
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto res = opp::wrap_type<ov::op::v0::Result>({transpose});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_transpose = node_to_output.at(transpose).get_node_shared_ptr();
        auto order_const = ov::as_type_ptr<ov::op::v0::Constant>(matched_transpose->input_value(1).get_node_shared_ptr());
        if (!order_const) return false;
        auto order = order_const->get_axis_vector_val();
        int64_t dim = static_cast<int64_t>(order[base_dim]);
        return insert_gather_before_matmul(node_to_output.at(matmul), dim, model);
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "GatherLastMatmulTranspose"), std::move(callback));
}

GatherLastMatmulMultiply::GatherLastMatmulMultiply(std::shared_ptr<ov::Model> model, bool pa_based_model) {
    int64_t dim = pa_based_model ? 0 : 1;
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
    auto div = opp::wrap_type<ov::op::v1::Divide>({matmul, opp::any_input()});
    auto tanh = opp::wrap_type<ov::op::v0::Tanh>({div});
    auto multiply = opp::wrap_type<ov::op::v1::Multiply>({tanh, opp::any_input()});
    auto res = opp::wrap_type<ov::op::v0::Result>({multiply});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        return insert_gather_before_matmul(node_to_output.at(matmul), dim, model);
    };
    register_matcher(std::make_shared<opp::Matcher>(res, "GatherLastMatmulMultiply"), std::move(callback));
}

}  // namespace genai
}  // namespace ov
