// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvino/pass/manager.hpp>
#include "tokenizer/add_second_input_transformation.hpp"
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/select.hpp>
#include <openvino/op/equal.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/multiply.hpp>
#include <memory>


using namespace ov::genai;
using namespace ov;
using namespace ov::op;

TEST(AddSecondInputTest, add_second_input_test) {
    std::shared_ptr<Model> model;

    {
        auto parameter_1 = std::make_shared<v0::Parameter>(element::f32, Shape{2, 3});
        auto parameter_2 = std::make_shared<v0::Parameter>(element::i32, Shape{2, 2});
        auto axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>({0}));
        model = std::make_shared<ov::Model>(OutputVector{axis}, ParameterVector{parameter_1, parameter_2});
    }

    {


    }
}
