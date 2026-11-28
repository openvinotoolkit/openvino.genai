// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "openvino/genai/lora_adapter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/runtime/core.hpp"

namespace {

// A=[1,1], B=[2], so the LoRA contribution is twice the sum of the
// selected split output. Build a tiny safetensors file in memory.
ov::Tensor make_adapter() {
    std::string header = R"({"linear.lora_A.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"linear.lora_B.weight":{"dtype":"F32","shape":[1,1],"data_offsets":[8,12]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    const uint64_t length = header.size();
    const float weights[] = {1.f, 1.f, 2.f};
    ov::Tensor bytes(ov::element::u8, {sizeof(length) + header.size() + sizeof(weights)});
    auto* data = bytes.data<uint8_t>();
    std::memcpy(data, &length, sizeof(length));
    std::memcpy(data + sizeof(length), header.data(), header.size());
    std::memcpy(data + sizeof(length) + header.size(), weights, sizeof(weights));
    return bytes;
}

TEST(LoraMultiOutput, UsesTheActualSplitOutput) {
    for (auto mode : {ov::genai::AdapterConfig::MODE_STATIC, ov::genai::AdapterConfig::MODE_DYNAMIC}) {
        for (size_t port : {0u, 1u}) {
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
            auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 2});
            auto split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, lengths);
            auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 2}, {0.f, 0.f});
            auto linear = std::make_shared<ov::op::v0::MatMul>(split->output(port), weight, false, true);
            linear->set_friendly_name("linear");
            auto model = std::make_shared<ov::Model>(ov::OutputVector{linear}, ov::ParameterVector{input});

            ov::genai::Adapter adapter(make_adapter());
            ov::genai::AdapterConfig config(adapter, 1.f, mode);
            ov::genai::AdapterController controller(model, config, "CPU");
            ov::Core core;
            auto request = core.compile_model(model, "CPU").create_infer_request();
            if (mode == ov::genai::AdapterConfig::MODE_DYNAMIC) {
                controller.apply(request);
            }
            auto values = request.get_input_tensor();
            const float inputs[] = {1.f, 2.f, 3.f, 4.f};
            std::memcpy(values.data<float>(), inputs, sizeof(inputs));
            request.infer();
            EXPECT_FLOAT_EQ(request.get_output_tensor().data<const float>()[0], port == 0 ? 6.f : 14.f);
        }
    }
}

}  // namespace
