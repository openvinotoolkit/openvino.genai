// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "helper.hpp"
#include "openvino/op/concat.hpp"

std::shared_ptr<ov::Model> get_dummy_model(ov::Core core, size_t num_layers) {
    ov::NodeVector keys, values;
    ov::ParameterVector params;
    ov::element::Type kv_cache_type = core.get_property("CPU", ov::hint::kv_cache_precision);

    auto shape = ov::PartialShape::dynamic(4);
    shape[1] = 12;
    shape[2] = 64;
    shape[3] = 64;

    for (size_t i = 0; i < num_layers; i++) {
        auto key = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, shape);
        key->get_output_tensor(0).set_names({"key_cache." + std::to_string(i)});
        value->get_output_tensor(0).set_names({"value_cache." + std::to_string(i)});
        keys.push_back(key);
        values.push_back(value);
        params.push_back(key);
        params.push_back(value);
    }
    const auto& concat1 = std::make_shared<ov::op::v0::Concat>(keys, 1);
    const auto& concat2 = std::make_shared<ov::op::v0::Concat>(values, 1);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{concat1, concat2}, params);
    return std::make_shared<ov::Model>(ov::NodeVector{concat1, concat2}, params);
}
