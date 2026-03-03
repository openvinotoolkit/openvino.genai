// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

namespace weights {
class WeightSource;
class WeightFinalizer;
}  // namespace weights

class WeightParameter {
public:
    using WeightLoaderFn = std::function<void(WeightParameter&,
                                              weights::WeightSource&,
                                              weights::WeightFinalizer&,
                                              const std::string& weight_name,
                                              const std::optional<int>& shard_id)>;

    WeightParameter(std::string name, OpContext* ctx);

    const std::string& name() const;
    OpContext* context() const;

    void set_weight_loader(WeightLoaderFn fn);
    const WeightLoaderFn* weight_loader() const;

    void bind(const Tensor& weight);
    bool is_bound() const;
    const Tensor& value() const;

    void add_shard(int shard_id, const Tensor& shard);
    void finalize();

    void tie_to(WeightParameter& other);
    void set_optional(bool optional);
    bool is_optional() const;

private:
    std::string name_;
    OpContext* ctx_ = nullptr;
    Tensor weight_;
    bool bound_ = false;
    bool optional_ = false;
    WeightParameter* tied_to_ = nullptr;
    std::unordered_map<int, Tensor> shards_;
    WeightLoaderFn weight_loader_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
