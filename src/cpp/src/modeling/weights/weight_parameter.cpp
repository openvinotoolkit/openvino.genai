// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/weights/weight_parameter.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <openvino/core/except.hpp>

#include "modeling/ops/ops.hpp"

namespace ov {
namespace genai {
namespace modeling {

WeightParameter::WeightParameter(std::string name, OpContext* ctx) : name_(std::move(name)), ctx_(ctx) {}

const std::string& WeightParameter::name() const {
    return name_;
}

OpContext* WeightParameter::context() const {
    return ctx_;
}

void WeightParameter::set_weight_loader(WeightLoaderFn fn) {
    weight_loader_ = std::move(fn);
}

const WeightParameter::WeightLoaderFn* WeightParameter::weight_loader() const {
    if (!weight_loader_) {
        return nullptr;
    }
    return &weight_loader_;
}

void WeightParameter::bind(const Tensor& weight) {
    if (tied_to_) {
        tied_to_ = nullptr;
    }
    auto* w_ctx = weight.context();
    if (ctx_ && w_ctx && ctx_ != w_ctx) {
        OPENVINO_THROW("WeightParameter context does not match weight context: ", name_);
    }
    weight_ = weight;
    bound_ = true;
    shards_.clear();
}

bool WeightParameter::is_bound() const {
    return bound_ || tied_to_ != nullptr;
}

const Tensor& WeightParameter::value() const {
    if (tied_to_) {
        return tied_to_->value();
    }
    if (!bound_) {
        OPENVINO_THROW("WeightParameter not bound: ", name_);
    }
    return weight_;
}

void WeightParameter::add_shard(int shard_id, const Tensor& shard) {
    if (bound_) {
        OPENVINO_THROW("WeightParameter already bound: ", name_);
    }
    if (tied_to_) {
        OPENVINO_THROW("WeightParameter is tied and cannot accept shards: ", name_);
    }
    shards_[shard_id] = shard;
}

void WeightParameter::finalize() {
    if (bound_ || tied_to_ || shards_.empty()) {
        return;
    }
    if (shards_.size() == 1) {
        bind(shards_.begin()->second);
        return;
    }

    std::vector<int> keys;
    keys.reserve(shards_.size());
    for (const auto& kv : shards_) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    std::vector<Tensor> parts;
    parts.reserve(keys.size());
    for (int k : keys) {
        parts.push_back(shards_.at(k));
    }

    auto merged = ops::concat(parts, 0);
    bind(merged);
}

void WeightParameter::tie_to(WeightParameter& other) {
    if (&other == this) {
        return;
    }
    tied_to_ = &other;
}

void WeightParameter::set_optional(bool optional) {
    optional_ = optional;
}

bool WeightParameter::is_optional() const {
    return optional_;
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
