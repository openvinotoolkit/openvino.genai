// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/weights/synthetic_weight_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>

#include <openvino/core/except.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

namespace {

uint32_t fnv1a_32(const std::string& text) {
    uint32_t hash = 2166136261u;
    for (unsigned char c : text) {
        hash ^= static_cast<uint32_t>(c);
        hash *= 16777619u;
    }
    return hash;
}

size_t num_elements(const ov::Shape& shape) {
    size_t n = 1;
    for (const auto dim : shape) {
        n *= dim;
    }
    return n;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

SyntheticWeightSource::SyntheticWeightSource(std::vector<SyntheticWeightSpec> specs,
                                             uint32_t seed,
                                             float low,
                                             float high)
    : seed_(seed),
      low_(low),
      high_(high) {
    specs_.reserve(specs.size());
    keys_.reserve(specs.size());
    for (auto& spec : specs) {
        if (spec.name.empty()) {
            OPENVINO_THROW("SyntheticWeightSpec.name must not be empty");
        }
        keys_.push_back(spec.name);
        specs_.emplace(spec.name, std::move(spec));
    }
}

std::vector<std::string> SyntheticWeightSource::keys() const {
    return keys_;
}

bool SyntheticWeightSource::has(const std::string& name) const {
    return specs_.find(name) != specs_.end();
}

const ov::Tensor& SyntheticWeightSource::get_tensor(const std::string& name) const {
    auto cached = cache_.find(name);
    if (cached != cache_.end()) {
        return cached->second;
    }

    const auto it = specs_.find(name);
    if (it == specs_.end()) {
        OPENVINO_THROW("SyntheticWeightSource unknown tensor: ", name);
    }

    const uint32_t local_seed = seed_ ^ fnv1a_32(name);
    auto inserted = cache_.emplace(name, make_tensor(it->second, local_seed));
    return inserted.first->second;
}

void SyntheticWeightSource::release_tensor(const std::string& name) {
    cache_.erase(name);
}

void SyntheticWeightSource::release_all_cached_tensors() {
    cache_.clear();
}

ov::Tensor SyntheticWeightSource::make_tensor(const SyntheticWeightSpec& spec, uint32_t local_seed) const {
    ov::Tensor tensor(spec.dtype, spec.shape);
    const size_t total = num_elements(spec.shape);
    std::mt19937 gen(local_seed);
    std::uniform_real_distribution<float> dist(low_, high_);

    if (spec.dtype == ov::element::f32) {
        auto* data = tensor.data<float>();
        for (size_t i = 0; i < total; ++i) {
            data[i] = dist(gen);
        }
        return tensor;
    }
    if (spec.dtype == ov::element::f16) {
        auto* data = tensor.data<ov::float16>();
        for (size_t i = 0; i < total; ++i) {
            data[i] = ov::float16(dist(gen));
        }
        return tensor;
    }
    if (spec.dtype == ov::element::bf16) {
        auto* data = tensor.data<ov::bfloat16>();
        for (size_t i = 0; i < total; ++i) {
            data[i] = ov::bfloat16(dist(gen));
        }
        return tensor;
    }
    if (spec.dtype == ov::element::i64) {
        auto* data = tensor.data<int64_t>();
        for (size_t i = 0; i < total; ++i) {
            data[i] = static_cast<int64_t>(dist(gen) * 1000.0f);
        }
        return tensor;
    }
    if (spec.dtype == ov::element::i32) {
        auto* data = tensor.data<int32_t>();
        for (size_t i = 0; i < total; ++i) {
            data[i] = static_cast<int32_t>(dist(gen) * 1000.0f);
        }
        return tensor;
    }
    if (spec.dtype == ov::element::u8) {
        auto* data = tensor.data<uint8_t>();
        const float denom = high_ - low_;
        for (size_t i = 0; i < total; ++i) {
            const float v = std::abs(denom) < 1e-12f ? 0.5f : (dist(gen) - low_) / denom;
            data[i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f);
        }
        return tensor;
    }

    OPENVINO_THROW("SyntheticWeightSource unsupported dtype for ", spec.name, ": ", spec.dtype);
}

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
