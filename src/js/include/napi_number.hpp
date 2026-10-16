// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

// Shared numeric conversions for the JavaScript bindings.
namespace ov::js {

constexpr int64_t NAPI_NUMBER_MIN_INTEGER = -(1LL << 53) + 1;
constexpr int64_t NAPI_NUMBER_MAX_INTEGER = (1LL << 53) - 1;

inline Napi::Value number_or_bigint(const Napi::Env& env, int64_t value) {
    if (value >= NAPI_NUMBER_MIN_INTEGER && value <= NAPI_NUMBER_MAX_INTEGER) {
        return Napi::Number::New(env, static_cast<double>(value));
    }
    return Napi::BigInt::New(env, value);
}

inline Napi::Value number_or_bigint(const Napi::Env& env, size_t value) {
    if (value <= static_cast<size_t>(NAPI_NUMBER_MAX_INTEGER)) {
        return Napi::Number::New(env, static_cast<double>(value));
    }
    return Napi::BigInt::New(env, static_cast<uint64_t>(value));
}

inline Napi::Value rounded_number(const Napi::Env& env, float value) {
    return Napi::Number::New(env, std::round(static_cast<double>(value) * 1e7) / 1e7);
}

}  // namespace ov::js
