// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <vector>

#include "openvino/core/any.hpp"

/** @brief Converts C++ values to JavaScript values. */
template <typename SourceType, typename TargetType>
TargetType cpp_to_js(const Napi::Env& env, const SourceType& value);

/**
 * @brief Convert a single ov::Any to a JS value.
 *
 * An empty ov::Any maps to null. Integers outside the JS safe-integer range are
 * returned as BigInt. Throws for any unsupported stored type.
 */
template <>
Napi::Value cpp_to_js<ov::Any, Napi::Value>(const Napi::Env& env, const ov::Any& any);

template <>
Napi::Value cpp_to_js<ov::AnyMap, Napi::Value>(const Napi::Env& env, const ov::AnyMap& any_map);

template <>
Napi::Value cpp_to_js<std::vector<ov::AnyMap>, Napi::Value>(const Napi::Env& env,
                                                            const std::vector<ov::AnyMap>& value);
