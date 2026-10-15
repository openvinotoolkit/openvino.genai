// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

/** @brief Converts C++ values to JavaScript values. */
template <typename SourceType, typename TargetType>
TargetType cpp_to_js(const Napi::Env& env, const SourceType& value);
