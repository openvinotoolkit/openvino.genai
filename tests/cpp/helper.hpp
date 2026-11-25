// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <exception>
#include <memory>
#include <typeinfo>
#include <vector>

#include "openvino/runtime/core.hpp"

std::shared_ptr<ov::Model> get_dummy_model(ov::Core core, size_t num_layers);

std::shared_ptr<ov::Model> get_dummy_hybrid_model(ov::Core core, size_t kv_num_layers, size_t la_num_layers);

// Same exception type and message: failures crossing threads or handles may be copies of the original.
inline void expect_same_exception(const std::exception_ptr& actual, const std::exception_ptr& expected) {
    ASSERT_NE(actual, nullptr);
    ASSERT_NE(expected, nullptr);
    try {
        std::rethrow_exception(expected);
    } catch (const std::exception& expected_error) {
        try {
            std::rethrow_exception(actual);
        } catch (const std::exception& actual_error) {
            EXPECT_EQ(typeid(actual_error), typeid(expected_error));
            EXPECT_STREQ(actual_error.what(), expected_error.what());
        } catch (...) {
            FAIL() << "Actual failure is not a std::exception";
        }
    } catch (...) {
        FAIL() << "Expected failure is not a std::exception";
    }
}
