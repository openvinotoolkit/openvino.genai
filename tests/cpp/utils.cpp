// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "utils.hpp"


using namespace ov::genai::utils;
using map_type = std::map<std::string, int64_t>;

TEST(TestIsContainer, test_is_container) {
    EXPECT_EQ(is_container<int>, false);
    EXPECT_EQ(is_container<int64_t>, false);
    EXPECT_EQ(is_container<float>, false);
    EXPECT_EQ(is_container<size_t>, false);
    EXPECT_EQ(is_container<std::string>, true);
    EXPECT_EQ(is_container<std::vector<float>>, true);
    EXPECT_EQ(is_container<map_type>, true);
    EXPECT_EQ(is_container<std::set<int64_t>>, true);
}

TEST(TestAddExtensions, test_add_extensions_to_core) {
    ov::AnyMap properties1 = {extensions({"/home/path1.so", "/home/path2.so"})};
    ov::AnyMap properties2 = {extensions(std::vector<std::filesystem::path>{"/home/path1.so", "/home/path2.so"})};
    ov::AnyMap properties3 = {extensions(std::vector<std::shared_ptr<ov::Extension>>{})};

    EXPECT_THROW(add_extensions_to_core(properties1), ov::Exception);
    EXPECT_THROW(add_extensions_to_core(properties2), ov::Exception);
    EXPECT_NO_THROW(add_extensions_to_core(properties3));
}
