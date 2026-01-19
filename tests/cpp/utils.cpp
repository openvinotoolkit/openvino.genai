// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "utils.hpp"
#include "openvino/genai/extensions.hpp"


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
#ifdef __linux__
    ov::AnyMap properties1 = {ov::genai::extensions(std::vector<std::filesystem::path>{"libopenvino_tokenizers.so"})};
    EXPECT_NO_THROW(add_extensions_to_core(properties1));
#endif
    // Use intentionally non-existent, platform-agnostic extension paths to trigger error handling.
    ov::AnyMap properties2 = {ov::genai::extensions(
        std::vector<std::filesystem::path>{"non_existent_extension1", "non_existent_extension2"})};
    ov::AnyMap properties3 = {ov::genai::extensions(std::vector<std::shared_ptr<ov::Extension>>{})};

    EXPECT_THROW(add_extensions_to_core(properties2), ov::Exception);
    EXPECT_NO_THROW(add_extensions_to_core(properties3));
}
