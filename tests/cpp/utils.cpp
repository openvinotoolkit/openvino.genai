// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "utils.hpp"

#include "openvino/runtime/properties.hpp"


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

TEST(TestInheritCacheProperties, inherits_cache_dir_when_absent_in_sub) {
    ov::AnyMap main_props = {{ov::cache_dir.name(), std::string("/tmp/main")}};
    ov::AnyMap sub_props = {};

    auto result = inherit_cache_properties(sub_props, main_props);

    ASSERT_EQ(result.count(ov::cache_dir.name()), 1u);
    EXPECT_EQ(result.at(ov::cache_dir.name()).as<std::string>(), "/tmp/main");
}

TEST(TestInheritCacheProperties, does_not_override_explicit_sub_value) {
    ov::AnyMap main_props = {{ov::cache_dir.name(), std::string("/tmp/main")}};
    ov::AnyMap sub_props  = {{ov::cache_dir.name(), std::string("/tmp/sub")}};

    auto result = inherit_cache_properties(sub_props, main_props);

    EXPECT_EQ(result.at(ov::cache_dir.name()).as<std::string>(), "/tmp/sub");
}

TEST(TestInheritCacheProperties, inherits_cache_mode_independently_of_other_sub_keys) {
    ov::AnyMap main_props = {{ov::cache_mode.name(), ov::CacheMode::OPTIMIZE_SPEED}};
    // Sub map is non-empty but does not set cache_mode.
    ov::AnyMap sub_props  = {{"NUM_STREAMS", std::string("2")}};

    auto result = inherit_cache_properties(sub_props, main_props);

    ASSERT_EQ(result.count(ov::cache_mode.name()), 1u);
    EXPECT_EQ(result.at(ov::cache_mode.name()).as<ov::CacheMode>(), ov::CacheMode::OPTIMIZE_SPEED);
    // Original sub key preserved.
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "2");
}

TEST(TestInheritCacheProperties, does_not_inherit_non_cache_keys) {
    ov::AnyMap main_props = {
        {ov::cache_dir.name(), std::string("/tmp/main")},
        {"NUM_STREAMS", std::string("4")},
    };
    ov::AnyMap sub_props = {};

    auto result = inherit_cache_properties(sub_props, main_props);

    EXPECT_EQ(result.count(ov::cache_dir.name()), 1u);
    EXPECT_EQ(result.count("NUM_STREAMS"), 0u);
}

TEST(TestInheritCacheProperties, empty_main_leaves_sub_unchanged) {
    ov::AnyMap main_props = {};
    ov::AnyMap sub_props  = {{"NUM_STREAMS", std::string("2")}};

    auto result = inherit_cache_properties(sub_props, main_props);

    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result.count(ov::cache_dir.name()), 0u);
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "2");
}
