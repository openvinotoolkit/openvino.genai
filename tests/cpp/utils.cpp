// Copyright (C) 2018-2026 Intel Corporation
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

TEST(TestGetModelProperties, returns_globals_when_no_meta_keys) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {"NUM_STREAMS", std::string("2")},
    };
    auto result = get_model_properties(main_props, "vision_embeddings");
    EXPECT_EQ(result.size(), 2u);
    ASSERT_EQ(result.count("CACHE_DIR"), 1u);
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.at("CACHE_DIR").as<std::string>(), "/tmp");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "2");
}

TEST(TestGetModelProperties, excludes_per_model_meta_key_from_globals) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {PER_MODEL_PROPERTIES, ov::AnyMap{}},
    };
    auto result = get_model_properties(main_props, "vision_embeddings");
    EXPECT_EQ(result.count(PER_MODEL_PROPERTIES), 0u);
    ASSERT_EQ(result.count("CACHE_DIR"), 1u);
    EXPECT_EQ(result.at("CACHE_DIR").as<std::string>(), "/tmp");
}

TEST(TestGetModelProperties, per_model_overrides_global) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("2")},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings");
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "8");
}

TEST(TestGetModelProperties, per_model_missing_role_leaves_globals) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("2")},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"text_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings");
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "2");
}

TEST(TestGetModelProperties, merges_disjoint_keys_across_layers) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings");
    ASSERT_EQ(result.count("CACHE_DIR"), 1u);
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.count(PER_MODEL_PROPERTIES), 0u);
    EXPECT_EQ(result.at("CACHE_DIR").as<std::string>(), "/tmp");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "8");
}
