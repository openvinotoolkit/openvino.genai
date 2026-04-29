// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/runtime/properties.hpp>
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

// --- DEVICE_PROPERTIES layer tests ---

TEST(TestGetModelProperties, device_properties_applied_when_device_given) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    ASSERT_EQ(result.count("CACHE_DIR"), 1u);
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "4");
    // DEVICE_PROPERTIES meta key must be stripped
    EXPECT_EQ(result.count(ov::device::properties.name()), 0u);
}

TEST(TestGetModelProperties, device_properties_ignored_when_device_missing_in_map) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("2")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "CPU");
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "2");
    EXPECT_EQ(result.count(ov::device::properties.name()), 0u);
}

TEST(TestGetModelProperties, device_properties_forwarded_when_device_empty) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "");
    // When device is empty, DEVICE_PROPERTIES is forwarded as-is
    EXPECT_EQ(result.count(ov::device::properties.name()), 1u);
    ASSERT_EQ(result.count("CACHE_DIR"), 1u);
}

TEST(TestGetModelProperties, device_properties_forwarded_when_device_defaulted) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    // Default device parameter is ""
    auto result = get_model_properties(main_props, "vision_embeddings");
    EXPECT_EQ(result.count(ov::device::properties.name()), 1u);
}

// --- Three-layer precedence tests ---

TEST(TestGetModelProperties, model_properties_wins_over_device_properties) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("1")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("16")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "16");
}

TEST(TestGetModelProperties, device_properties_wins_over_globals) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("1")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "4");
}

TEST(TestGetModelProperties, three_layers_disjoint_keys_merge) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"PERF_HINT", std::string("THROUGHPUT")}}},
        }},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.at("CACHE_DIR").as<std::string>(), "/tmp");
    EXPECT_EQ(result.at("PERF_HINT").as<std::string>(), "THROUGHPUT");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "8");
    EXPECT_EQ(result.count(ov::device::properties.name()), 0u);
    EXPECT_EQ(result.count(PER_MODEL_PROPERTIES), 0u);
}

TEST(TestGetModelProperties, model_properties_no_match_falls_to_device_properties) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("1")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"text_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("16")}}},
        }},
    };
    // vision_embeddings not in MODEL_PROPERTIES, falls back to DEVICE_PROPERTIES[GPU]
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    ASSERT_EQ(result.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "4");
}

TEST(TestGetModelProperties, input_map_not_modified) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    const size_t original_size = main_props.size();
    get_model_properties(main_props, "vision_embeddings", "GPU");
    // Input map must not be modified
    EXPECT_EQ(main_props.size(), original_size);
    EXPECT_EQ(main_props.count(PER_MODEL_PROPERTIES), 1u);
    EXPECT_EQ(main_props.count(ov::device::properties.name()), 1u);
}

// --- validate_vlm_model_properties tests ---

TEST(TestValidateVlmModelProperties, no_op_when_key_absent) {
    ov::AnyMap props = {{"CACHE_DIR", std::string("/tmp")}};
    EXPECT_NO_THROW(validate_vlm_model_properties(props, get_known_vlm_model_roles()));
}

TEST(TestValidateVlmModelProperties, accepts_known_roles) {
    ov::AnyMap props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
            {"language_model", ov::AnyMap{{"CACHE_DIR", std::string("/tmp")}}},
        }},
    };
    EXPECT_NO_THROW(validate_vlm_model_properties(props, get_known_vlm_model_roles()));
}

TEST(TestValidateVlmModelProperties, throws_on_unknown_role) {
    ov::AnyMap props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"nonexistent_model", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    EXPECT_THROW(validate_vlm_model_properties(props, get_known_vlm_model_roles()), ov::Exception);
}

TEST(TestValidateVlmModelProperties, throws_with_role_name_in_message) {
    ov::AnyMap props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"bogus_role", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    try {
        validate_vlm_model_properties(props, get_known_vlm_model_roles());
        FAIL() << "Expected exception not thrown";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("bogus_role"), std::string::npos);
    }
}

TEST(TestValidateVlmModelProperties, accepts_custom_known_roles) {
    const std::vector<std::string> custom_roles = {"my_encoder", "my_decoder"};
    ov::AnyMap props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"my_encoder", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    EXPECT_NO_THROW(validate_vlm_model_properties(props, custom_roles));
}

TEST(TestValidateVlmModelProperties, throws_on_unknown_with_custom_roles) {
    const std::vector<std::string> custom_roles = {"my_encoder", "my_decoder"};
    ov::AnyMap props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    EXPECT_THROW(validate_vlm_model_properties(props, custom_roles), ov::Exception);
}

// --- get_known_vlm_model_roles tests ---

TEST(TestGetKnownVlmModelRoles, returns_nonempty_list) {
    const auto& roles = get_known_vlm_model_roles();
    EXPECT_FALSE(roles.empty());
}

TEST(TestGetKnownVlmModelRoles, contains_expected_roles) {
    const auto& roles = get_known_vlm_model_roles();
    auto has = [&](const std::string& r) {
        return std::find(roles.begin(), roles.end(), r) != roles.end();
    };
    EXPECT_TRUE(has("vision_embeddings"));
    EXPECT_TRUE(has("text_embeddings"));
    EXPECT_TRUE(has("language_model"));
    EXPECT_TRUE(has("resampler"));
}

TEST(TestGetKnownVlmModelRoles, returns_same_instance) {
    // The function returns a static reference, so address must be stable
    const auto& first = get_known_vlm_model_roles();
    const auto& second = get_known_vlm_model_roles();
    EXPECT_EQ(&first, &second);
}
