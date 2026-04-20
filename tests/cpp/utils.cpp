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

TEST(TestGetModelProperties, returns_globals_when_no_meta_keys) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {"NUM_STREAMS", std::string("2")},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result.at("CACHE_DIR").as<std::string>(), "/tmp");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "2");
}

TEST(TestGetModelProperties, excludes_meta_keys_from_globals) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {PER_MODEL_PROPERTIES, ov::AnyMap{}},
        {DEVICE_PROPERTIES, ov::AnyMap{}},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.count(PER_MODEL_PROPERTIES), 0u);
    EXPECT_EQ(result.count(DEVICE_PROPERTIES), 0u);
    EXPECT_EQ(result.at("CACHE_DIR").as<std::string>(), "/tmp");
}

TEST(TestGetModelProperties, device_overrides_global) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("2")},
        {DEVICE_PROPERTIES, ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "8");
}

TEST(TestGetModelProperties, device_layer_skipped_when_device_not_listed) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("2")},
        {DEVICE_PROPERTIES, ov::AnyMap{
            {"NPU", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "2");
}

TEST(TestGetModelProperties, per_model_overrides_device_and_global) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("2")},
        {DEVICE_PROPERTIES, ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "8");
}

TEST(TestGetModelProperties, per_model_missing_role_leaves_lower_layers) {
    ov::AnyMap main_props = {
        {"NUM_STREAMS", std::string("2")},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"text_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "2");
}

TEST(TestGetModelProperties, does_not_mutate_input_map) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
        {DEVICE_PROPERTIES, ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    auto snapshot = main_props;
    (void)get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(main_props.size(), snapshot.size());
    EXPECT_EQ(main_props.count(PER_MODEL_PROPERTIES), 1u);
    EXPECT_EQ(main_props.count(DEVICE_PROPERTIES), 1u);
    EXPECT_EQ(main_props.at("CACHE_DIR").as<std::string>(), "/tmp");
}

TEST(TestGetModelProperties, device_lookup_is_case_insensitive) {
    ov::AnyMap main_props = {
        {DEVICE_PROPERTIES, ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    EXPECT_EQ(get_model_properties(main_props, "vision_embeddings", "gpu")
                  .at("NUM_STREAMS").as<std::string>(), "8");
    EXPECT_EQ(get_model_properties(main_props, "vision_embeddings", "Gpu")
                  .at("NUM_STREAMS").as<std::string>(), "8");
}

TEST(TestGetModelProperties, role_lookup_is_case_insensitive) {
    ov::AnyMap main_props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    EXPECT_EQ(get_model_properties(main_props, "VISION_EMBEDDINGS", "GPU")
                  .at("NUM_STREAMS").as<std::string>(), "8");
    EXPECT_EQ(get_model_properties(main_props, "Vision_Embeddings", "GPU")
                  .at("NUM_STREAMS").as<std::string>(), "8");
}

TEST(TestGetModelProperties, merges_disjoint_keys_across_layers) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
        {DEVICE_PROPERTIES, ov::AnyMap{
            {"GPU", ov::AnyMap{{"GPU_QUEUE_PRIORITY", std::string("HIGH")}}},
        }},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("8")}}},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.at("CACHE_DIR").as<std::string>(), "/tmp");
    EXPECT_EQ(result.at("GPU_QUEUE_PRIORITY").as<std::string>(), "HIGH");
    EXPECT_EQ(result.at("NUM_STREAMS").as<std::string>(), "8");
}
