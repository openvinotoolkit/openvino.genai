// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <memory>
#include <variant>

#include <gtest/gtest.h>
#include <openvino/runtime/auto/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include "openvino/genai/omni/pipeline.hpp"
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

// --- get_per_layer_embeddings_device_and_properties tests ---

TEST(TestGetPerLayerEmbeddingsDeviceAndProperties, gpu_defaults_to_auto_with_cpu_priority) {
    ov::AnyMap main_props = {
        {"CACHE_DIR", std::string("/tmp")},
    };
    auto [device, props] = get_per_layer_embeddings_device_and_properties(main_props, "GPU");
    EXPECT_EQ(device, "AUTO");
    ASSERT_EQ(props.count("CACHE_DIR"), 1u);
    EXPECT_EQ(props.at("CACHE_DIR").as<std::string>(), "/tmp");
    ASSERT_EQ(props.count(ov::device::priorities.name()), 1u);
    EXPECT_EQ(props.at(ov::device::priorities.name()).as<std::string>(), "CPU");
    ASSERT_EQ(props.count(ov::intel_auto::enable_startup_fallback.name()), 1u);
    EXPECT_FALSE(props.at(ov::intel_auto::enable_startup_fallback.name()).as<bool>());
}

TEST(TestGetPerLayerEmbeddingsDeviceAndProperties, gpu_indexed_defaults_to_auto_with_cpu_priority) {
    ov::AnyMap main_props = {};
    auto [device, props] = get_per_layer_embeddings_device_and_properties(main_props, "GPU.0");
    EXPECT_EQ(device, "AUTO");
    ASSERT_EQ(props.count(ov::device::priorities.name()), 1u);
    EXPECT_EQ(props.at(ov::device::priorities.name()).as<std::string>(), "CPU");
}

TEST(TestGetPerLayerEmbeddingsDeviceAndProperties, explicit_role_device_overrides_gpu_default) {
    ov::AnyMap main_props = {
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"NUM_STREAMS", std::string("2")}}},
        }},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"text_embeddings_per_layer", ov::AnyMap{
                {"device", std::string("GPU")},
            }},
        }},
    };
    auto [device, props] = get_per_layer_embeddings_device_and_properties(main_props, "GPU");
    EXPECT_EQ(device, "GPU");
    EXPECT_EQ(props.count("device"), 0u);
    EXPECT_EQ(props.count(ov::device::priorities.name()), 0u);
    ASSERT_EQ(props.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(props.at("NUM_STREAMS").as<std::string>(), "2");
}

TEST(TestGetPerLayerEmbeddingsDeviceAndProperties, explicit_auto_priorities_preserved) {
    ov::AnyMap main_props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"text_embeddings_per_layer", ov::AnyMap{
                {ov::device::priorities.name(), std::string("GPU,CPU")},
            }},
        }},
    };
    auto [device, props] = get_per_layer_embeddings_device_and_properties(main_props, "GPU");
    EXPECT_EQ(device, "AUTO");
    ASSERT_EQ(props.count(ov::device::priorities.name()), 1u);
    EXPECT_EQ(props.at(ov::device::priorities.name()).as<std::string>(), "GPU,CPU");
}

TEST(TestGetPerLayerEmbeddingsDeviceAndProperties, cpu_and_meta_devices_preserve_default_device) {
    ov::AnyMap main_props = {};
    auto [cpu_dev, cpu_props] = get_per_layer_embeddings_device_and_properties(main_props, "CPU");
    EXPECT_EQ(cpu_dev, "CPU");
    EXPECT_EQ(cpu_props.count(ov::device::priorities.name()), 0u);

    auto [auto_dev, auto_props] = get_per_layer_embeddings_device_and_properties(main_props, "AUTO");
    EXPECT_EQ(auto_dev, "AUTO");
    EXPECT_EQ(auto_props.count(ov::device::priorities.name()), 0u);
    EXPECT_EQ(auto_props.count(ov::intel_auto::enable_startup_fallback.name()), 0u);

    auto [auto_gpu_dev, auto_gpu_props] = get_per_layer_embeddings_device_and_properties(main_props, "AUTO:GPU");
    EXPECT_EQ(auto_gpu_dev, "AUTO:GPU");
    EXPECT_EQ(auto_gpu_props.count(ov::device::priorities.name()), 0u);

    auto [multi_dev, multi_props] = get_per_layer_embeddings_device_and_properties(main_props, "MULTI:GPU,CPU");
    EXPECT_EQ(multi_dev, "MULTI:GPU,CPU");
    EXPECT_EQ(multi_props.count(ov::device::priorities.name()), 0u);
}

TEST(TestGetPerLayerEmbeddingsDeviceAndProperties, non_ple_role_does_not_strip_device_key) {
    ov::AnyMap main_props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{
                {"device", std::string("CPU")},
            }},
        }},
    };
    auto result = get_model_properties(main_props, "vision_embeddings", "GPU");
    EXPECT_EQ(result.count("device"), 1u);
}

TEST(TestGetPerLayerEmbeddingsDeviceAndProperties, gpu_to_auto_preserves_child_device_properties) {
    ov::AnyMap main_props = {
        {ov::device::properties.name(), ov::AnyMap{
            {"CPU", ov::AnyMap{{"INFERENCE_NUM_THREADS", 4}}},
            {"AUTO", ov::AnyMap{{ov::intel_auto::enable_startup_fallback.name(), true}}},
        }},
    };
    auto [device, props] = get_per_layer_embeddings_device_and_properties(main_props, "GPU");
    EXPECT_EQ(device, "AUTO");
    // DEVICE_PROPERTIES[AUTO] is flattened into top-level props and takes precedence over default false
    ASSERT_EQ(props.count(ov::intel_auto::enable_startup_fallback.name()), 1u);
    EXPECT_TRUE(props.at(ov::intel_auto::enable_startup_fallback.name()).as<bool>());
    // DEVICE_PROPERTIES[CPU] is preserved inside DEVICE_PROPERTIES so AUTO can forward it to CPU
    ASSERT_EQ(props.count(ov::device::properties.name()), 1u);
    const auto& forwarded_dp = props.at(ov::device::properties.name()).as<ov::AnyMap>();
    EXPECT_EQ(forwarded_dp.count("AUTO"), 0u);
    ASSERT_EQ(forwarded_dp.count("CPU"), 1u);
    const auto& cpu_dp = forwarded_dp.at("CPU").as<ov::AnyMap>();
    ASSERT_EQ(cpu_dp.count("INFERENCE_NUM_THREADS"), 1u);
    EXPECT_EQ(cpu_dp.at("INFERENCE_NUM_THREADS").as<int>(), 4);
}

// --- validate_vlm_model_properties tests ---

TEST(TestValidateVlmModelProperties, no_op_when_key_absent) {
    ov::AnyMap props = {{"CACHE_DIR", std::string("/tmp")}};
    EXPECT_NO_THROW(validate_vlm_model_properties(props));
}

TEST(TestValidateVlmModelProperties, accepts_known_roles) {
    ov::AnyMap props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"vision_embeddings", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
            {"language_model", ov::AnyMap{{"CACHE_DIR", std::string("/tmp")}}},
        }},
    };
    EXPECT_NO_THROW(validate_vlm_model_properties(props));
}

TEST(TestValidateVlmModelProperties, throws_on_unknown_role) {
    ov::AnyMap props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"nonexistent_model", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    EXPECT_THROW(validate_vlm_model_properties(props), ov::Exception);
}

TEST(TestValidateVlmModelProperties, throws_with_role_name_in_message) {
    ov::AnyMap props = {
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"bogus_role", ov::AnyMap{{"NUM_STREAMS", std::string("4")}}},
        }},
    };
    try {
        validate_vlm_model_properties(props);
        FAIL() << "Expected exception not thrown";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("bogus_role"), std::string::npos);
    }
}

namespace {

class DummySpeechStreamer : public ov::genai::OmniSpeechStreamerBase {
public:
    ov::genai::StreamingStatus write(const ov::Tensor& chunk) override {
        return ov::genai::StreamingStatus::RUNNING;
    }
    void end() override {}
};

}  // namespace

TEST(TestGetSpeechStreamerFromMap, returns_monostate_when_key_absent) {
    ov::AnyMap props = {{"CACHE_DIR", std::string("/tmp")}};
    EXPECT_TRUE(std::holds_alternative<std::monostate>(get_speech_streamer_from_map(props)));
}

TEST(TestGetSpeechStreamerFromMap, round_trips_a_callback) {
    std::function<ov::genai::StreamingStatus(const ov::Tensor&)> callback =
        [](const ov::Tensor&) { return ov::genai::StreamingStatus::RUNNING; };
    ov::AnyMap props = {ov::genai::speech_streamer(callback)};

    auto streamer = get_speech_streamer_from_map(props);
    ASSERT_TRUE(std::holds_alternative<std::function<ov::genai::StreamingStatus(const ov::Tensor&)>>(streamer));
}

TEST(TestGetSpeechStreamerFromMap, round_trips_a_streamer_object) {
    auto dummy = std::make_shared<DummySpeechStreamer>();
    ov::AnyMap props = {ov::genai::speech_streamer(dummy)};

    auto streamer = get_speech_streamer_from_map(props);
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<ov::genai::OmniSpeechStreamerBase>>(streamer));
    EXPECT_EQ(std::get<std::shared_ptr<ov::genai::OmniSpeechStreamerBase>>(streamer), dummy);
}

TEST(TestGetSpeechStreamerFromMap, round_trips_monostate) {
    ov::AnyMap props = {ov::genai::speech_streamer(std::monostate{})};
    EXPECT_TRUE(std::holds_alternative<std::monostate>(get_speech_streamer_from_map(props)));
}
