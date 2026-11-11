// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <memory>
#include <variant>

#include <gtest/gtest.h>
#include <openvino/runtime/properties.hpp>
#include "openvino/genai/common_types.hpp"
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

TEST(TestValidateVlmModelProperties, accepts_known_roles_in_device_mapping) {
    ov::AnyMap props = {
        ov::genai::device_mapping(std::map<std::string, std::string>{
            {"text_embeddings_per_layer", "CPU"},
            {"text_embeddings", "CPU"},
            {"audio_embeddings", "GPU"},
            {"vision_embeddings_tiles", "GPU"},
        }),
    };
    EXPECT_NO_THROW(validate_vlm_model_properties(props));
}

TEST(TestValidateVlmModelProperties, throws_on_unknown_role_in_device_mapping) {
    ov::AnyMap props = {
        {DEVICE_MAPPING_PROPERTY, ov::AnyMap{
            {"invalid_submodel_role", std::string("CPU")},
        }},
    };
    EXPECT_THROW(validate_vlm_model_properties(props), ov::Exception);
}

// --- get_model_device & get_model_device_and_properties tests ---

TEST(TestGetModelDevice, ple_defaults_to_cpu_when_pipeline_on_gpu) {
    ov::AnyMap props;
    EXPECT_EQ(get_model_device(props, "text_embeddings_per_layer", "GPU"), "CPU");
    EXPECT_EQ(get_model_device(props, "text_embeddings_per_layer", "GPU.0"), "CPU");
    EXPECT_EQ(get_model_device(props, "text_embeddings_per_layer", "GPU.1"), "CPU");
    EXPECT_EQ(get_model_device(props, "text_embeddings_per_layer", "CPU"), "CPU");
}

TEST(TestGetModelDevice, other_models_default_to_pipeline_device_on_gpu) {
    ov::AnyMap props;
    EXPECT_EQ(get_model_device(props, "text_embeddings", "GPU"), "GPU");
    EXPECT_EQ(get_model_device(props, "vision_embeddings", "GPU"), "GPU");
    EXPECT_EQ(get_model_device(props, "language_model", "GPU"), "GPU");
}

TEST(TestGetModelDevice, device_mapping_overrides_default_with_any_map) {
    ov::AnyMap props = {
        {DEVICE_MAPPING_PROPERTY, ov::AnyMap{
            {"text_embeddings_per_layer", std::string("GPU")},
            {"text_embeddings", std::string("CPU")},
        }},
    };
    EXPECT_EQ(get_model_device(props, "text_embeddings_per_layer", "GPU"), "GPU");
    EXPECT_EQ(get_model_device(props, "text_embeddings", "GPU"), "CPU");
    EXPECT_EQ(get_model_device(props, "vision_embeddings", "GPU"), "GPU");
}

TEST(TestGetModelDevice, device_mapping_overrides_default_with_std_map_property) {
    ov::AnyMap props = {
        ov::genai::device_mapping(std::map<std::string, std::string>{
            {"text_embeddings_per_layer", "GPU.0"},
            {"text_embeddings", "CPU"},
        }),
    };
    EXPECT_EQ(get_model_device(props, "text_embeddings_per_layer", "GPU.0"), "GPU.0");
    EXPECT_EQ(get_model_device(props, "text_embeddings", "GPU.0"), "CPU");
}

TEST(TestGetModelDevice, model_properties_device_overrides_device_mapping_and_default) {
    ov::AnyMap props = {
        ov::genai::device_mapping(std::map<std::string, std::string>{
            {"text_embeddings_per_layer", "CPU"},
            {"text_embeddings", "GPU"},
        }),
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"text_embeddings_per_layer", ov::AnyMap{{"device", std::string("GPU")}}},
            {"text_embeddings", ov::AnyMap{{"DEVICE", std::string("CPU")}}},
        }},
    };
    EXPECT_EQ(get_model_device(props, "text_embeddings_per_layer", "GPU"), "GPU");
    EXPECT_EQ(get_model_device(props, "text_embeddings", "GPU"), "CPU");
}

TEST(TestGetModelDeviceAndProperties, strips_device_mapping_and_role_device_and_selects_resolved_device_properties) {
    ov::AnyMap props = {
        {"CACHE_DIR", std::string("/tmp")},
        ov::genai::device_mapping(std::map<std::string, std::string>{
            {"text_embeddings", "CPU"},
        }),
        {ov::device::properties.name(), ov::AnyMap{
            {"GPU", ov::AnyMap{{"PERF_HINT", std::string("LATENCY")}}},
            {"CPU", ov::AnyMap{{"NUM_STREAMS", std::string("1")}}},
        }},
        {PER_MODEL_PROPERTIES, ov::AnyMap{
            {"text_embeddings_per_layer", ov::AnyMap{
                {"ENABLE_MMAP", true},
            }},
            {"vision_embeddings", ov::AnyMap{
                {"device", std::string("CPU")},
                {"INFERENCE_PRECISION_HINT", std::string("f32")},
            }},
        }},
    };

    // 1. text_embeddings_per_layer defaults to CPU when default_device is GPU, so it gets CPU DEVICE_PROPERTIES
    const auto [ple_device, ple_props] =
        get_model_device_and_properties(props, "text_embeddings_per_layer", "GPU");
    EXPECT_EQ(ple_device, "CPU");
    EXPECT_EQ(ple_props.count(DEVICE_MAPPING_PROPERTY), 0u);
    EXPECT_EQ(ple_props.count(PER_MODEL_PROPERTIES), 0u);
    EXPECT_EQ(ple_props.count(ov::device::properties.name()), 0u);
    EXPECT_EQ(ple_props.count("PERF_HINT"), 0u);
    ASSERT_EQ(ple_props.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(ple_props.at("NUM_STREAMS").as<std::string>(), "1");
    ASSERT_EQ(ple_props.count("ENABLE_MMAP"), 1u);
    EXPECT_EQ(ple_props.at("ENABLE_MMAP").as<bool>(), true);

    // 2. text_embeddings mapped via device_mapping to CPU gets CPU DEVICE_PROPERTIES
    const auto [emb_device, emb_props] =
        get_model_device_and_properties(props, "text_embeddings", "GPU");
    EXPECT_EQ(emb_device, "CPU");
    EXPECT_EQ(emb_props.count(DEVICE_MAPPING_PROPERTY), 0u);
    ASSERT_EQ(emb_props.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(emb_props.at("NUM_STREAMS").as<std::string>(), "1");

    // 3. vision_embeddings mapped via MODEL_PROPERTIES["vision_embeddings"]["device"] = "CPU"
    //    strips "device" key from final plugin properties
    const auto [vis_device, vis_props] =
        get_model_device_and_properties(props, "vision_embeddings", "GPU");
    EXPECT_EQ(vis_device, "CPU");
    EXPECT_EQ(vis_props.count("device"), 0u);
    EXPECT_EQ(vis_props.count("DEVICE"), 0u);
    ASSERT_EQ(vis_props.count("INFERENCE_PRECISION_HINT"), 1u);
    EXPECT_EQ(vis_props.at("INFERENCE_PRECISION_HINT").as<std::string>(), "f32");
    ASSERT_EQ(vis_props.count("NUM_STREAMS"), 1u);
    EXPECT_EQ(vis_props.at("NUM_STREAMS").as<std::string>(), "1");

    // 4. language_model stays on GPU and gets GPU DEVICE_PROPERTIES
    const auto [lm_device, lm_props] =
        get_model_device_and_properties(props, "language_model", "GPU");
    EXPECT_EQ(lm_device, "GPU");
    ASSERT_EQ(lm_props.count("PERF_HINT"), 1u);
    EXPECT_EQ(lm_props.at("PERF_HINT").as<std::string>(), "LATENCY");
    EXPECT_EQ(lm_props.count("NUM_STREAMS"), 0u);
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
