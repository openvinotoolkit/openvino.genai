// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include <filesystem>

using namespace ov::genai::module;

// ============================================================================
// Test Data Structure
// ============================================================================

struct SaveVideoTestData {
    std::string name;                    // Test case name
    ov::Shape shape;                     // Tensor shape [B, F, H, W, C]
    std::string filename_prefix;         // Output filename prefix
    uint32_t fps;                        // Frames per second
    int quality;                         // JPEG quality
    size_t expected_file_count;          // Expected number of output files

    // Helper to create test tensor with gradient pattern
    ov::Tensor create_tensor() const {
        ov::Tensor tensor(ov::element::u8, shape);
        fill_u8_tensor(tensor);
        return tensor;
    }

private:
    void fill_u8_tensor(ov::Tensor& tensor) const {
        uint8_t* data = tensor.data<uint8_t>();
        size_t total_size = tensor.get_size();

        // Fill with gradient pattern for each frame
        for (size_t i = 0; i < total_size; i++) {
            data[i] = static_cast<uint8_t>((i * 7) % 256);
        }
    }
};

// ============================================================================
// Test Data Definitions
// ============================================================================

namespace TEST_DATA {

SaveVideoTestData basic_video() {
    return {
        "Basic_Video",
        ov::Shape{1, 10, 64, 64, 3},  // 1 video, 10 frames, 64x64, RGB
        "test_video",
        25,
        85,
        1
    };
}

SaveVideoTestData single_frame_video() {
    return {
        "Single_Frame",
        ov::Shape{1, 1, 32, 32, 3},  // 1 video, 1 frame
        "test_single",
        30,
        90,
        1
    };
}

SaveVideoTestData grayscale_video() {
    return {
        "Grayscale_Video",
        ov::Shape{1, 5, 48, 48, 1},  // 1 video, 5 frames, grayscale
        "test_gray",
        24,
        80,
        1
    };
}

SaveVideoTestData rgba_video() {
    return {
        "RGBA_Video",
        ov::Shape{1, 8, 40, 40, 4},  // 1 video, 8 frames, RGBA
        "test_rgba",
        25,
        85,
        1
    };
}

SaveVideoTestData batch_video() {
    return {
        "Batch_Video",
        ov::Shape{3, 5, 32, 32, 3},  // 3 videos, 5 frames each
        "test_batch",
        25,
        85,
        3
    };
}

SaveVideoTestData large_video() {
    return {
        "Large_Video",
        ov::Shape{1, 20, 128, 128, 3},  // 1 video, 20 frames, 128x128
        "test_large",
        30,
        75,
        1
    };
}

}  // namespace TEST_DATA

// ============================================================================
// Test Parameters
// ============================================================================

// Test parameters:
// <0> SaveVideoTestData - test data structure containing tensor info and expected results
// <1> std::string - device name (e.g., "CPU", "GPU")
using test_params = std::tuple<SaveVideoTestData, std::string>;

// ============================================================================
// Test Fixture
// ============================================================================

class SaveVideoModuleTest : public ov::genai::module::ModuleTestBase,
                            public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    SaveVideoTestData m_test_data;
    std::string m_output_folder = "./test_video_output";

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& test_data = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        return test_data.name + "_" + device;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_test_data, m_device) = GetParam();

        // Clean up output folder before test
        if (std::filesystem::exists(m_output_folder)) {
            std::filesystem::remove_all(m_output_folder);
        }
    }

    void TearDown() override {
        // Clean up output folder after test
        if (std::filesystem::exists(m_output_folder)) {
            std::filesystem::remove_all(m_output_folder);
        }
    }

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "video_gen";

        YAML::Node pipeline_modules = config["pipeline_modules"];

        // ParameterModule
        YAML::Node params_module;
        params_module["type"] = "ParameterModule";
        params_module["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        params_module["outputs"].push_back(output_node("video_data", "OVTensor"));
        pipeline_modules["pipeline_params"] = params_module;

        // SaveVideoModule
        YAML::Node save_module;
        save_module["type"] = "SaveVideoModule";
        save_module["device"] = m_device;

        save_module["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        save_module["inputs"].push_back(input_node("raw_data", "OVTensor", "pipeline_params.video_data"));

        save_module["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        save_module["outputs"].push_back(output_node("saved_video", "String"));
        save_module["outputs"].push_back(output_node("saved_videos", "VecString"));

        YAML::Node save_params;
        save_params["filename_prefix"] = m_test_data.filename_prefix;
        save_params["output_folder"] = m_output_folder;
        save_params["fps"] = std::to_string(m_test_data.fps);
        save_params["quality"] = std::to_string(m_test_data.quality);
        save_params["convert_bgr2rgb"] = "false";
        save_module["params"] = save_params;

        pipeline_modules["save_video"] = save_module;

        // ResultModule
        YAML::Node result_module;
        result_module["type"] = "ResultModule";
        result_module["device"] = "CPU";
        result_module["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        result_module["inputs"].push_back(input_node("saved_video", "String", "save_video.saved_video"));
        result_module["inputs"].push_back(input_node("saved_videos", "VecString", "save_video.saved_videos"));
        pipeline_modules["pipeline_results"] = result_module;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["video_data"] = m_test_data.create_tensor();
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        // Verify saved_video output
        auto video_path = pipe.get_output("saved_video").as<std::string>();
        EXPECT_FALSE(video_path.empty()) << "saved_video output should not be empty";
        EXPECT_TRUE(std::filesystem::exists(video_path))
            << "Output video file should exist: " << video_path;
        EXPECT_TRUE(video_path.find(m_test_data.filename_prefix) != std::string::npos)
            << "Video path should contain prefix '" << m_test_data.filename_prefix << "'";
        EXPECT_TRUE(video_path.find(".avi") != std::string::npos)
            << "Video path should have .avi extension";

        // Verify saved_videos output (vector of paths)
        auto videos = pipe.get_output("saved_videos").as<std::vector<std::string>>();
        EXPECT_EQ(videos.size(), m_test_data.expected_file_count)
            << "saved_videos should contain " << m_test_data.expected_file_count << " file path(s)";

        // Verify all files exist and have reasonable size
        for (const auto& filepath : videos) {
            EXPECT_TRUE(std::filesystem::exists(filepath))
                << "Output video file should exist: " << filepath;
            EXPECT_TRUE(filepath.find(m_test_data.filename_prefix) != std::string::npos)
                << "Video path should contain prefix '" << m_test_data.filename_prefix << "'";

            // Check file size is greater than AVI header minimum (~500 bytes)
            auto file_size = std::filesystem::file_size(filepath);
            EXPECT_GT(file_size, 500u) << "Video file should have content: " << filepath;
        }

        // For single video, verify saved_videos[0] matches saved_video
        if (m_test_data.expected_file_count == 1) {
            EXPECT_EQ(videos[0], video_path)
                << "saved_videos[0] should match saved_video output";
        }
    }
};

// ============================================================================
// Test Case
// ============================================================================

TEST_P(SaveVideoModuleTest, ModuleTest) {
    run();
}

// ============================================================================
// Test Data Configuration
// ============================================================================

namespace save_video_test {

auto test_data = std::vector<SaveVideoTestData> {
    TEST_DATA::basic_video(),
    TEST_DATA::single_frame_video(),
    TEST_DATA::grayscale_video(),
    TEST_DATA::rgba_video(),
    TEST_DATA::batch_video(),
    TEST_DATA::large_video()
};

}  // namespace save_video_test

// ============================================================================
// Test Suite Instantiation
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    ModuleTestSuite,
    SaveVideoModuleTest,
    ::testing::Combine(
        ::testing::ValuesIn(save_video_test::test_data),
        ::testing::Values("CPU")  // SaveVideoModule only supports CPU
    ),
    SaveVideoModuleTest::get_test_case_name
);
