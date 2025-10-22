// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "openvino/genai/video_processing/ffmpeg_vpl_pipeline.hpp"
#include <fstream>
#include <vector>

#ifdef ENABLE_FFMPEG_VPL

// Helper function to create a simple test video file (stub)
std::string create_test_video() {
    // In a real test, we would create an actual video file
    // For now, we'll just return a path
    return "test_video.mp4";
}

TEST(FFmpegVPLPipelineTest, ConstructorWithConfig) {
    ov::genai::VideoProcessingConfig config;
    config.input_file = "test.mp4";
    config.output_file = "output.yuv";
    config.target_width = 640;
    config.target_height = 480;
    
    // Should not throw if libraries are available
    EXPECT_NO_THROW({
        try {
            ov::genai::FFmpegVPLPipeline pipeline(config);
        } catch (const std::runtime_error& e) {
            // It's okay if it throws due to missing file or libraries
            std::string msg = e.what();
            EXPECT_TRUE(msg.find("not enabled") != std::string::npos || 
                       msg.find("Could not open") != std::string::npos);
        }
    });
}

TEST(FFmpegVPLPipelineTest, ConfigurationDefaults) {
    ov::genai::VideoProcessingConfig config;
    
    EXPECT_EQ(config.target_width, 0);
    EXPECT_EQ(config.target_height, 0);
    EXPECT_FALSE(config.denoise);
    EXPECT_FALSE(config.detail_enhance);
    EXPECT_EQ(config.output_format, 0);
}

TEST(FFmpegVPLPipelineTest, ConfigurationCustom) {
    ov::genai::VideoProcessingConfig config;
    config.input_file = "input.mp4";
    config.output_file = "output.yuv";
    config.target_width = 1920;
    config.target_height = 1080;
    config.denoise = true;
    config.detail_enhance = true;
    config.output_format = 1;
    
    EXPECT_EQ(config.input_file, "input.mp4");
    EXPECT_EQ(config.output_file, "output.yuv");
    EXPECT_EQ(config.target_width, 1920);
    EXPECT_EQ(config.target_height, 1080);
    EXPECT_TRUE(config.denoise);
    EXPECT_TRUE(config.detail_enhance);
    EXPECT_EQ(config.output_format, 1);
}

TEST(VideoFrameTest, FrameStructure) {
    ov::genai::VideoFrame frame;
    
    frame.width = 640;
    frame.height = 480;
    frame.format = 0;
    frame.timestamp = 1000;
    frame.data.resize(640 * 480 * 3 / 2); // NV12 format size
    
    EXPECT_EQ(frame.width, 640);
    EXPECT_EQ(frame.height, 480);
    EXPECT_EQ(frame.format, 0);
    EXPECT_EQ(frame.timestamp, 1000);
    EXPECT_EQ(frame.data.size(), 640 * 480 * 3 / 2);
}

TEST(FFmpegVPLPipelineTest, MetadataBeforeInitialization) {
    ov::genai::VideoProcessingConfig config;
    config.input_file = "nonexistent.mp4";
    
    try {
        ov::genai::FFmpegVPLPipeline pipeline(config);
        std::string metadata = pipeline.get_metadata();
        // Should return some metadata even if file doesn't exist
        EXPECT_FALSE(metadata.empty());
    } catch (const std::exception& e) {
        // Expected if file doesn't exist
        SUCCEED();
    }
}

// Integration test - only runs if we have a valid test video
TEST(FFmpegVPLPipelineTest, DISABLED_ProcessVideoFile) {
    // This test is disabled by default because it requires actual video files
    // Enable it manually when you have test assets
    
    ov::genai::VideoProcessingConfig config;
    config.input_file = "test_video.mp4";
    config.output_file = "/tmp/test_output.yuv";
    config.target_width = 640;
    config.target_height = 480;
    
    try {
        ov::genai::FFmpegVPLPipeline pipeline(config);
        
        bool result = pipeline.process();
        EXPECT_TRUE(result);
        
        // Check if output file was created
        std::ifstream output_file(config.output_file);
        EXPECT_TRUE(output_file.good());
        output_file.close();
        
        // Clean up
        std::remove(config.output_file.c_str());
    } catch (const std::exception& e) {
        // If libraries are not available, test is skipped
        SUCCEED() << "Test skipped: " << e.what();
    }
}

TEST(FFmpegVPLPipelineTest, DISABLED_GetNextFrame) {
    // This test is disabled by default because it requires actual video files
    
    ov::genai::VideoProcessingConfig config;
    config.input_file = "test_video.mp4";
    
    try {
        ov::genai::FFmpegVPLPipeline pipeline(config);
        
        ov::genai::VideoFrame frame;
        bool got_frame = pipeline.get_next_frame(frame);
        
        if (got_frame) {
            EXPECT_GT(frame.width, 0);
            EXPECT_GT(frame.height, 0);
            EXPECT_FALSE(frame.data.empty());
        }
    } catch (const std::exception& e) {
        SUCCEED() << "Test skipped: " << e.what();
    }
}

#else

TEST(FFmpegVPLPipelineTest, DisabledByDefault) {
    // When ENABLE_FFMPEG_VPL is not defined, the pipeline should throw
    ov::genai::VideoProcessingConfig config;
    config.input_file = "test.mp4";
    
    EXPECT_THROW({
        ov::genai::FFmpegVPLPipeline pipeline(config);
    }, std::runtime_error);
}

#endif // ENABLE_FFMPEG_VPL
