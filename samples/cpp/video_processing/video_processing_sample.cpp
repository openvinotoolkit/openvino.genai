// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_processing/ffmpeg_vpl_pipeline.hpp"
#include <iostream>
#include <string>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_video> <output_file> [options]\n"
              << "Options:\n"
              << "  --width <width>        Target width for video processing\n"
              << "  --height <height>      Target height for video processing\n"
              << "  --denoise              Enable denoising filter\n"
              << "  --enhance              Enable detail enhancement filter\n"
              << "  --format <format>      Output format: nv12 (default) or rgb\n"
              << "\nExample:\n"
              << "  " << program_name << " input.mp4 output.yuv --width 1280 --height 720 --denoise\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    ov::genai::VideoProcessingConfig config;
    config.input_file = argv[1];
    config.output_file = argv[2];
    
    // Parse optional arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--width" && i + 1 < argc) {
            config.target_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.target_height = std::stoi(argv[++i]);
        } else if (arg == "--denoise") {
            config.denoise = true;
        } else if (arg == "--enhance") {
            config.detail_enhance = true;
        } else if (arg == "--format" && i + 1 < argc) {
            std::string format = argv[++i];
            if (format == "rgb") {
                config.output_format = 1;
            } else if (format == "nv12") {
                config.output_format = 0;
            } else {
                std::cerr << "Unknown format: " << format << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    try {
        std::cout << "Creating FFmpeg + oneVPL pipeline...\n";
        ov::genai::FFmpegVPLPipeline pipeline(config);
        
        std::cout << "Video metadata:\n" << pipeline.get_metadata() << std::endl;
        
        std::cout << "Processing video...\n";
        if (pipeline.process()) {
            std::cout << "Video processing completed successfully!\n";
            std::cout << "Output saved to: " << config.output_file << std::endl;
        } else {
            std::cerr << "Video processing failed!\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
