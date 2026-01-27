// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/genai/visual_language/pipeline.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<size_t> make_indices(size_t total_frames, size_t num_frames) {
    std::vector<size_t> indices;
    indices.reserve(num_frames);

    auto step = float(total_frames) / num_frames;

    for (size_t i = 0; i < num_frames; ++i) {
        size_t idx = std::min(size_t(i * step), total_frames - 1);
        indices.push_back(idx);
    }

    return indices;
}

ov::Tensor load_video(const std::filesystem::path& video_path, size_t num_frames = 8) {
    cv::VideoCapture cap(video_path.string());

    if (!cap.isOpened()) {
        OPENVINO_THROW("Could not open the video file.");
    }
    size_t total_num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    auto indices = make_indices(total_num_frames, num_frames);

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    size_t width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    size_t height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    ov::Tensor video_tensor(ov::element::u8, ov::Shape{num_frames, height, width, 3});
    auto video_tensor_data = video_tensor.data<uint8_t>();

    size_t frame_idx = 0;
    while (cap.read(frame)) {
        OPENVINO_ASSERT(frame.cols == width && frame.rows == height && frame.channels() == 3);
        if (std::find(indices.begin(), indices.end(), frame_idx) != indices.end()) {
            memcpy(video_tensor_data, frame.data, frame.total() * 3 * sizeof(uint8_t));
            video_tensor_data += frame.total() * 3;
        }
        frame_idx++;
    }
    OPENVINO_ASSERT(frame_idx == total_num_frames, "Frame count mismatch: expected " + std::to_string(total_num_frames) + ", got " + std::to_string(frame_idx));
    
    return video_tensor;
}

std::vector<ov::Tensor> load_videos(const std::filesystem::path& input_path) {
    if (input_path.empty() || !fs::exists(input_path)) {
        OPENVINO_THROW("Path to videos is empty or does not exist.");
    }
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_videos{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> videos;
        for (const fs::path& dir_entry : sorted_videos) {
            videos.push_back(load_video(dir_entry));
        }
        return videos;
    }
    return {load_video(input_path)};
}

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        OPENVINO_THROW(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <VIDEO_FILE OR DIR_WITH_VIDEOS> <DEVICE>");
    }
    std::vector<ov::Tensor> videos = load_videos(argv[2]);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    std::string device = (argc == 4) ? argv[3] : "CPU";
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(argv[1], device, enable_compile_cache);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt;

    pipe.start_chat();
    std::cout << "question:\n";

    std::getline(std::cin, prompt);
    pipe.generate(prompt,
                  ov::genai::videos(videos),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword));
    std::cout << "\n----------\n"
        "question:\n";
    while (std::getline(std::cin, prompt)) {
        // New images and videos can be passed at each turn
        pipe.generate(prompt,
                      ov::genai::generation_config(generation_config),
                      ov::genai::streamer(print_subword));
        std::cout << "\n----------\n"
            "question:\n";
    }
    pipe.finish_chat();
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
