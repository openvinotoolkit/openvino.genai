// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_video.hpp"

#include <openvino/core/except.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cstring>
#include <set>

namespace fs = std::filesystem;

namespace {
std::vector<size_t> make_indices(size_t total_frames, size_t num_frames) {
    std::vector<size_t> indices;
    indices.reserve(num_frames);

    const float step = static_cast<float>(total_frames) / num_frames;

    for (size_t i = 0; i < num_frames; ++i) {
        indices.push_back(std::min(static_cast<size_t>(i * step), total_frames - 1));
    }

    return indices;
}
}  // namespace

namespace utils {
std::pair<ov::Tensor, ov::genai::VideoMetadata> load_video(const fs::path& video_path, size_t num_frames) {
    cv::VideoCapture cap(video_path.string());
    OPENVINO_ASSERT(cap.isOpened(), "Could not open the video file: ", video_path.string());

    const size_t total_num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    ov::genai::VideoMetadata video_metadata;
    video_metadata.fps = cap.get(cv::CAP_PROP_FPS);
    // Passing video metadata with frame indices defined enables sampling based on provided indices within the pipeline,
    // and any model-specific sampling logic will be skipped (if defined).
    // Leave frames_indices empty to apply model-specific sampling (e.g. for Qwen3-VL).
    video_metadata.frames_indices = make_indices(total_num_frames, num_frames);

    const size_t width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const size_t height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    ov::Tensor video_tensor(ov::element::u8, ov::Shape{total_num_frames, height, width, 3});
    auto video_tensor_data = video_tensor.data<uint8_t>();

    cv::Mat frame;
    size_t frame_idx = 0;
    while (cap.read(frame)) {
        OPENVINO_ASSERT(static_cast<size_t>(frame.cols) == width && static_cast<size_t>(frame.rows) == height &&
                            frame.channels() == 3,
                        "Unexpected frame geometry while decoding ", video_path.string());
        std::memcpy(video_tensor_data, frame.data, frame.total() * 3 * sizeof(uint8_t));
        video_tensor_data += frame.total() * 3;
        frame_idx++;
    }
    OPENVINO_ASSERT(frame_idx == total_num_frames,
                    "Frame count mismatch: expected ", total_num_frames, ", got ", frame_idx);

    return {std::move(video_tensor), std::move(video_metadata)};
}

std::pair<std::vector<ov::Tensor>, std::vector<ov::genai::VideoMetadata>> load_videos(const fs::path& input_path) {
    OPENVINO_ASSERT(!input_path.empty() && fs::exists(input_path), "Path to videos is empty or does not exist.");
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_videos{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> videos;
        std::vector<ov::genai::VideoMetadata> videos_metadata;
        for (const fs::path& dir_entry : sorted_videos) {
            auto [video, video_metadata] = load_video(dir_entry);
            videos.push_back(std::move(video));
            videos_metadata.push_back(std::move(video_metadata));
        }
        return {std::move(videos), std::move(videos_metadata)};
    }
    const auto [video, video_metadata] = load_video(input_path);
    return {{video}, {video_metadata}};
}
}  // namespace utils
