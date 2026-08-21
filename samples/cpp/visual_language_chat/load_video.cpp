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
    OPENVINO_ASSERT(total_frames > 0, "Video reports zero frames, nothing to sample.");
    OPENVINO_ASSERT(num_frames > 0, "Number of frames to sample must be positive.");

    // A short video can't yield num_frames distinct indices; sampling fewer beats emitting duplicates.
    const size_t sampled_frames = std::min(num_frames, total_frames);

    std::vector<size_t> indices;
    indices.reserve(sampled_frames);

    const float step = static_cast<float>(total_frames) / sampled_frames;

    for (size_t i = 0; i < sampled_frames; ++i) {
        indices.push_back(std::min(static_cast<size_t>(i * step), total_frames - 1));
    }

    return indices;
}
}  // namespace

namespace utils {
std::pair<ov::Tensor, ov::genai::VideoMetadata> load_video(const fs::path& video_path, size_t num_frames) {
    cv::VideoCapture cap(video_path.string());
    OPENVINO_ASSERT(cap.isOpened(), "Could not open the video file: ", video_path.string());

    // OpenCV reports 0 or -1 when a container/codec doesn't expose a frame count. Validate as a
    // double first: casting -1 to size_t would sail past any positivity check downstream.
    const double reported_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    OPENVINO_ASSERT(reported_frames > 0,
                    "Could not determine the frame count of ",
                    video_path.string(),
                    ". The container or codec may not expose it.");
    const size_t total_num_frames = static_cast<size_t>(reported_frames);

    ov::genai::VideoMetadata video_metadata;
    video_metadata.fps = cap.get(cv::CAP_PROP_FPS);
    // Passing video metadata with frame indices defined enables sampling based on provided indices within the pipeline,
    // and any model-specific sampling logic will be skipped (if defined).
    // Leave frames_indices empty to apply model-specific sampling (e.g. for Qwen3-VL).
    video_metadata.frames_indices = make_indices(total_num_frames, num_frames);

    const double reported_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const double reported_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    OPENVINO_ASSERT(reported_width > 0 && reported_height > 0,
                    "Could not determine the frame size of ",
                    video_path.string());
    const size_t width = static_cast<size_t>(reported_width);
    const size_t height = static_cast<size_t>(reported_height);
    ov::Tensor video_tensor(ov::element::u8, ov::Shape{total_num_frames, height, width, 3});
    auto video_tensor_data = video_tensor.data<uint8_t>();

    cv::Mat frame;
    size_t frame_idx = 0;
    // Bound by the reported count: containers that under-report it would otherwise overrun the tensor.
    while (frame_idx < total_num_frames && cap.read(frame)) {
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
