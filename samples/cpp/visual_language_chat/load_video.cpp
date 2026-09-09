// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_video.hpp"

#include <openvino/core/except.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cstring>
#include <set>

namespace fs = std::filesystem;

namespace {

std::vector<size_t> make_indices(size_t total_frames, size_t num_frames) {
    OPENVINO_ASSERT(total_frames > 0, "Video must contain at least one frame");
    const size_t sampled_frames = std::min(total_frames, num_frames);
    std::vector<size_t> indices;
    indices.reserve(sampled_frames);
    const float step = static_cast<float>(total_frames) / sampled_frames;
    for (size_t frame_idx = 0; frame_idx < sampled_frames; ++frame_idx) {
        indices.push_back(std::min(static_cast<size_t>(frame_idx * step), total_frames - 1));
    }
    return indices;
}

}  // namespace

std::pair<ov::Tensor, ov::genai::VideoMetadata> utils::load_video(const fs::path& video_path,
                                                                  size_t num_frames) {
    cv::VideoCapture capture(video_path.string());
    OPENVINO_ASSERT(capture.isOpened(), "Could not open video file: ", video_path.string());

    const size_t total_frames = static_cast<size_t>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    const size_t width = static_cast<size_t>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const size_t height = static_cast<size_t>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    OPENVINO_ASSERT(total_frames > 0 && width > 0 && height > 0,
                    "Could not read video metadata: ",
                    video_path.string());

    ov::genai::VideoMetadata metadata;
    metadata.fps = static_cast<float>(capture.get(cv::CAP_PROP_FPS));
    metadata.frames_indices = make_indices(total_frames, num_frames);

    ov::Tensor video(ov::element::u8, {total_frames, height, width, 3});
    uint8_t* destination = video.data<uint8_t>();
    cv::Mat frame;
    size_t decoded_frames = 0;
    while (capture.read(frame)) {
        OPENVINO_ASSERT(static_cast<size_t>(frame.cols) == width &&
                            static_cast<size_t>(frame.rows) == height && frame.channels() == 3,
                        "Video frame layout changed while decoding: ",
                        video_path.string());
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        std::memcpy(destination, rgb.data, rgb.total() * rgb.elemSize());
        destination += rgb.total() * rgb.elemSize();
        ++decoded_frames;
    }
    OPENVINO_ASSERT(decoded_frames == total_frames,
                    "Frame count mismatch: expected ",
                    total_frames,
                    ", got ",
                    decoded_frames);
    return {video, metadata};
}

std::pair<std::vector<ov::Tensor>, std::vector<ov::genai::VideoMetadata>> utils::load_videos(
    const fs::path& input_path,
    size_t num_frames) {
    OPENVINO_ASSERT(!input_path.empty() && fs::exists(input_path),
                    "Path to videos is empty or does not exist: ",
                    input_path.string());
    if (fs::is_directory(input_path)) {
        const std::set<fs::path> sorted_paths{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> videos;
        std::vector<ov::genai::VideoMetadata> metadata;
        videos.reserve(sorted_paths.size());
        metadata.reserve(sorted_paths.size());
        for (const fs::path& path : sorted_paths) {
            auto [video, video_metadata] = load_video(path, num_frames);
            videos.push_back(std::move(video));
            metadata.push_back(std::move(video_metadata));
        }
        return {std::move(videos), std::move(metadata)};
    }

    auto [video, metadata] = load_video(input_path, num_frames);
    return {{std::move(video)}, {std::move(metadata)}};
}
