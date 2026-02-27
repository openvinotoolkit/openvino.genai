// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "imwrite_video.hpp"

void save_video(const std::string& filename,
                const ov::Tensor& video_tensor,  // [B, F, H, W, C], u8
                float fps) {
    const ov::Shape shape = video_tensor.get_shape();

    if (shape.empty() || video_tensor.get_size() == 0) {
        throw std::runtime_error("save_video(): input tensor is empty, skip saving: " + filename);
    }

    const size_t B = shape[0], F = shape[1], H = shape[2], W = shape[3], C = shape[4];
    const uint8_t* video_data = video_tensor.data<const uint8_t>();

    for (size_t b = 0; b < B; ++b) {
        std::string out = filename;
        if (B != 1) {
            std::filesystem::path p(filename);
            std::string ext = p.has_extension() ? p.extension().string() : ".avi";
            out = (p.parent_path() / (p.stem().string() + "_b" + std::to_string(b) + ext)).string();
        }

        const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        cv::VideoWriter writer(out, fourcc, static_cast<double>(fps), cv::Size(W, H), true);
        if (!writer.isOpened())
            throw std::runtime_error("VideoWriter failed to open: " + out);

        const size_t frame_bytes = H * W * C;
        const size_t batch_stride = F * frame_bytes;
        const uint8_t* batch_ptr = video_data + b * batch_stride;

        for (size_t f = 0; f < F; ++f) {
            const uint8_t* frame_ptr = batch_ptr + f * frame_bytes;

            cv::Mat src(H, W, CV_8UC3, const_cast<uint8_t*>(frame_ptr));
            cv::Mat bgr;
            cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);

            writer.write(bgr);
        }
    }
}
