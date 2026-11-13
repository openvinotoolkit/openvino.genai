// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "openvino/genai/video_generation/video2video_pipeline.hpp"

// Video super-resolution (FlashVSR, x4): reads the input video frame by frame, streams the frames through the
// pipeline and writes the upscaled frames as soon as they are available.
int main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && argc <= 5, "Usage: ", argv[0], " <MODEL_DIR> <INPUT_VIDEO> [OUTPUT_VIDEO] [DEVICE]");
    const std::string models_path = argv[1], input_path = argv[2];
    const std::string output_path = argc > 3 ? argv[3] : "genai_video2video.avi";
    const std::string device = argc > 4 ? argv[4] : "CPU";  // GPU can be used as well

    cv::VideoCapture capture(input_path);
    OPENVINO_ASSERT(capture.isOpened(), "Cannot open ", input_path);
    const double fps = capture.get(cv::CAP_PROP_FPS) > 0 ? capture.get(cv::CAP_PROP_FPS) : 25.0;

    ov::genai::VideoToVideoPipeline pipe(models_path, device);
    pipe.start_stream(ov::genai::sparse_attention_ratio(2.0f));

    cv::VideoWriter writer;
    size_t written = 0;
    auto write_frames = [&](ov::Tensor frames) {
        const auto& shape = frames.get_shape();  // [n, H, W, 3] RGB
        for (size_t i = 0; i < shape[0]; ++i) {
            if (!writer.isOpened()) {
                const cv::Size size(static_cast<int>(shape[2]), static_cast<int>(shape[1]));
                writer.open(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size);
                OPENVINO_ASSERT(writer.isOpened(), "Cannot open ", output_path, " for writing");
            }
            cv::Mat rgb(static_cast<int>(shape[1]),
                        static_cast<int>(shape[2]),
                        CV_8UC3,
                        frames.data<uint8_t>() + i * shape[1] * shape[2] * 3);
            cv::Mat bgr;
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            writer.write(bgr);
            ++written;
        }
    };

    cv::Mat bgr, rgb;
    size_t read = 0;
    while (capture.read(bgr)) {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        ov::Tensor frame(ov::element::u8, ov::Shape{static_cast<size_t>(rgb.rows), static_cast<size_t>(rgb.cols), 3});
        std::memcpy(frame.data(), rgb.data, frame.get_byte_size());
        write_frames(pipe.push_frames(frame));
        ++read;
    }
    write_frames(pipe.end_stream());

    const auto metrics = pipe.get_performance_metrics();
    std::cout << "Upscaled " << read << " frames to " << output_path << " (" << written << " frames)\n"
              << "Load time: " << metrics.load_time << " ms, processing: " << metrics.generate_duration << " ms\n";
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
