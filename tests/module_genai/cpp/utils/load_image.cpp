
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <sstream>
#include <opencv2/opencv.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "load_image.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

std::vector<ov::Tensor> utils::load_images(const std::filesystem::path& input_path) {
    if (input_path.empty() || !fs::exists(input_path)) {
        throw std::runtime_error{"Path to images is empty or does not exist."};
    }
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_images{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> images;
        for (const fs::path& dir_entry : sorted_images) {
            images.push_back(utils::load_image(dir_entry));
        }
        return images;
    }
    return {utils::load_image(input_path)};
}

ov::Tensor utils::load_video(const std::filesystem::path& input_path) {
    auto rgbs = load_images(input_path);
    if (rgbs.size() == 0) {
        return {};
    }

    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{rgbs.size(), rgbs[0].get_shape()[1], rgbs[0].get_shape()[2], rgbs[0].get_shape()[3]});
    std::cout << "video.shape = " << video.get_shape() << std::endl;

    auto stride = rgbs[0].get_byte_size();
    std::cout << "stride = " << stride << std::endl;
    auto dst = reinterpret_cast<char*>(video.data());
    int b = 0;
    for (auto rgb : rgbs)
    {
        std::memcpy(dst + stride * b, rgb.data(), stride);
        b++;
    }
    return video;
}

// Return video with shape: [num_frames, height, width, 3]
ov::Tensor utils::create_countdown_frames()
{
    int frames_count = 5, height = 240, width = 360;
    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{(size_t)frames_count, (size_t)height, (size_t)width, 3});

    for (int i = frames_count; i > 0; i--)
    {
        cv::Mat frame = cv::Mat::zeros(height, width, CV_8UC3);
        std::string text = std::to_string(i);

        int baseline = 0;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 3.0; // Python '3' is a double in C++ OpenCV
        int thickness = 4;

        // The C++ getTextSize returns the size as a cv::Size
        cv::Size textSize = cv::getTextSize(
            text,
            fontFace,
            fontScale,
            thickness,
            &baseline // baseline is passed by pointer
        );

        int text_width = textSize.width;
        int text_height = textSize.height;
        int text_x = (width - text_width) / 2;
        int text_y = (height + text_height) / 2;

        cv::Scalar color = cv::Scalar(255, 255, 255); // BGR: White
        cv::Point org(text_x, text_y);                // Origin point for the text

        cv::putText(
            frame,
            text,
            org,
            fontFace,
            fontScale,
            color,
            thickness,
            cv::LINE_AA // The line type constant
        );

        int idx = frames_count - i;
        std::memcpy((char*)video.data() + idx * height * width * 3, frame.data, height * width * 3);

        // cv::imshow("Centered Text Frame", frame);
        // cv::waitKey(0);
    }
    return video;
}

ov::Tensor utils::load_image(const std::filesystem::path &image_path)
{
    int x = 0, y = 0, channels_in_file = 0;
    constexpr int desired_channels = 3;
    unsigned char *image = stbi_load(
        image_path.string().c_str(),
        &x, &y, &channels_in_file, desired_channels);
    if (!image)
    {
        std::stringstream error_message;
        error_message << "Failed to load the image '" << image_path << "'";
        throw std::runtime_error{error_message.str()};
    }
    struct SharedImageAllocator
    {
        unsigned char *image;
        int channels, height, width;
        void *allocate(size_t bytes, size_t) const
        {
            if (image && channels * height * width == bytes)
            {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void *, size_t, size_t) noexcept
        {
            stbi_image_free(image);
            image = nullptr;
        }
        bool is_equal(const SharedImageAllocator &other) const noexcept { return this == &other; }
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(y), size_t(x), size_t(desired_channels)},
        SharedImageAllocator{image, desired_channels, y, x});
}

namespace TEST_DATA {

std::string img_cat_120_100() {
    std::string full_path = get_data_path() + "/cat_120_100.png";
    OPENVINO_ASSERT(check_file_exists(full_path), "File does not exist: " + full_path);
    return full_path;
}

std::string img_dog_120_120() {
    std::string full_path = get_data_path() + "/dog_120_120.png";
    OPENVINO_ASSERT(check_file_exists(full_path), "File does not exist: " + full_path);
    return full_path;
}

ov::Tensor audio_dummy_data(float duration, int sample_rate) {
    int total_samples = static_cast<int>(sample_rate * duration);
    auto audio = ov::Tensor(ov::element::f32, ov::Shape{(size_t)total_samples});
    float* audio_data = audio.data<float>();
    for (int i = 0; i < total_samples; ++i) {
        audio_data[i] = i / static_cast<float>(total_samples);
    }
    return audio;
}
}  // namespace TEST_DATA