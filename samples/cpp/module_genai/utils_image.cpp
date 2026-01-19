// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils_image.hpp"

#include <sstream>
#include <set>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace image_utils {

// ============================================================================
// Image Loading Functions
// ============================================================================

ov::Tensor load_image(const std::filesystem::path& image_path) {
    int x = 0, y = 0, channels_in_file = 0;
    constexpr int desired_channels = 3;
    unsigned char* image = stbi_load(
        image_path.string().c_str(),
        &x, &y, &channels_in_file, desired_channels);
    if (!image) {
        std::stringstream error_message;
        error_message << "Failed to load the image '" << image_path << "'";
        throw std::runtime_error{error_message.str()};
    }
    struct SharedImageAllocator {
        unsigned char* image;
        int channels, height, width;
        void* allocate(size_t bytes, size_t) const {
            if (image && channels * height * width == bytes) {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void*, size_t, size_t) noexcept {
            stbi_image_free(image);
            image = nullptr;
        }
        bool is_equal(const SharedImageAllocator& other) const noexcept { return this == &other; }
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(y), size_t(x), size_t(desired_channels)},
        SharedImageAllocator{image, desired_channels, y, x});
}

std::vector<ov::Tensor> load_images(const std::filesystem::path& input_path) {
    if (input_path.empty() || !fs::exists(input_path)) {
        throw std::runtime_error{"Path to images is empty or does not exist."};
    }
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_images{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> images;
        for (const fs::path& dir_entry : sorted_images) {
            images.push_back(load_image(dir_entry));
        }
        return images;
    }
    return {load_image(input_path)};
}

ov::Tensor load_video(const std::filesystem::path& input_path) {
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
    for (auto rgb : rgbs) {
        std::memcpy(dst + stride * b, rgb.data(), stride);
        b++;
    }
    return video;
}

ov::Tensor create_countdown_frames() {
    int frames_count = 5, height = 240, width = 360;
    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{(size_t)frames_count, (size_t)height, (size_t)width, 3});

    for (int i = frames_count; i > 0; i--) {
        cv::Mat frame = cv::Mat::zeros(height, width, CV_8UC3);
        std::string text = std::to_string(i);

        int baseline = 0;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 3.0;
        int thickness = 4;

        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

        int text_width = textSize.width;
        int text_height = textSize.height;
        int text_x = (width - text_width) / 2;
        int text_y = (height + text_height) / 2;

        cv::Scalar color = cv::Scalar(255, 255, 255);
        cv::Point org(text_x, text_y);

        cv::putText(frame, text, org, fontFace, fontScale, color, thickness, cv::LINE_AA);

        int idx = frames_count - i;
        std::memcpy((char*)video.data() + idx * height * width * 3, frame.data, height * width * 3);
    }
    return video;
}

// ============================================================================
// Image Saving Functions
// ============================================================================

bool save_image_bmp(const std::string& filename, const ov::Tensor& image, bool convert_rgb2bgr) {
    try {
        ov::Shape shape = image.get_shape();

        size_t height, width, channels;
        const uint8_t* data = image.data<const uint8_t>();

        if (shape.size() == 4) {
            if (shape[3] == 3) {
                height = shape[1];
                width = shape[2];
                channels = shape[3];
            } else if (shape[1] == 3) {
                std::cerr << "[ERROR] NCHW format not supported for BMP save" << std::endl;
                return false;
            } else {
                std::cerr << "[ERROR] Unknown 4D tensor format" << std::endl;
                return false;
            }
        } else if (shape.size() == 3) {
            height = shape[0];
            width = shape[1];
            channels = shape[2];
        } else {
            std::cerr << "[ERROR] Unsupported tensor shape for image save" << std::endl;
            return false;
        }

        if (channels != 3) {
            std::cerr << "[ERROR] Expected 3 channels, got " << channels << std::endl;
            return false;
        }

        unsigned char file_header[14] = {
            'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0
        };

        unsigned char info_header[40] = {
            40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        };

        int row_padding = (4 - (width * 3) % 4) % 4;
        int data_size = static_cast<int>((width * 3 + row_padding) * height);
        int file_size = 54 + data_size;

        file_header[2] = file_size & 0xFF;
        file_header[3] = (file_size >> 8) & 0xFF;
        file_header[4] = (file_size >> 16) & 0xFF;
        file_header[5] = (file_size >> 24) & 0xFF;

        info_header[4] = width & 0xFF;
        info_header[5] = (width >> 8) & 0xFF;
        info_header[6] = (width >> 16) & 0xFF;
        info_header[7] = (width >> 24) & 0xFF;

        int32_t neg_height = -static_cast<int32_t>(height);
        info_header[8] = neg_height & 0xFF;
        info_header[9] = (neg_height >> 8) & 0xFF;
        info_header[10] = (neg_height >> 16) & 0xFF;
        info_header[11] = (neg_height >> 24) & 0xFF;

        info_header[20] = data_size & 0xFF;
        info_header[21] = (data_size >> 8) & 0xFF;
        info_header[22] = (data_size >> 16) & 0xFF;
        info_header[23] = (data_size >> 24) & 0xFF;

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Failed to open file: " << filename << std::endl;
            return false;
        }

        file.write(reinterpret_cast<char*>(file_header), 14);
        file.write(reinterpret_cast<char*>(info_header), 40);

        unsigned char padding[3] = {0, 0, 0};

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = (y * width + x) * 3;
                if (convert_rgb2bgr) {
                    file.put(static_cast<char>(data[idx + 2]));
                    file.put(static_cast<char>(data[idx + 1]));
                    file.put(static_cast<char>(data[idx]));
                } else {
                    file.write(reinterpret_cast<const char*>(data + idx), 3);
                }
            }
            file.write(reinterpret_cast<char*>(padding), row_padding);
        }

        file.close();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to save image: " << e.what() << std::endl;
        return false;
    }
}

std::string generate_output_filename(const std::string& prefix, const std::string& suffix) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    std::ostringstream oss;
    oss << prefix << "_"
        << std::put_time(&tm_now, "%Y%m%d_%H%M%S")
        << suffix;
    return oss.str();
}

} // namespace image_utils
