// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>

std::vector<uint8_t> read_bgr_from_txt(const std::string& file_name) {
    std::ifstream input_data(file_name, std::ifstream::in);

    std::vector<uint8_t> res;
    std::string line;
    while (std::getline(input_data, line)) {
        try {
            int value = std::stoi(line);
            if (value < 0 || value > 255) {
                std::cerr << "invalid uint8: " << value << std::endl;
                continue;
            }
            res.push_back(static_cast<uint8_t>(value));
        } catch (const std::invalid_argument& e) {
            std::cerr << "invalid line: " << line << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "out of range: " << line << std::endl;
        }
    }

    return res;
}

// Function to resize a single channel, could use libeigen or even openvino later
cv::Mat resize_single_channel(const cv::Mat& channel, int Ht, int Wt, float k) {
    cv::Mat resized_channel;
    int interpolation = (k < 1) ? cv::INTER_AREA : cv::INTER_LANCZOS4;
    cv::resize(channel, resized_channel, cv::Size(Wt, Ht), 0, 0, interpolation);
    return resized_channel;
}

// Smart resize function
ov::Tensor smart_resize(const ov::Tensor& input_tensor, int Ht, int Wt) {
    // Get input tensor dimensions (NHWC)
    auto input_shape = input_tensor.get_shape();
    auto N = input_shape[0];
    auto Ho = input_shape[1];
    auto Wo = input_shape[2];
    auto Co = input_shape[3];

    // Determine the scaling factor
    float k = static_cast<float>(Ht + Wt) / static_cast<float>(Ho + Wo);

    // Prepare the output tensor
    ov::Shape output_shape = {N, static_cast<unsigned long>(Ht), static_cast<unsigned long>(Wt), Co};
    ov::Tensor output_tensor(ov::element::u8, output_shape);
    uint8_t* output_data = output_tensor.data<uint8_t>();
    const uint8_t* input_data = input_tensor.data<uint8_t>();

    // Process each channel separately
    for (int c = 0; c < Co; ++c) {
        // Extract single channel
        cv::Mat channel(Ho, Wo, CV_8UC1);
        for (int h = 0; h < Ho; ++h) {
            for (int w = 0; w < Wo; ++w) {
                channel.at<uint8_t>(h, w) = input_data[h * Wo * Co + w * Co + c];
            }
        }
        // Resize the single channel
        cv::Mat resized_channel = resize_single_channel(channel, Ht, Wt, k);

        // Copy resized channel back to the output tensor
        for (int h = 0; h < Ht; ++h) {
            for (int w = 0; w < Wt; ++w) {
                output_data[h * Wt * Co + w * Co + c] = resized_channel.at<uint8_t>(h, w);
            }
        }
    }

    return output_tensor;
}

ov::Tensor smart_resize_k(const ov::Tensor& input_tensor, float fx, float fy) {
    auto input_shape = input_tensor.get_shape();
    auto Ho = input_shape[1];
    auto Wo = input_shape[2];
    int Ht = Ho * fy;
    int Wt = Wo * fx;
    return smart_resize(input_tensor, Ht, Wt);
}

// Function to pad the tensor
std::pair<ov::Tensor, std::vector<int>> pad_right_down_corner(const ov::Tensor& img, int stride, uint8_t pad_val) {
    // Get input tensor dimensions (NHWC)
    auto input_shape = img.get_shape();
    auto N = input_shape[0];
    auto H = input_shape[1];
    auto W = input_shape[2];
    auto C = input_shape[3];

    // Calculate padding sizes
    std::vector<int> pad(4);
    pad[0] = 0;                                              // up
    pad[1] = 0;                                              // left
    pad[2] = (H % stride == 0) ? 0 : stride - (H % stride);  // down
    pad[3] = (W % stride == 0) ? 0 : stride - (W % stride);  // right

    // Calculate new dimensions
    int H_new = H + pad[0] + pad[2];
    int W_new = W + pad[1] + pad[3];

    // Create a new tensor with the new dimensions
    ov::Shape output_shape = {N, static_cast<unsigned long>(H_new), static_cast<unsigned long>(W_new), C};
    ov::Tensor img_padded(ov::element::u8, output_shape);

    // Initialize img_padded with padValue
    uint8_t* padded_data = img_padded.data<uint8_t>();
    std::fill(padded_data, padded_data + N * H_new * W_new * C, pad_val);

    // Copy the original image into the new padded image
    const uint8_t* img_data = img.data<uint8_t>();

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    int src_idx = ((n * H + h) * W + w) * C + c;
                    int dst_idx = ((n * H_new + (h + pad[0])) * W_new + (w + pad[1])) * C + c;
                    padded_data[dst_idx] = img_data[src_idx];
                }
            }
        }
    }

    return {img_padded, pad};
}