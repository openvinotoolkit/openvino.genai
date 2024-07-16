// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>

// Helper function to initialize a tensor with zeros
ov::Tensor init_tensor_with_zeros(const ov::Shape& shape, ov::element::Type type);

std::vector<uint8_t> read_bgr_from_txt(const std::string& file_name);

// Function to resize a single channel, could use libeigen or even openvino later
cv::Mat resize_single_channel(const cv::Mat& channel, int Ht, int Wt, float k);

// Smart resize function
template <typename T>
ov::Tensor smart_resize(const ov::Tensor& input_tensor, int Ht, int Wt);

// Overload
ov::Tensor smart_resize(const ov::Tensor& input_tensor, int Ht, int Wt);

ov::Tensor smart_resize_k(const ov::Tensor& input_tensor, float fx, float fy);

// Function to pad the tensor
std::pair<ov::Tensor, std::vector<int>> pad_right_down_corner(const ov::Tensor& img, int stride, uint8_t pad_val);

// Function to crop the tensor
template <typename T>
ov::Tensor crop_right_down_corner(const ov::Tensor& input, std::vector<int> pad);

// Overloaded function to handle specific conversions
ov::Tensor crop_right_down_corner(const ov::Tensor& input, const std::vector<int>& pad);

// Function to convert uint8_t rgb tensor to float32 normalized tensor
ov::Tensor normalize_rgb_tensor(const ov::Tensor& input);

template <typename T>
ov::Tensor cv_gaussian_blur(const ov::Tensor& input_tensor, int sigma);

ov::Tensor cv_gaussian_blur(const ov::Tensor& input, int sigma);

template <typename T>
void reshape_tensor(const ov::Tensor& input, ov::Tensor& output, const std::vector<size_t>& new_order) {
    const auto& input_shape = input.get_shape();
    auto input_data = input.data<T>();
    auto output_data = output.data<T>();

    // Calculate strides for input and output tensors
    std::vector<size_t> input_strides(input_shape.size(), 1);
    for (int i = input_shape.size() - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    const auto& output_shape = output.get_shape();
    std::vector<size_t> output_strides(output_shape.size(), 1);
    for (int i = output_shape.size() - 2; i >= 0; --i) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Helper function to calculate flat index in the tensor
    auto calculate_index = [&](const std::vector<size_t>& shape, const std::vector<size_t>& indices) {
        size_t index = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            index += indices[i] * shape[i];
        }
        return index;
    };

    // Iterate over the input tensor and copy data to the output tensor based on new_order
    std::vector<size_t> input_indices(input_shape.size(), 0);
    std::vector<size_t> output_indices(output_shape.size(), 0);
    for (size_t n = 0; n < input_shape[0]; ++n) {
        for (size_t c = 0; c < input_shape[1]; ++c) {
            for (size_t h = 0; h < input_shape[2]; ++h) {
                for (size_t w = 0; w < input_shape[3]; ++w) {
                    input_indices = {n, c, h, w};
                    for (size_t i = 0; i < new_order.size(); ++i) {
                        output_indices[i] = input_indices[new_order[i]];
                    }
                    output_data[calculate_index(output_strides, output_indices)] =
                        input_data[calculate_index(input_strides, input_indices)];
                }
            }
        }
    }
}

ov::Tensor read_image_to_tensor(const std::string& image_path);