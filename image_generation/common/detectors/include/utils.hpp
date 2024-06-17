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
template <typename T>
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
    ov::Tensor output_tensor(ov::element::from<T>(), output_shape);
    T* output_data = output_tensor.data<T>();
    const T* input_data = input_tensor.data<T>();

    // Process each channel separately
    for (int c = 0; c < Co; ++c) {
        // Extract single channel
        cv::Mat channel(Ho, Wo, cv::DataType<T>::type);
        for (int h = 0; h < Ho; ++h) {
            for (int w = 0; w < Wo; ++w) {
                channel.at<T>(h, w) = input_data[h * Wo * Co + w * Co + c];
            }
        }
        // Resize the single channel
        cv::Mat resized_channel = resize_single_channel(channel, Ht, Wt, k);

        // Copy resized channel back to the output tensor
        for (int h = 0; h < Ht; ++h) {
            for (int w = 0; w < Wt; ++w) {
                output_data[h * Wt * Co + w * Co + c] = resized_channel.at<T>(h, w);
            }
        }
    }

    return output_tensor;
}

ov::Tensor smart_resize(const ov::Tensor& input_tensor, int Ht, int Wt) {
    if (input_tensor.get_element_type() == ov::element::u8) {
        return smart_resize<uint8_t>(input_tensor, Ht, Wt);
    } else if (input_tensor.get_element_type() == ov::element::f32) {
        return smart_resize<float>(input_tensor, Ht, Wt);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
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

// Function to crop the tensor
template <typename T>
ov::Tensor crop_right_down_corner(const ov::Tensor& input, std::vector<int> pad) {
    // Get input tensor dimensions (NHWC)
    auto input_shape = input.get_shape();
    auto N = input_shape[0];
    auto H = input_shape[1];
    auto W = input_shape[2];
    auto C = input_shape[3];

    int down = pad[2];
    int right = pad[3];

    // Calculate new dimensions
    int H_new = H - down;
    int W_new = W - right;

    // Create a new tensor with the new dimensions
    ov::Shape output_shape = {N, static_cast<unsigned long>(H_new), static_cast<unsigned long>(W_new), C};
    ov::Tensor output(ov::element::from<T>(), output_shape);

    T* cropped_data = output.data<T>();
    const T* input_data = input.data<T>();

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_new; ++h) {
            for (int w = 0; w < W_new; ++w) {
                for (int c = 0; c < C; ++c) {
                    int src_idx = ((n * H + h) * W + w) * C + c;
                    int dst_idx = ((n * H_new + h) * W_new + w) * C + c;
                    cropped_data[dst_idx] = input_data[src_idx];
                }
            }
        }
    }

    return output;
}

// Overloaded function to handle specific conversions
ov::Tensor crop_right_down_corner(const ov::Tensor& input, const std::vector<int>& pad) {
    if (input.get_element_type() == ov::element::u8) {
        return crop_right_down_corner<uint8_t>(input, pad);
    } else if (input.get_element_type() == ov::element::f32) {
        return crop_right_down_corner<float>(input, pad);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
}

// Function to convert uint8_t rgb tensor to float32 normalized tensor
ov::Tensor normalize_rgb_tensor(const ov::Tensor& input) {
    ov::Tensor output(ov::element::f32, input.get_shape());

    float* output_data = output.data<float>();
    const uint8_t* input_data = input.data<uint8_t>();

    auto input_shape = input.get_shape();
    auto N = input_shape[0];
    auto H = input_shape[1];
    auto W = input_shape[2];
    auto C = input_shape[3];
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    int idx = ((n * H + h) * W + w) * C + c;
                    output_data[idx] = static_cast<float>(input_data[idx]) / 256 - 0.5;
                }
            }
        }
    }

    return output;
}

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

template <typename T>
ov::Tensor cv_gaussian_blur(const ov::Tensor& input_tensor, int sigma) {
    // Get input tensor dimensions (NHWC)
    // Assume N and C are always 1 (we apply it to each channel of the heatmap)
    auto input_shape = input_tensor.get_shape();
    auto H = input_shape[1];
    auto W = input_shape[2];

    // Convert input tensor data pointer to cv::Mat
    const T* input_data = input_tensor.data<T>();
    cv::Mat img(H, W, cv::DataType<T>::type, const_cast<T*>(input_data));

    // Calculate kernel size
    int truncate = 4;
    int radius = static_cast<int>(truncate * sigma + 0.5);
    int ksize = 2 * radius + 1;

    // Apply Gaussian blur
    cv::Mat blurred_img;
    cv::GaussianBlur(img, blurred_img, cv::Size(ksize, ksize), sigma, sigma);

    // Copy blurred image data back to output tensor
    ov::Tensor output_tensor(input_tensor.get_element_type(), input_shape);
    T* output_data = output_tensor.data<T>();
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            output_data[i * W + j] = blurred_img.at<T>(i, j);
        }
    }

    return output_tensor;
}

ov::Tensor cv_gaussian_blur(const ov::Tensor& input, int sigma) {
    if (input.get_element_type() == ov::element::u8) {
        return cv_gaussian_blur<uint8_t>(input, sigma);
    } else if (input.get_element_type() == ov::element::f32) {
        return cv_gaussian_blur<float>(input, sigma);
    } else {
        throw std::runtime_error("Unsupported tensor type");
    }
}

// find the peaks from heatmap, returns a vector of tuple
// (x, y, score, id)
std::vector<std::vector<std::tuple<int, int, float, int>>> find_heatmap_peaks(const ov::Tensor& heatmap_avg /* f32 */,
                                                                              float thre1) {
    auto heatmap_shape = heatmap_avg.get_shape();
    auto H = heatmap_shape[1];
    auto W = heatmap_shape[2];
    auto C = heatmap_shape[3];

    std::vector<std::vector<std::tuple<int, int, float, int>>> all_peaks;
    int peak_counter = 0;

    for (int c = 0; c < C - 1; ++c) {
        // Create a new shape for the single channel tensor
        ov::Tensor single_channel_heatmap(heatmap_avg.get_element_type(), {1, H, W, 1});

        // Copy the data for the current channel to the new tensor
        const auto* input_data = heatmap_avg.data<float>();
        auto* single_channel_data = single_channel_heatmap.data<float>();

        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                single_channel_data[h * W + w] = input_data[h * W * C + w * C + c];
            }
        }

        // Apply Gaussian blur
        ov::Tensor one_heatmap = cv_gaussian_blur(single_channel_heatmap, 3);

        // Create directional maps
        ov::Shape shape = one_heatmap.get_shape();

        ov::Tensor map_left(one_heatmap.get_element_type(), shape);
        ov::Tensor map_right(one_heatmap.get_element_type(), shape);
        ov::Tensor map_up(one_heatmap.get_element_type(), shape);
        ov::Tensor map_down(one_heatmap.get_element_type(), shape);

        const auto one_heatmap_data = one_heatmap.data<float>();
        auto map_left_data = map_left.data<float>();
        auto map_right_data = map_right.data<float>();
        auto map_up_data = map_up.data<float>();
        auto map_down_data = map_down.data<float>();

        for (size_t h = 1; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                map_left_data[h * W + w] = one_heatmap_data[(h - 1) * W + w];
            }
        }

        for (size_t h = 0; h < H - 1; ++h) {
            for (size_t w = 0; w < W; ++w) {
                map_right_data[h * W + w] = one_heatmap_data[(h + 1) * W + w];
            }
        }

        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 1; w < W; ++w) {
                map_up_data[h * W + w] = one_heatmap_data[h * W + w - 1];
            }
        }

        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W - 1; ++w) {
                map_down_data[h * W + w] = one_heatmap_data[h * W + w + 1];
            }
        }

        std::vector<std::tuple<int, int, float, int>> peaks_with_score_and_id;
        std::vector<std::tuple<int, int, float>> peaks_with_score;

        for (size_t h = 1; h < H - 1; ++h) {
            for (size_t w = 1; w < W - 1; ++w) {
                if (one_heatmap_data[h * W + w] >= map_left_data[h * W + w] &&
                    one_heatmap_data[h * W + w] >= map_right_data[h * W + w] &&
                    one_heatmap_data[h * W + w] >= map_up_data[h * W + w] &&
                    one_heatmap_data[h * W + w] >= map_down_data[h * W + w] && one_heatmap_data[h * W + w] > thre1) {
                    peaks_with_score.emplace_back(w, h, one_heatmap_data[h * W + w]);
                }
            }
        }

        for (auto& peak : peaks_with_score) {
            peaks_with_score_and_id.emplace_back(std::get<0>(peak),
                                                 std::get<1>(peak),
                                                 std::get<2>(peak),
                                                 peak_counter++);
        }

        all_peaks.push_back(peaks_with_score_and_id);
    }

    return all_peaks;
}

std::tuple<std::vector<std::vector<std::tuple<int, int, float, int, int>>>, std::vector<int>> calculate_connections(
    const ov::Tensor& paf_avg,
    const std::vector<std::vector<std::tuple<int, int, float, int>>>& all_peaks,
    const ov::Tensor& oriImg,
    float thre2) {
    const int mid_num = 10;
    std::vector<std::vector<int>> limbSeq = {{2, 3},
                                             {2, 6},
                                             {3, 4},
                                             {4, 5},
                                             {6, 7},
                                             {7, 8},
                                             {2, 9},
                                             {9, 10},
                                             {10, 11},
                                             {2, 12},
                                             {12, 13},
                                             {13, 14},
                                             {2, 1},
                                             {1, 15},
                                             {15, 17},
                                             {1, 16},
                                             {16, 18},
                                             {3, 17},
                                             {6, 18}};

    std::vector<std::vector<int>> mapIdx = {{31, 32},
                                            {39, 40},
                                            {33, 34},
                                            {35, 36},
                                            {41, 42},
                                            {43, 44},
                                            {19, 20},
                                            {21, 22},
                                            {23, 24},
                                            {25, 26},
                                            {27, 28},
                                            {29, 30},
                                            {47, 48},
                                            {49, 50},
                                            {53, 54},
                                            {51, 52},
                                            {55, 56},
                                            {37, 38},
                                            {45, 46}};
    std::vector<std::vector<std::tuple<int, int, float, int, int>>> connection_all;
    std::vector<int> special_k;

    auto paf_shape = paf_avg.get_shape();
    auto H = paf_shape[1];
    auto W = paf_shape[2];
    auto C = paf_shape[3];

    auto paf_data = paf_avg.data<float>();

    for (size_t k = 0; k < mapIdx.size(); ++k) {
        auto score_mid_x_channel = (mapIdx[k][0] - 19);
        auto score_mid_y_channel = (mapIdx[k][1] - 19);

        const auto& candA = all_peaks[limbSeq[k][0] - 1];
        const auto& candB = all_peaks[limbSeq[k][1] - 1];
        int nA = candA.size();
        int nB = candB.size();

        if (nA == 0 || nB == 0) {
            special_k.push_back(k);
            connection_all.push_back({});
            continue;
        }

        std::vector<std::tuple<int, int, float, float>> connection_candidate;

        for (int i = 0; i < nA; ++i) {
            for (int j = 0; j < nB; ++j) {
                float vec_x = std::get<0>(candB[j]) - std::get<0>(candA[i]);
                float vec_y = std::get<1>(candB[j]) - std::get<1>(candA[i]);
                float norm = std::sqrt(vec_x * vec_x + vec_y * vec_y);
                norm = std::max(0.001f, norm);
                vec_x /= norm;
                vec_y /= norm;

                std::vector<std::pair<float, float>> startend(mid_num);
                for (int l = 0; l < mid_num; ++l) {
                    startend[l].first = std::get<0>(candA[i]) + l * vec_x / (mid_num - 1);
                    startend[l].second = std::get<1>(candA[i]) + l * vec_y / (mid_num - 1);
                }

                std::vector<float> vec_scores(mid_num);
                for (int l = 0; l < mid_num; ++l) {
                    int x = std::round(startend[l].first);
                    int y = std::round(startend[l].second);
                    x = std::clamp(x, 0, int(W) - 1);
                    y = std::clamp(y, 0, int(H) - 1);

                    float score_mid_pts_x = paf_data[y * W * C + x * C + score_mid_x_channel];
                    float score_mid_pts_y = paf_data[y * W * C + x * C + score_mid_y_channel];
                    vec_scores[l] = vec_x * score_mid_pts_x + vec_y * score_mid_pts_y;
                }

                float score_with_dist_prior = std::accumulate(vec_scores.begin(), vec_scores.end(), 0.0f) / mid_num +
                                              std::min(0.5f * oriImg.get_shape()[1] / norm - 1.0f, 0.0f);
                int criterion1 = std::count_if(vec_scores.begin(), vec_scores.end(), [thre2](float v) {
                                     return v > thre2;
                                 }) > 0.8f * vec_scores.size();
                int criterion2 = score_with_dist_prior > 0;

                if (criterion1 && criterion2) {
                    connection_candidate.emplace_back(
                        i,
                        j,
                        score_with_dist_prior,
                        score_with_dist_prior + std::get<2>(candA[i]) + std::get<2>(candB[j]));
                }
            }
        }

        std::sort(connection_candidate.begin(), connection_candidate.end(), [](const auto& a, const auto& b) {
            return std::get<2>(a) > std::get<2>(b);
        });

        std::vector<std::tuple<int, int, float, int, int>> connection;
        for (const auto& candidate : connection_candidate) {
            int i = std::get<0>(candidate);
            int j = std::get<1>(candidate);
            float score = std::get<2>(candidate);
            if (std::none_of(connection.begin(), connection.end(), [i, j](const auto& conn) {
                    return std::get<3>(conn) == i || std::get<4>(conn) == j;
                })) {
                connection.emplace_back(std::get<3>(candA[i]), std::get<3>(candB[j]), score, i, j);
                if (connection.size() >= std::min(nA, nB)) {
                    break;
                }
            }
        }

        connection_all.push_back(connection);
    }

    return {connection_all, special_k};
}