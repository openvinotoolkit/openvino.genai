// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openpose_detector.hpp"

#include <cstdint>
#include <iostream>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>

#include "openvino/runtime/core.hpp"
#include "utils.hpp"

const std::vector<std::vector<int>> OpenposeDetector::limbSeq = {{2, 3},
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

const std::vector<std::vector<int>> OpenposeDetector::mapIdx = {{31, 32},
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

void OpenposeDetector::load(const std::string& model_path) {
    std::cout << "Loading model from: " << model_path << std::endl;

    ov::Core core;
    std::string device = "CPU";
    auto model = core.read_model(model_path + "/openpose.xml");
    // TODO: W / H dimension should be dynamic, we reshape it before comlile
    body_model = core.compile_model(model, device);
}

std::pair<ov::Tensor, ov::Tensor> OpenposeDetector::inference(const ov::Tensor& input) {
    std::cout << "Running inference" << std::endl;
    ov::InferRequest req = body_model.create_infer_request();
    req.set_input_tensor(input);
    req.infer();

    auto res1 = req.get_output_tensor(0);
    auto res2 = req.get_output_tensor(1);

    return {res1, res2};
}

void OpenposeDetector::forward(const ov::Tensor& ori_img,
                               std::vector<std::vector<float>>& subset,
                               std::vector<std::vector<float>>& candidate) {
    auto img_shape = ori_img.get_shape();  // NHWC
    // assume Batch should be always 1
    auto ori_img_H = img_shape[1];
    auto ori_img_W = img_shape[2];

    std::vector<float> scale_search = {0.5};
    int boxsize = 368;
    int stride = 8;
    int pad_val = 128;
    float thre1 = 0.1f;
    float thre2 = 0.05f;

    std::vector<float> multiplier;
    for (float scale : scale_search) {
        multiplier.push_back(scale * boxsize / ori_img_H);
    }

    // Initialize the heatmap and PAF averages
    ov::Tensor heatmap_avg = init_tensor_with_zeros({1, ori_img_H, ori_img_W, 19}, ov::element::f32);
    ov::Tensor paf_avg = init_tensor_with_zeros({1, ori_img_H, ori_img_W, 38}, ov::element::f32);
    // Print the shape of the initialized tensors
    std::cout << "Heatmap Average Tensor Shape: " << heatmap_avg.get_shape() << std::endl;
    std::cout << "PAF Average Tensor Shape: " << paf_avg.get_shape() << std::endl;

    for (size_t m = 0; m < multiplier.size(); ++m) {
        float scale = multiplier[m];
        ov::Tensor image_to_test = smart_resize_k(ori_img, scale, scale);
        auto [image_to_test_padded, pad] = pad_right_down_corner(image_to_test, stride, pad_val);
        std::cout << "image_to_test_padded.shape: " << image_to_test_padded.get_shape() << std::endl;  // NHWC
        // NHWC -> NCHW
        ov::Tensor im(ov::element::u8,
                      {1,
                       image_to_test_padded.get_shape()[3],
                       image_to_test_padded.get_shape()[1],
                       image_to_test_padded.get_shape()[2]});
        reshape_tensor<uint8_t>(image_to_test_padded, im, {0, 3, 1, 2});
        std::cout << "im.shape: " << im.get_shape() << std::endl;
        // normalize to float32
        auto input = normalize_rgb_tensor(im);

        // Model inference code
        auto [Mconv7_stage6_L1, Mconv7_stage6_L2] = inference(input);

        std::cout << "Mconv7_stage6_L1.shape: " << Mconv7_stage6_L1.get_shape() << std::endl;
        std::cout << "Mconv7_stage6_L2.shape: " << Mconv7_stage6_L2.get_shape() << std::endl;

        // heatmap NCHW -> NHWC
        ov::Tensor heatmap(
            ov::element::f32,
            {1, Mconv7_stage6_L2.get_shape()[2], Mconv7_stage6_L2.get_shape()[3], Mconv7_stage6_L2.get_shape()[1]});
        reshape_tensor<float>(Mconv7_stage6_L2, heatmap, {0, 2, 3, 1});
        std::cout << "heatmap.shape: " << heatmap.get_shape() << std::endl;
        std::cout << "heatmap.element: " << heatmap.get_element_type() << std::endl;

        // Resize
        heatmap = smart_resize_k(heatmap, static_cast<float>(stride), static_cast<float>(stride));
        std::cout << "heatmap.shape: " << heatmap.get_shape() << std::endl;
        std::cout << "heatmap.element: " << heatmap.get_element_type() << std::endl;

        // Crop padding
        heatmap = crop_right_down_corner(heatmap, pad);
        std::cout << "cropped heatmap.shape: " << heatmap.get_shape() << std::endl;
        std::cout << "cropped heatmap.element: " << heatmap.get_element_type() << std::endl;
        // Resize
        heatmap = smart_resize(heatmap, ori_img_H, ori_img_W);
        std::cout << "heatmap.shape: " << heatmap.get_shape() << std::endl;

        // PAF NCHW -> NHWC
        ov::Tensor paf(
            ov::element::f32,
            {1, Mconv7_stage6_L1.get_shape()[2], Mconv7_stage6_L1.get_shape()[3], Mconv7_stage6_L1.get_shape()[1]});
        reshape_tensor<float>(Mconv7_stage6_L1, paf, {0, 2, 3, 1});
        std::cout << "paf.shape: " << paf.get_shape() << std::endl;
        // Resize
        paf = smart_resize_k(paf, static_cast<float>(stride), static_cast<float>(stride));
        std::cout << "paf.shape: " << paf.get_shape() << std::endl;
        std::cout << "paf.element: " << paf.get_element_type() << std::endl;

        // Crop padding
        paf = crop_right_down_corner(paf, pad);
        std::cout << "paf.shape: " << paf.get_shape() << std::endl;
        std::cout << "paf.element: " << paf.get_element_type() << std::endl;

        // Resize
        paf = smart_resize(paf, ori_img_H, ori_img_W);
        std::cout << "cropped paf.shape: " << paf.get_shape() << std::endl;

        // Accumulate results
        auto heatmap_avg_data = heatmap_avg.data<float>();
        auto heatmap_data = heatmap.data<float>();
        for (size_t i = 0; i < heatmap_avg.get_size(); ++i) {
            heatmap_avg_data[i] += heatmap_data[i] / multiplier.size();
        }

        auto paf_avg_data = paf_avg.data<float>();
        auto paf_data = paf.data<float>();
        for (size_t i = 0; i < paf_avg.get_size(); ++i) {
            paf_avg_data[i] += paf_data[i] / multiplier.size();
        }
    }

    // find the keypoints from heatmap
    std::vector<std::vector<std::tuple<int, int, float, int>>> all_peaks;
    find_heatmap_peaks(heatmap_avg, thre1, all_peaks);

    // iterate and print peaks
    for (auto& peak : all_peaks) {
        std::cout << "Peak: " << std::get<0>(peak[0]) << " " << std::get<1>(peak[0]) << " " << std::get<2>(peak[0])
                  << std::endl;
        std::cout << "Counter: " << std::get<3>(peak[0]) << std::endl;
    }

    std::vector<std::vector<std::tuple<int, int, float, int, int>>> connection_all;
    std::vector<int> special_k;
    calculate_connections(paf_avg, all_peaks, ori_img, thre2, connection_all, special_k);

    for (auto& connection : connection_all) {
        std::cout << "Connection: " << std::get<0>(connection[0]) << " " << std::get<1>(connection[0]) << " "
                  << std::get<2>(connection[0]) << " " << std::get<3>(connection[0]) << std::endl;
    }

    process_connections(all_peaks, connection_all, special_k, subset, candidate);

    // print candidate
    for (auto& cand : candidate) {
        std::cout << "Candidate: " << cand[0] << " " << cand[1] << " " << cand[2] << " " << cand[3] << std::endl;
    }

    for (auto& sub : subset) {
        std::cout << "Subset: ";
        for (auto& s : sub) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
    }
}

// find the peaks from heatmap, returns a vector of tuple
// (x, y, score, id)
void OpenposeDetector::find_heatmap_peaks(const ov::Tensor& heatmap_avg /* f32 */,
                                          float thre1,
                                          std::vector<std::vector<std::tuple<int, int, float, int>>>& all_peaks) {
    auto heatmap_shape = heatmap_avg.get_shape();
    auto H = heatmap_shape[1];
    auto W = heatmap_shape[2];
    auto C = heatmap_shape[3];

    all_peaks.clear();
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
}

void OpenposeDetector::calculate_connections(
    const ov::Tensor& paf_avg,
    const std::vector<std::vector<std::tuple<int, int, float, int>>>& all_peaks,
    const ov::Tensor& oriImg,
    const float thre2,
    std::vector<std::vector<std::tuple<int, int, float, int, int>>>& connection_all,
    std::vector<int>& special_k) {
    const int mid_num = 10;

    connection_all.clear();
    special_k.clear();

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
}

void OpenposeDetector::process_connections(
    const std::vector<std::vector<std::tuple<int, int, float, int>>>& all_peaks,
    const std::vector<std::vector<std::tuple<int, int, float, int, int>>>& connection_all,
    const std::vector<int>& special_k,
    std::vector<std::vector<float>>& subset,
    std::vector<std::vector<float>>& candidate) {
    // Initialize subset and candidate
    subset.clear();
    candidate.clear();

    // Flatten all_peaks into candidate
    for (const auto& peaks : all_peaks) {
        for (const auto& peak : peaks) {
            candidate.push_back({static_cast<float>(std::get<0>(peak)),
                                 static_cast<float>(std::get<1>(peak)),
                                 std::get<2>(peak),
                                 static_cast<float>(std::get<3>(peak))});
        }
    }

    for (size_t k = 0; k < mapIdx.size(); ++k) {
        if (std::find(special_k.begin(), special_k.end(), k) == special_k.end()) {
            const auto& parts = connection_all[k];
            int indexA = limbSeq[k][0] - 1;
            int indexB = limbSeq[k][1] - 1;

            for (size_t i = 0; i < connection_all[k].size(); ++i) {
                int found = 0;
                int partA = std::get<0>(parts[i]);
                int partB = std::get<1>(parts[i]);
                float part_score = std::get<2>(parts[i]);
                float partA_score = candidate[partA][2];
                float partB_score = candidate[partB][2];

                std::vector<int> subset_idx = {-1, -1};
                for (size_t j = 0; j < subset.size(); ++j) {
                    if (subset[j][indexA] == partA || subset[j][indexB] == partB) {
                        subset_idx[found] = j;
                        found += 1;
                    }
                }

                if (found == 1) {
                    int j = subset_idx[0];
                    if (subset[j][indexB] != partB) {
                        subset[j][indexB] = partB;
                        subset[j][18] += 1;
                        subset[j][19] += part_score + partB_score;
                    }
                } else if (found == 2) {
                    int j1 = subset_idx[0];
                    int j2 = subset_idx[1];
                    std::vector<int> membership(subset[j1].begin(), subset[j1].begin() + 18);
                    std::transform(membership.begin(),
                                   membership.end(),
                                   subset[j2].begin(),
                                   membership.begin(),
                                   std::plus<int>());
                    if (std::none_of(membership.begin(), membership.end(), [](int v) {
                            return v > 1;
                        })) {
                        std::transform(subset[j2].begin(),
                                       subset[j2].begin() + 18,
                                       subset[j1].begin(),
                                       subset[j1].begin(),
                                       std::plus<float>());
                        subset[j1][19] += subset[j2][19];
                        subset[j1][19] += part_score;
                        subset.erase(subset.begin() + j2);
                    } else {
                        subset[j1][indexB] = partB;
                        subset[j1][18] += 1;
                        subset[j1][19] += part_score + partB_score;
                    }
                } else if (!found && k < 17) {
                    std::vector<float> row(20, -1);
                    row[indexA] = partA;
                    row[indexB] = partB;
                    row[18] = 2;
                    row[19] = partA_score + partB_score + part_score;
                    subset.push_back(row);
                }
            }
        }
    }

    // Filter out invalid subsets
    auto it = std::remove_if(subset.begin(), subset.end(), [](const std::vector<float>& row) {
        return row[18] < 4 || (row[19] / row[18]) < 0.4;
    });
    subset.erase(it, subset.end());

    // Swap row[18] and row[19], to keep with python side
    for (auto& row : subset) {
        std::swap(row[18], row[19]);
    }
}