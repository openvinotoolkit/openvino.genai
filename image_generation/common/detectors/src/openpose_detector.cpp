// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openpose_detector.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>

#include "imwrite.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"

// Helper function to initialize a tensor with zeros
ov::Tensor initializeTensorWithZeros(const ov::Shape& shape, ov::element::Type type) {
    ov::Tensor tensor(type, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    return tensor;
}

void OpenposeDetector::load(const std::string& model_path) {
    std::cout << "Loading model from: " << model_path << std::endl;

    ov::Core core;
    std::string device = "CPU";
    auto model = core.read_model(model_path + "/openpose.xml");
    // TODO: W / H dimension should be dynamic, we reshape it before comlile
    body_model = core.compile_model(model, device);
}

ov::Tensor OpenposeDetector::preprocess(ov::Tensor input /* NHWC */) {
    std::cout << "Preprocessing data" << std::endl;

    return input;
}

std::pair<ov::Tensor, ov::Tensor> OpenposeDetector::inference(ov::Tensor input) {
    std::cout << "Running inference" << std::endl;
    ov::InferRequest req = body_model.create_infer_request();
    req.set_input_tensor(input);
    req.infer();

    auto res1 = req.get_output_tensor(0);
    auto res2 = req.get_output_tensor(1);

    return {res1, res2};
}

void OpenposeDetector::forward(const std::string& im_txt, unsigned long w, unsigned long h, unsigned long c) {
    // Set up initial parameters
    std::vector<float> scale_search = {0.5};
    int boxsize = 368;
    int stride = 8;
    int pad_val = 128;
    float thre1 = 0.1f;
    float thre2 = 0.05f;

    // functional tests
    // Load Image
    std::cout << "Load " << im_txt << std::endl;
    std::vector<std::uint8_t> im_array = read_bgr_from_txt(im_txt);

    ov::Shape img_shape = {1, h, w, c};  // NHWC
    ov::Tensor img_tensor(ov::element::u8, img_shape);

    // validate the read function
    std::uint8_t* tensor_data = img_tensor.data<std::uint8_t>();
    std::copy(im_array.begin(), im_array.end(), tensor_data);
    std::cerr << "Tensor shape: " << img_tensor.get_shape() << std::endl;
    imwrite(std::string("im.bmp"), img_tensor, false);

    // validate the resize function
    ov::Tensor small_img_tensor = smart_resize_k(img_tensor, 0.5, 0.5);
    imwrite(std::string("im.half.bmp"), small_img_tensor, false);

    ov::Tensor big_img_tensor = smart_resize_k(img_tensor, 2, 2);
    imwrite(std::string("im.double.bmp"), big_img_tensor, false);

    ov::Tensor need_pad_img_tensor = smart_resize(img_tensor, 761, 505);
    auto [img_padded, pad] = pad_right_down_corner(need_pad_img_tensor, stride, pad_val);
    imwrite(std::string("im.paded.bmp"), img_padded, false);

    auto img_cropped = crop_right_down_corner(img_padded, pad);
    imwrite(std::string("im.cropped.bmp"), img_cropped, false);

    // ===========================================

    std::vector<std::uint8_t> input_array = read_bgr_from_txt(im_txt);
    ov::Tensor ori_img(ov::element::u8, {1, h, w, c});

    // validate the read function
    std::uint8_t* input_data = ori_img.data<std::uint8_t>();
    std::copy(input_array.begin(), input_array.end(), input_data);

    ov::Shape ori_img_shape = ori_img.get_shape();
    auto ori_img_H = ori_img_shape[1];
    auto ori_img_W = ori_img_shape[2];

    // Compute multipliers
    std::vector<float> multiplier;
    for (float scale : scale_search) {
        multiplier.push_back(scale * boxsize / ori_img_H);
    }

    // Initialize the heatmap and PAF averages
    ov::Tensor heatmap_avg = initializeTensorWithZeros({1, ori_img_H, ori_img_W, 19}, ov::element::f32);
    ov::Tensor paf_avg = initializeTensorWithZeros({1, ori_img_H, ori_img_W, 38}, ov::element::f32);
    // Print the shape of the initialized tensors
    std::cout << "Heatmap Average Tensor Shape: " << heatmap_avg.get_shape() << std::endl;
    std::cout << "PAF Average Tensor Shape: " << paf_avg.get_shape() << std::endl;

    for (size_t m = 0; m < multiplier.size(); ++m) {
        float scale = multiplier[m];
        ov::Tensor image_to_test = smart_resize_k(img_tensor, scale, scale);
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
    auto peaks = find_heatmap_peaks(heatmap_avg, thre1);
    // iterate and print peaks
    for (auto& peak : peaks) {
        std::cout << "Peak: " << std::get<0>(peak) << " " << std::get<1>(peak) << " " << std::get<2>(peak) << std::endl;
        std::cout << "Counter: " << std::get<3>(peak) << std::endl;
    }
}

void OpenposeDetector::postprocess() {
    std::cout << "Postprocessing results" << std::endl;
}
