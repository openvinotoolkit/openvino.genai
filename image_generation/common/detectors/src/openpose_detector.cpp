// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openpose_detector.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <openvino/core/shape.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>

#include "imwrite.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"

int OpenposeDetector::foo() {
    return 1;
}

void OpenposeDetector::load(const std::string& model_path) {
    std::cout << "Loading model from: " << model_path << std::endl;

    ov::Core core;
    std::string device = "CPU";
    auto model = core.read_model(model_path + "/openpose.xml");
    body_model = core.compile_model(model, device);
}

void OpenposeDetector::preprocess() {
    std::cout << "Preprocessing data" << std::endl;
}

void OpenposeDetector::inference(const std::string& im_txt) {
    std::cout << "Running inference" << std::endl;
}

void OpenposeDetector::load_bgr(const std::string& im_txt, unsigned long w, unsigned long h, unsigned long c) {
    std::cout << "Load " << im_txt << std::endl;
    std::vector<std::uint8_t> im_array = read_bgr_from_txt(im_txt);

    ov::Shape img_shape = {1, h, w, c};  // NHWC
    ov::Tensor img_tensor(ov::element::u8, img_shape);

    std::uint8_t* tensor_data = img_tensor.data<std::uint8_t>();
    std::copy(im_array.begin(), im_array.end(), tensor_data);
    std::cerr << "Tensor shape: " << img_tensor.get_shape() << std::endl;
    imwrite(std::string("im.bmp"), img_tensor, false);
}

void OpenposeDetector::postprocess() {
    std::cout << "Postprocessing results" << std::endl;
}
