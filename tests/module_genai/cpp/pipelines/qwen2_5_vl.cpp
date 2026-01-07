// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <openvino/genai/module_genai/pipeline.hpp>

#include "utils/load_image.hpp"
#include "utils/utils.hpp"
#if 0
TEST(Qwen2_5_VL_Module_Pipeline, generate_text_from_image) {
    std::string config_yaml = "ut_pipelines/Qwen2.5-VL-3B-Instruct/config.yaml";
    std::string test_image = "cat_120_100.png";

    ov::AnyMap inputs;
    inputs["prompts_data"] = std::vector<std::string>{"Please describle this image"};
    inputs["img1"] = utils::load_image(test_image);

    ov::genai::module::ModulePipeline pipe(config_yaml);

    auto t1 = std::chrono::high_resolution_clock::now();
    pipe.generate(inputs);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "  Generate time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"
              << std::endl;

    auto output = pipe.get_output("generated_text");

    std::cout << "  Generated Text: " << output.as<std::string>() << std::endl;
}
#endif