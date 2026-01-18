// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <thread>
#include <chrono>
#include <filesystem>
#include <openvino/genai/module_genai/pipeline.hpp>

#include "utils/load_image.hpp"
#include "utils/utils.hpp"
#include "utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "../utils/model_yaml.hpp"

using namespace ov::genai::module;

// Test for ModulePipeline generate_async and generate functions.
// The test verifies that both functions produce the same output given the same input.

TEST(PipelineTestAccuracy, GenerateVsGenerateAsync) {
    std::string device = TEST_MODEL::get_device();
    std::string qwen2_5_vl_model_path = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
    std::string test_img_cat = TEST_DATA::img_cat_120_100();

    std::string yaml_context = TEST_MODEL::get_qwen2_5_vl_config_yaml(qwen2_5_vl_model_path, device);

    ov::AnyMap inputs;
    inputs["prompts_data"] = std::vector<std::string>{"Please describe this image"};
    inputs["img1"] = utils::load_image(test_img_cat);

    ov::genai::module::ModulePipeline pipe(yaml_context);

    auto t1 = std::chrono::high_resolution_clock::now();
    pipe.generate(inputs);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::string output_text_sync = pipe.get_output("generated_text").as<std::string>();

    auto t3 = std::chrono::high_resolution_clock::now();
    pipe.generate_async(inputs);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::string output_text_async = pipe.get_output("generated_text").as<std::string>();

    std::cout << "  Synchronous Generate time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    std::cout << "  Asynchronous Generate time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << std::endl;
    std::cout << "  Generated Text: " << output_text_sync << std::endl;

    EXPECT_EQ(output_text_sync, output_text_async);
}
