// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include "utils.hpp"
#include <openvino/genai/module_genai/pipeline.hpp>
#include <filesystem>
#include <chrono>
#include "ut_modules_base.hpp"

static std::string get_config_ymal_path(int argc, char *argv[]) {
    if (argc > 2) {
        return std::string(argv[2]);
    }
    return "config.yaml";
}

static bool print_subword(std::string &&subword)
{
    return !(std::cout << subword << std::flush);
}

std::vector<ov::Any> test_module_pipeline(const std::filesystem::path& config_path,
                          ov::AnyMap inputs,
                          const std::vector<std::string>& output_names) {
    ov::genai::module::ModulePipeline pipe(config_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    pipe.generate(inputs);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "  Generate time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

    std::vector<ov::Any> outputs;
    for (const auto& name : output_names) {
        outputs.push_back(pipe.get_output(name));
    }
    return outputs;
}

void test_qwen2_5_vl_module_pipeline(int argc, char *argv[])
{
    std::string config_yaml = "ut_pipelines/Qwen2.5-VL-3B-Instruct/config.yaml";
    if (argc > 2) {
        config_yaml = std::string(argv[2]);
    }

    std::cout << "== Init ModulePipeline: " << config_yaml << std::endl;

    ov::AnyMap inputs;
    inputs["prompts_data"] = std::vector<std::string>{"Please describle this image"};
    inputs["img1"] = utils::load_image("ut_test_data/cat_120_100.png");

    auto outputs = test_module_pipeline(std::filesystem::path(config_yaml), inputs, {"generated_text"});
    auto generated_text = outputs[0].as<std::string>();
    std::cout << "  Generated Text: " << generated_text << std::endl;

    // bool contains_white_cat = generated_text.find("white cat") != std::string::npos;
    // if (!contains_white_cat) {
    //     throw std::runtime_error("  llm inference module does not work as expected");
    // }
}

int test_genai_module_ut_pipelines(int argc, char *argv[])
{
    // ov::genai::module::PrintAllModulesConfig();

    test_qwen2_5_vl_module_pipeline(argc, argv);
    return EXIT_SUCCESS;
}
