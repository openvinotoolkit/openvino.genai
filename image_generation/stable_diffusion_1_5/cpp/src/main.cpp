// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cxxopts.hpp"
#include "stable_diffusion.hpp"

int32_t main(int32_t argc, char* argv[]) {
    cxxopts::Options options("OV_SD_CPP", "SD implementation in C++ using OpenVINO\n");

    options.add_options()(
        "p,posPrompt",
        "Initial positive prompt for SD ",
        cxxopts::value<std::string>()->default_value(
        "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting"))(
        "n,negPrompt",
        "Defaut is empty with space",
        cxxopts::value<std::string>()->default_value(" "))(
        "d,device",
        "AUTO, CPU, or GPU",
        cxxopts::value<std::string>()->default_value("CPU"))(
        "step",
        "Number of diffusion steps",
        cxxopts::value<size_t>()->default_value("20"))(
        "s,seed",
        "Number of random seed to generate latent for one image output",
        cxxopts::value<size_t>()->default_value("42"))(
        "num",
        "Number of image output",
        cxxopts::value<size_t>()->default_value("1"))(
        "height",
        "height",
        cxxopts::value<size_t>()->default_value("512"))(
        "width",
        "width",
        cxxopts::value<size_t>()->default_value("512"))(
        "log",
        "generate logging into log.txt for debug",
        cxxopts::value<bool>()->default_value("false"))(
        "c,useCache",
        "use model caching",
        cxxopts::value<bool>()->default_value("false"))(
        "r,readNPLatent",
        "read numpy generated latents from file",
        cxxopts::value<bool>()->default_value("false"))(
        "m,modelPath",
        "Specify path of SD model IR",
        cxxopts::value<std::string>()->default_value("../models/dreamlike-anime-1.0"))(
        "t,type",
        "Specify the type of SD model IR(FP16_static or FP16_dyn)",
        cxxopts::value<std::string>()->default_value("FP16_static"))(
        "l,loraPath",
        "Specify path of lora file. (*.safetensors).",
        cxxopts::value<std::string>()->default_value(""))(
        "a,alpha",
        "alpha for lora",
        cxxopts::value<float>()->default_value("0.75"))("h,help", "Print usage");
    cxxopts::ParseResult result;

    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const std::string positive_prompt = result["posPrompt"].as<std::string>();
    const std::string negative_prompt = result["negPrompt"].as<std::string>();
    const std::string device = result["device"].as<std::string>();
    uint32_t steps = result["step"].as<size_t>();
    uint32_t seed = result["seed"].as<size_t>();
    uint32_t num_images = result["num"].as<size_t>();
    uint32_t height = result["height"].as<size_t>();
    uint32_t width = result["width"].as<size_t>();
    const bool use_logger = result["log"].as<bool>();
    const bool use_cache = result["useCache"].as<bool>();
    const bool read_NP_latent = result["readNPLatent"].as<bool>();
    const std::string model_path = result["modelPath"].as<std::string>();
    const std::string model_type = result["type"].as<std::string>();
    const std::string lora_path = result["loraPath"].as<std::string>();
    float alpha = result["alpha"].as<float>();

    if (!std::filesystem::exists(model_path + "/" + model_type)) {
        std::cout << "Model path: " << model_path + "/" + model_type << " does not exist\n";
        return EXIT_FAILURE;
    }

    std::vector<std::string> output_images_paths;
    std::vector<uint32_t> seed_vec;
    std::string folder_name = "images";
    try {
        std::filesystem::create_directory(folder_name);
    } catch (const std::exception& e) {
        std::cerr << "fail to create dir" << e.what() << std::endl;
    }

    if (num_images == 1) {
        seed_vec.push_back(seed);
        std::string output_bmp_path = std::string{"./images/seed_"} + std::to_string(seed) + std::string{".bmp"};
        output_images_paths.push_back(output_bmp_path);
    } else {
        for (uint32_t n = 0; n < num_images; n++) {
            seed_vec.push_back(n);
            std::string output_bmp_path = std::string{"./images/seed_"} + std::to_string(n) + std::string{".bmp"};
            output_images_paths.push_back(output_bmp_path);
        }
    }

    auto start_total = std::chrono::steady_clock::now();
    stable_diffusion(positive_prompt,
                     output_images_paths,
                     device,
                     steps,
                     seed_vec,
                     num_images,
                     height,
                     width,
                     negative_prompt,
                     use_logger,
                     use_cache,
                     model_path,
                     model_type,
                     lora_path,
                     alpha,
                     read_NP_latent);
    auto end_total = std::chrono::steady_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::duration<float>>(end_total - start_total);

    return EXIT_SUCCESS;
}
