// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <string>

#include "core/core.hpp"
#include "cxxopts.hpp"
#include "imwrite.hpp"
#include "openvino/runtime/core.hpp"

ov::Tensor postprocess_image(ov::Tensor decoded_image) {
    ov::Tensor generated_image(ov::element::u8, decoded_image.get_shape());
    // convert to u8 image
    const float* decoded_data = decoded_image.data<const float>();
    std::uint8_t* generated_data = generated_image.data<std::uint8_t>();
    for (size_t i = 0; i < decoded_image.get_size(); ++i) {
        generated_data[i] = static_cast<std::uint8_t>(std::clamp(decoded_data[i] * 0.5f + 0.5f, 0.0f, 1.0f) * 255);
    }

    return generated_image;
}

int32_t main(int32_t argc, char* argv[]) try {
    cxxopts::Options options("stable_diffusion", "Stable Diffusion implementation in C++ using OpenVINO\n");

    options.add_options()(
        "p,posPrompt",
        "Initial positive prompt for SD ",
        cxxopts::value<std::string>()->default_value("Dancing Darth Vader, best quality, extremely detailed"))(
        "n,negPrompt",
        "Defaut is empty with space",
        cxxopts::value<std::string>()->default_value(" "))(
        "d,device",
        "AUTO, CPU, or GPU.\nDoesn't apply to Tokenizer model, OpenVINO Tokenizers can be inferred on a CPU device "
        "only",
        cxxopts::value<std::string>()->default_value(
            "CPU"))("step", "Number of diffusion steps", cxxopts::value<size_t>()->default_value("20"))(
        "s,seed",
        "Number of random seed to generate latent for one image output",
        cxxopts::value<size_t>()->default_value(
            "42"))("num", "Number of image output", cxxopts::value<size_t>()->default_value("1"))(
        "height",
        "Destination image height",
        cxxopts::value<size_t>()->default_value(
            "512"))("width", "Destination image width", cxxopts::value<size_t>()->default_value("512"))(
        "x,controlnet_input",
        "Read controlnet input from file",
        cxxopts::value<std::string>()->default_value(""))(
        "l,latent",
        "Read numpy generated latents from file",
        cxxopts::value<std::string>()->default_value(
            ""))("c,useCache", "Use model caching", cxxopts::value<bool>()->default_value("false"))(
        "m,modelPath",
        "Specify path of SD model IRs",
        cxxopts::value<std::string>()->default_value("./models"))("i,inputImage",
                                                                  "Specify path of Input image",
                                                                  cxxopts::value<std::string>()->default_value(""))(
        "o,outputImage",
        "Specify path of output image",
        cxxopts::value<std::string>()->default_value(""))("h,help", "Print usage");
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

    std::string positive_prompt = result["posPrompt"].as<std::string>();
    std::string negative_prompt = result["negPrompt"].as<std::string>();
    const std::string device = result["device"].as<std::string>();
    const uint32_t num_inference_steps = result["step"].as<size_t>();
    const uint32_t user_seed = result["seed"].as<size_t>();
    const uint32_t num_images = result["num"].as<size_t>();
    const uint32_t height = result["height"].as<size_t>();
    const uint32_t width = result["width"].as<size_t>();
    const bool use_cache = result["useCache"].as<bool>();
    const std::string model_base_path = result["modelPath"].as<std::string>();
    const std::string input_image_path = result["inputImage"].as<std::string>();
    const std::string output_image_path = result["outputImage"].as<std::string>();
    const std::string np_latent = result["latent"].as<std::string>();
    const std::string controlnet_input = result["controlnet_input"].as<std::string>();
    

    const std::string folder_name = "images";
    try {
        std::filesystem::create_directory(folder_name);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create dir" << e.what() << std::endl;
    }

    std::cout << "OpenVINO version: " << ov::get_openvino_version() << std::endl;

    const std::string model_path = model_base_path;
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model path " << model_path << " don't exist"
                  << "\n";
        std::cerr << "Refer to README.md to know how to export OpenVINO model with particular data type." << std::endl;
        return EXIT_FAILURE;
    }

    // Stable Diffusion Controlnet pipeline
    StableDiffusionControlnetPipeline pipeline(model_path, device);
    for (uint32_t n = 0; n < num_images; n++) {
        std::uint32_t seed = num_images == 1 ? user_seed : user_seed + n;
        StableDiffusionControlnetPipelineParam param = {positive_prompt,
                                                        negative_prompt,
                                                        input_image_path,
                                                        num_inference_steps,
                                                        seed,
                                                        np_latent,
                                                        controlnet_input};
        auto decoded_image = pipeline.Run(param);
        auto image = postprocess_image(decoded_image);
        if (output_image_path != "") {
            imwrite(output_image_path, image, true);
        } else {
            imwrite(std::string("./images/seed_") + std::to_string(seed) + ".bmp", image, true);
        }
    }
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
