// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include <cxxopts.hpp>
#include "imwrite.hpp"

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_image_generation", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value("The Sky is blue because"))
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("o,output_dir", "Path to save output image", cxxopts::value<std::string>()->default_value(""))
    ("wh,width", "The width of the resulting image", cxxopts::value<size_t>()->default_value(std::to_string(512)))
    ("ht,height", "The height of the resulting image", cxxopts::value<size_t>()->default_value(std::to_string(512)))
    ("is,num_inference_steps", "The number of inference steps used to denoise initial noised latent to final image", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("ni,num_images_per_prompt", "The number of images to generate per generate() call", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("h,help", "Print usage");

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

    std::string prompt = result["prompt"].as<std::string>();
    const std::string models_path = result["model"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    const std::string output_dir = result["output_dir"].as<std::string>();

    ov::genai::Text2ImagePipeline pipe(models_path, device);
    ov::genai::ImageGenerationConfig config = pipe.get_generation_config();
    config.width = result["width"].as<size_t>();
    config.height = result["height"].as<size_t>();
    config.num_inference_steps = result["num_inference_steps"].as<size_t>();
    config.num_images_per_prompt = result["num_images_per_prompt"].as<size_t>();
    pipe.set_generation_config(config);
    
    for (size_t i = 0; i < num_warmup; i++) {
        pipe.generate(prompt);
    }

    std::vector<float> generate_durations;
    std::vector<float> total_inference_durations;
    float load_time;
    for (size_t i = 0; i < num_iter; i++) {
        ov::Tensor image = pipe.generate(prompt);
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        generate_durations.emplace_back(metrics.get_generate_duration());
        total_inference_durations.emplace_back(metrics.get_inference_total_duration());
        std::string image_name = output_dir + "/image_" + std::to_string(i) + ".bmp";
        imwrite(image_name, image, true);
        load_time = metrics.get_load_time();
    }

    float generate_mean = std::accumulate(generate_durations.begin(),
                                          generate_durations.end(),
                                          0.0f,
                                          [](const float& acc, const float& duration) -> float {
                                              return acc + duration;
                                          });
    if (!generate_durations.empty()) {
        generate_mean /= generate_durations.size();
    }

    float inference_mean = std::accumulate(total_inference_durations.begin(),
                                           total_inference_durations.end(),
                                           0.0f,
                                           [](const float& acc, const float& duration) -> float {
                                               return acc + duration;
                                           });
    if (!total_inference_durations.empty()) {
        inference_mean /= total_inference_durations.size();
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Load time: " << load_time << " ms" << std::endl;
    std::cout << "One generate avg time: " << generate_mean << " ms" << std::endl;
    std::cout << "Total inference for one generate avg time: " << inference_mean << " ms" << std::endl;

    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
