// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <openvino/genai/module_genai/pipeline.hpp>

#include <stdexcept>

#include "utils/vision_utils.hpp"
#include "yaml-cpp/yaml.h"
#include "utils/utils.hpp"

inline ov::AnyMap parse_inputs_from_yaml_cfg_for_vlm(const std::filesystem::path& cfg_yaml_path,
                                                     const std::string& prompt = std::string{},
                                                     const std::string& image_path = std::string{},
                                                     const std::string& video_path = std::string{},
                                                     const std::string& audio_path = std::string{}) {
    ov::AnyMap inputs;
    YAML::Node input_params = utils::find_param_module_in_yaml(cfg_yaml_path);

    // Loop input_params to find "prompt", "image", "video", "audio"
    for (const auto& entry : input_params) {
        if (!entry["name"] || !entry["type"]) {
            continue;
        }

        const std::string param_name = entry["name"].as<std::string>();
        const std::string param_type = entry["type"].as<std::string>();

        if (param_type == "String" && utils::contains_key(param_name, {"prompt"})) {
            if (prompt.empty()) {
                throw std::runtime_error("Prompt string is empty.");
            }
            inputs[param_name] = prompt;
            continue;
        }

        if (param_type == "OVTensor" && utils::contains_key(param_name, {"img", "image"})) {
            if (image_path.empty()) {
                throw std::runtime_error("Image path is empty.");
            }
            inputs[param_name] = image_utils::load_image(image_path);
            continue;
        }

        if (param_type == "OVTensor" && utils::contains_key(param_name, {"video"})) {
            if (video_path.empty()) {
                throw std::runtime_error("Video path is empty.");
            }
            inputs[param_name] = image_utils::load_video(video_path);
            continue;
        }

        if (param_type == "String" && utils::contains_key(param_name, {"audio"})) {
            if (audio_path.empty()) {
                throw std::runtime_error("Audio path is empty.");
            }
            inputs[param_name] = audio_utils::read_wav(audio_path);
            continue;
        }
    }
    return inputs;
}

int main(int argc, char* argv[]) {
    try {
        if (argc <= 1) {
            throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                     "\n"
                                     "  -cfg config.yaml \n"
                                     "  -prompt: input prompt\n"
                                     "  -img: [Optional] image path\n"
                                     "  -video: [Optional] video path\n"
                                     "  -audio: [Optional] audio path\n");
        }

        std::filesystem::path config_path = utils::get_input_arg(argc, argv, "-cfg", std::string{});
        std::string prompt = utils::get_input_arg(argc, argv, "-prompt", std::string{});
        std::string img_path = utils::get_input_arg(argc, argv, "-img", std::string{});
        std::string video_path = utils::get_input_arg(argc, argv, "-video", std::string{});
        std::string audio_path = utils::get_input_arg(argc, argv, "-audio", std::string{});

        ov::AnyMap inputs = parse_inputs_from_yaml_cfg_for_vlm(config_path, prompt, img_path, video_path, audio_path);

        for (const auto& [key, value] : inputs) {
            std::cout << "[Input] " << key << ": ";
            if (value.is<std::string>()) {
                std::cout << value.as<std::string>();
            } else if (value.is<ov::Tensor>()) {
                const auto& tensor = value.as<ov::Tensor>();
                std::cout << "Tensor (rank=" << tensor.get_shape().size() << ")";
            } else {
                std::cout << "<non-string input>";
            }
            std::cout << std::endl;
        }

        ov::genai::module::ModulePipeline pipe(config_path);

        pipe.generate(inputs);

        std::cout << "Generation Result: " << pipe.get_output("generated_text").as<std::string>() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}