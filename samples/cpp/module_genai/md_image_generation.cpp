#include <iostream>
#include <openvino/genai/module_genai/pipeline.hpp>

#include <stdexcept>

#include "utils/vision_utils.hpp"
#include "yaml-cpp/yaml.h"
#include "utils/utils.hpp"

inline ov::AnyMap parse_inputs_from_yaml_cfg_for_image_generation(const std::filesystem::path& cfg_yaml_path,
                                                                  const std::string& prompt = std::string{},
                                                                  const std::string& width = std::string{},
                                                                  const std::string& height = std::string{},
                                                                  const std::string& num_inference_steps = std::string{},
                                                                  const std::string& guidance_scale = std::string{},
                                                                  const std::string& max_sequence_length = std::string{}) {
    ov::AnyMap inputs;
    YAML::Node input_params = utils::find_param_module_in_yaml(cfg_yaml_path);

    // Loop input_params to find "prompt", "image", "video"
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

        if (param_type == "Int" && utils::contains_key(param_name, {"width", "w"})) {
            if (width.empty()) {
                throw std::runtime_error("Width is empty.");
            }
            inputs[param_name] = std::stoi(width);
            continue;
        }

        if (param_type == "Int" && utils::contains_key(param_name, {"height", "h"})) {
            if (height.empty()) {
                throw std::runtime_error("Height is empty.");
            }
            inputs[param_name] = std::stoi(height);
            continue;
        }

        if (param_type == "Int" && utils::contains_key(param_name, {"num_inference_steps", "steps"})) {
            if (num_inference_steps.empty()) {
                throw std::runtime_error("Number of inference steps is empty.");
            }
            inputs[param_name] = std::stoi(num_inference_steps);
            continue;
        }

        if (param_type == "Float" && utils::contains_key(param_name, {"guidance_scale", "guidance"})) {
            if (guidance_scale.empty()) {
                throw std::runtime_error("Guidance scale is empty.");
            }
            inputs[param_name] = std::stof(guidance_scale);
            continue;
        }

        if (param_type == "Int" && utils::contains_key(param_name, {"max_sequence_length", "max_seq_len"})) {
            if (max_sequence_length.empty()) {
                throw std::runtime_error("Max sequence length is empty.");
            }
            inputs[param_name] = std::stoi(max_sequence_length);
            continue;
        }
    }
    return inputs;
}

int main(int argc, char* argv[]) {
    try {
        if (argc <= 1) {
            throw std::runtime_error(std::string{"Usage: "} + argv[0] + "\n"
                                     "  -cfg config.yaml \n"
                                     "  -prompt: input prompt\n"
                                     "  --height: default 512\n"
                                     "  --width: default 512\n"
                                     "  --num_inference_steps: default 9\n"
                                     "  --guidance_scale: default 2.0\n"
                                     "  --max_sequence_length: default 512\n");
        }

        std::filesystem::path config_path = utils::get_input_arg(argc, argv, "-cfg");
        std::string prompt = utils::get_input_arg(argc, argv, "-prompt");
        std::string width = utils::get_input_arg(argc, argv, "--width", "512");
        std::string height = utils::get_input_arg(argc, argv, "--height", "512");
        std::string num_inference_steps = utils::get_input_arg(argc, argv, "--num_inference_steps", "9");
        std::string guidance_scale = utils::get_input_arg(argc, argv, "--guidance_scale", "2.0");
        std::string max_sequence_length = utils::get_input_arg(argc, argv, "--max_sequence_length", "512");

        ov::AnyMap inputs = parse_inputs_from_yaml_cfg_for_image_generation(config_path, prompt, width, height, num_inference_steps, guidance_scale, max_sequence_length);

        for (const auto& [key, value] : inputs) {
            std::cout << "[Input] " << key << ": " << value.as<std::string>() << std::endl;
        }

        ov::genai::module::ModulePipeline pipe(config_path);

        pipe.generate(inputs);

        ov::Tensor generated_image = pipe.get_output("generated_image").as<ov::Tensor>();

        image_utils::save_image_bmp("generated_image.bmp", generated_image);
        std::cout << "Generated image saved to generated_image.bmp" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}