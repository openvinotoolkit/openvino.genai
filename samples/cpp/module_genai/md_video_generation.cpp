#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <openvino/genai/module_genai/pipeline.hpp>

#include "utils/utils.hpp"
#include "yaml-cpp/yaml.h"

namespace {

inline ov::AnyMap parse_inputs_from_yaml_cfg_for_video_generation(const std::filesystem::path& cfg_yaml_path,
                                                                  const std::string& prompt,
                                                                  const std::string& negative_prompt,
                                                                  const std::string& width,
                                                                  const std::string& height,
                                                                  const std::string& num_frames,
                                                                  const std::string& num_inference_steps,
                                                                  const std::string& guidance_scale,
                                                                  const std::string& max_sequence_length,
                                                                  const std::string& batch_size,
                                                                  const std::string& num_videos_per_prompt,
                                                                  const std::string& seed) {
    ov::AnyMap inputs;
    YAML::Node input_params = utils::find_param_module_in_yaml(cfg_yaml_path);

    for (const auto& entry : input_params) {
        if (!entry["name"] || !entry["type"]) {
            continue;
        }

        const std::string param_name = entry["name"].as<std::string>();
        const std::string param_type = entry["type"].as<std::string>();

        if (param_type == "String" && param_name == "prompt") {
            if (prompt.empty()) {
                throw std::runtime_error("Prompt string is empty.");
            }
            inputs[param_name] = prompt;
            continue;
        }

        if (param_type == "String" && utils::contains_key(param_name, {"negative_prompt"})) {
            // negative_prompt is optional
            inputs[param_name] = negative_prompt;
            continue;
        }

        if (param_type == "Int" && utils::contains_key(param_name, {"width"})) {
            if (width.empty()) {
                throw std::runtime_error("Width is empty.");
            }
            inputs[param_name] = std::stoi(width);
            continue;
        }

        if (param_type == "Int" && utils::contains_key(param_name, {"height"})) {
            if (height.empty()) {
                throw std::runtime_error("Height is empty.");
            }
            inputs[param_name] = std::stoi(height);
            continue;
        }

        if (param_type == "Int" && utils::contains_key(param_name, {"num_frames"})) {
            if (num_frames.empty()) {
                throw std::runtime_error("Number of frames is empty.");
            }
            inputs[param_name] = std::stoi(num_frames);
            continue;
        }

        if (param_type == "Int" && utils::contains_key(param_name, {"num_inference_steps"})) {
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

        if (param_type == "Int" && utils::contains_key(param_name, {"batch_size"})) {
            if (batch_size.empty()) {
                throw std::runtime_error("Batch size is empty.");
            }
            inputs[param_name] = std::stoi(batch_size);
            continue;
        }

        // In Python sample, it's called num_videos_per_prompt but ParameterModule uses num_images_per_prompt
        if (param_type == "Int" && utils::contains_key(param_name, {"num_images_per_prompt", "num_videos_per_prompt"})) {
            if (num_videos_per_prompt.empty()) {
                throw std::runtime_error("num_videos_per_prompt is empty.");
            }
            inputs[param_name] = std::stoi(num_videos_per_prompt);
            continue;
        }

        if (param_type == "Int" && utils::contains_key(param_name, {"seed"})) {
            if (seed.empty()) {
                throw std::runtime_error("Seed is empty.");
            }
            inputs[param_name] = static_cast<int64_t>(std::stoll(seed));
            continue;
        }
    }

    // Basic sanity checks mirroring the Python sample
    if (inputs.count("height") && inputs.count("width")) {
        const int h = inputs.at("height").as<int>();
        const int w = inputs.at("width").as<int>();
        if (h % 16 != 0 || w % 16 != 0) {
            throw std::runtime_error("`height` and `width` must be divisible by 16.");
        }
    }

    return inputs;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc <= 1) {
            throw std::runtime_error(std::string{"Usage: "} + argv[0] + "\n"
                                     "  -cfg <config.yaml>\n"
                                     "  -prompt <text>\n"
                                     "  --negative_prompt <text>\n"
                                     "  --height <int> (default 480)\n"
                                     "  --width <int> (default 480)\n"
                                     "  --num_frames <int> (default 40)\n"
                                     "  --num_inference_steps <int> (default 50)\n"
                                     "  --guidance_scale <float> (default 5.0)\n"
                                     "  --max_sequence_length <int> (default 512)\n"
                                     "  --batch_size <int> (default 1)\n"
                                     "  --num_videos_per_prompt <int> (default 1)\n"
                                     "  --seed <int> (default 42)\n");
        }

        std::filesystem::path config_path = utils::get_input_arg(argc, argv, "-cfg");
        std::string prompt = utils::get_input_arg(argc, argv, "-prompt");
        std::string negative_prompt = utils::get_input_arg(argc, argv, "--negative_prompt", "");
        std::string width = utils::get_input_arg(argc, argv, "--width", "480");
        std::string height = utils::get_input_arg(argc, argv, "--height", "480");
        std::string num_frames = utils::get_input_arg(argc, argv, "--num_frames", "40");
        std::string num_inference_steps = utils::get_input_arg(argc, argv, "--num_inference_steps", "50");
        std::string guidance_scale = utils::get_input_arg(argc, argv, "--guidance_scale", "5.0");
        std::string max_sequence_length = utils::get_input_arg(argc, argv, "--max_sequence_length", "512");
        std::string batch_size = utils::get_input_arg(argc, argv, "--batch_size", "1");
        std::string num_videos_per_prompt = utils::get_input_arg(argc, argv, "--num_videos_per_prompt", "1");
        std::string seed = utils::get_input_arg(argc, argv, "--seed", "42");

        ov::AnyMap inputs = parse_inputs_from_yaml_cfg_for_video_generation(config_path,
                                                                            prompt,
                                                                            negative_prompt,
                                                                            width,
                                                                            height,
                                                                            num_frames,
                                                                            num_inference_steps,
                                                                            guidance_scale,
                                                                            max_sequence_length,
                                                                            batch_size,
                                                                            num_videos_per_prompt,
                                                                            seed);

        std::cout << "Final pipeline inputs:" << std::endl;
        for (const auto& [key, value] : inputs) {
            std::cout << "  - " << key << ": " << utils::any_to_string(value) << std::endl;
        }

        ov::genai::module::ModulePipeline pipe(config_path);
        pipe.generate(inputs);

        auto saved_video_output = pipe.get_output("saved_video");
        if (!saved_video_output.is<std::string>()) {
            throw std::runtime_error("Pipeline output `saved_video` is not a string. Ensure your YAML includes SaveVideoModule and ResultModule exports `saved_video`.");
        }
        const std::string saved_video_path = saved_video_output.as<std::string>();
        if (saved_video_path.empty()) {
            throw std::runtime_error("Pipeline returned empty `saved_video` path.");
        }

        std::cout << "Video saved to " << saved_video_path << std::endl;
        if (std::filesystem::exists(saved_video_path)) {
            std::cout << "Video file size: " << std::filesystem::file_size(saved_video_path) << " bytes" << std::endl;
        } else {
            std::cout << "Warning: output file does not exist on disk: " << saved_video_path << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}