// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include <map>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace ov {
namespace genai {

namespace module {

// ConfigModelsMap: key is module name, value is pair of model IR as string and shared_ptr<ov::Model>
using ConfigModelsMap = std::map<std::string, std::map<std::string, std::shared_ptr<ov::Model>>>;

class OPENVINO_GENAI_EXPORTS ModulePipeline {

public:
    // config_yaml_path: yaml file.
    ModulePipeline(const std::filesystem::path& config_yaml_path, ConfigModelsMap models_map = {});

    // config_yaml_content: yaml content string.
    ModulePipeline(const std::string& config_yaml_content, ConfigModelsMap models_map = {});

    ~ModulePipeline();

    // input all parameters in config.yaml
    // "prompt": string
    // "image": image ov::Tensor or std::vector<ov::Tensor>
    // "video": video ov::Tensor
    void generate(ov::AnyMap& inputs, StreamerVariant streamer = std::monostate());

    // execute generate asynchronously
    void generate_async(ov::AnyMap& inputs, StreamerVariant streamer = std::monostate());

    ov::Any get_output(const std::string& output_name);

    ov::Any get_output();

    void start_chat(const std::string& system_message = {});

    void finish_chat();

    // Validation result structure
    struct ValidationResult {
        bool valid = false;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };

    // Validate YAML config file
    static ValidationResult validate_config(const std::filesystem::path& config_yaml_path);

    // Validate YAML config content string
    static ValidationResult validate_config_string(const std::string& config_yaml_content);

    // Convert ComfyUI JSON file to YAML config string context
    // @param comfyui_json_path: Path to ComfyUI JSON file (workflow or API format)
    // @param pipeline_inputs: [in/out] Pipeline parameters. Input: model_path_base (default: "./models/"), default_device (default: "CPU").
    //                         Output: Extracted parameters (prompt, guidance_scale, num_inference_steps, width, height, etc.)
    // @return YAML config string, empty string if conversion fails
    static std::string comfyui_json_to_yaml(
        const std::filesystem::path& comfyui_json_path,
        ov::AnyMap& pipeline_inputs);

    // Convert ComfyUI JSON string to YAML config string context
    // @param comfyui_json_content: ComfyUI JSON content string (workflow or API format)
    // @param pipeline_inputs: [in/out] Pipeline parameters. Input: model_path_base (default: "./models/"), default_device (default: "CPU").
    //                         Output: Extracted parameters (prompt, guidance_scale, num_inference_steps, width, height, etc.)
    // @return YAML config string, empty string if conversion fails
    static std::string comfyui_json_string_to_yaml(
        const std::string& comfyui_json_content,
        ov::AnyMap& pipeline_inputs);

private:
    void* m_pipeline_impl = nullptr;
};

void OPENVINO_GENAI_EXPORTS PrintAllModulesConfig();

std::vector<std::string> OPENVINO_GENAI_EXPORTS ListAllModules();

void OPENVINO_GENAI_EXPORTS PrintModuleConfig(const std::string& module_name);

}  // namespace module
}  // namespace genai
}  // namespace ov
