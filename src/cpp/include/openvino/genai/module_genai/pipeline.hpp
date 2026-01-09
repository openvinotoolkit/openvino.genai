// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace ov {
namespace genai {

namespace module {

class OPENVINO_GENAI_EXPORTS ModulePipeline {

public:
    // config_yaml_path: yaml file.
    ModulePipeline(const std::filesystem::path& config_yaml_path);

    // config_yaml_content: yaml content string.
    ModulePipeline(const std::string& config_yaml_content);

    ~ModulePipeline();

    // input all parameters in config.yaml
    // "prompt": string
    // "image": image ov::Tensor or std::vector<ov::Tensor>
    // "video": video ov::Tensor
    void generate(ov::AnyMap& inputs, StreamerVariant streamer = std::monostate());

    // execute generate asynchronously
    void generate_async(ov::AnyMap& inputs, StreamerVariant streamer = std::monostate());

    ov::Any get_output(const std::string& output_name);

    void start_chat(const std::string& system_message = {});

    void finish_chat();

private:
    void* m_pipeline_impl = nullptr;
};

void OPENVINO_GENAI_EXPORTS PrintAllModulesConfig();

std::vector<std::string> OPENVINO_GENAI_EXPORTS ListAllModules();

void OPENVINO_GENAI_EXPORTS PrintModuleConfig(const std::string& module_name);

}  // namespace module
}  // namespace genai
}  // namespace ov
