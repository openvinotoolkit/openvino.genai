// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "module.hpp"
#include "modules/md_io.hpp"
#include "utils/yaml_utils.hpp"

namespace ov {
namespace genai {

namespace module {
class ModulePipelineImpl {
private:
    PipelineModuleInstance m_modules;

public:
    // config_yaml_path: yaml file.
    ModulePipelineImpl(const std::filesystem::path& config_yaml_path);

    ModulePipelineImpl(const std::string& config_yaml_content);

    ~ModulePipelineImpl();

    // input all parameters in config.yaml
    // "prompt": string
    // "image": image ov::Tensor or std::vector<ov::Tensor>
    // "video": video ov::Tensor
    void generate(ov::AnyMap& inputs, StreamerVariant streamer = std::monostate());

    ov::Any get_output(const std::string& output_name);

    void start_chat(const std::string& system_message = {});

    void finish_chat();

private:
    std::map<std::string, ov::Any> outputs;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
