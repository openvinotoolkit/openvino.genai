// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module_print_config.hpp"
#include "module_genai/module_type.hpp"
#include "openvino/core/any.hpp"
#include "openvino/genai/visibility.hpp"
#include "visual_language/vision_encoder.hpp"

namespace ov {
namespace genai {
namespace module {

enum class DataType : int {
    Unknown = 0,
    OVTensor = 1,
    VecOVTensor = 2,
    OVRemoteTensor = 3,
    VecOVRemoteTensor = 4,
    String = 10,
    VecString = 11,
    Int = 20,
    VecInt = 21,
    VecVecInt = 22,
    Float = 30,
    VecFloat = 31
};

struct OutputPort {
    std::string name;
    DataType dt_type;
};

struct InputPort {
    std::string name;
    DataType dt_type;
    std::string source_module_name;
    std::string source_module_out_name;
};

class IBaseModuleDesc {
public:
    virtual ~IBaseModuleDesc() = default;

    std::string name = "Unknown";
    ModuleType type = ModuleType::Unknown;
    std::vector<InputPort> inputs;
    std::vector<OutputPort> outputs;
    std::string device = "CPU"; // default CPU
    std::string description;
    ThreadMode thread_mode = ThreadMode::AUTO;
    std::unordered_map<std::string, std::string> params;
    std::string model_type;

    using PTR = std::shared_ptr<IBaseModuleDesc>;
    static PTR create() {
        return std::make_shared<IBaseModuleDesc>();
    }

    std::string get_full_path(const std::string& fn);
    std::filesystem::path config_root_path = ".";  // default to current directory
};

// map: module name -> module desc
using PipelineModulesDesc = std::unordered_map<std::string, IBaseModuleDesc::PTR>;

class PipelineDesc {
protected:
    PipelineDesc() = default;
    PipelineDesc(const PipelineDesc&) = delete;
    PipelineDesc& operator=(const PipelineDesc&) = delete;

public:
    ~PipelineDesc() = default;
    // global_context;
    std::string model_type;

    // main pipeline desc
    PipelineModulesDesc main_pipeline_desc;
    // sub-pipeline name -> sub-pipeline desc
    std::vector<std::pair<std::string, PipelineModulesDesc>> sub_pipeline_descs;

    void setConfigModelsMap(const ConfigModelsMap& models_map) {
        m_models_map = models_map;
    }
    const ConfigModelsMap& getConfigModelsMap() const {
        return m_models_map;
    }

    using PTR = std::shared_ptr<PipelineDesc>;
    static PTR create() {
        return std::shared_ptr<PipelineDesc>(new PipelineDesc());
    }

private:
    ConfigModelsMap m_models_map;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
