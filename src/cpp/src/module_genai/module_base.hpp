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
    std::string device;
    std::string description;
    std::unordered_map<std::string, std::string> params;
    std::string model_type;

    using PTR = std::shared_ptr<IBaseModuleDesc>;
    static PTR create() {
        return std::make_shared<IBaseModuleDesc>();
    }

    std::string get_full_path(const std::string& fn);
    std::filesystem::path config_root_path = ".";  // default to current directory
};

class IBaseModule {
public:
    ~IBaseModule() = default;
    using PTR = std::shared_ptr<IBaseModule>;
    using WEAK_PTR = std::weak_ptr<IBaseModule>;
    struct InputModule {
        IBaseModule::WEAK_PTR module_ptr;
        // std::string out_port_name;
        DataType dt_type;

        ov::Any data;
        std::string parent_port_name;
    };
    struct OutputModule {
        std::vector<IBaseModule::PTR> module_ptrs;
        DataType dt_type;
        ov::Any data;
    };

    IBaseModule() = delete;
    IBaseModule(const IBaseModuleDesc::PTR& desc);

    virtual void prepare_inputs();

    virtual void run() = 0;

    const std::string& get_module_name() const;

    // Port name -> InputModule
    std::map<std::string, InputModule> inputs;
    std::map<std::string, OutputModule> outputs;
    IBaseModuleDesc::PTR module_desc;
    bool is_input_module = false;
    bool is_output_module = false;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
