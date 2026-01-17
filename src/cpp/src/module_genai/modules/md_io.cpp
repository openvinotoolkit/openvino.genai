// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_io.hpp"

#include "module_genai/module_factory.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(ParameterModule);
GENAI_REGISTER_MODULE_SAME(ResultModule);

void ParameterModule::print_static_config() {
    std::cout << R"(
  image:                        # Module Name
    type: "ParameterModule"
    description: "Input parameters. Supported DataType: [OVTensor, VecOVTensor, String, VecString]"
    outputs:
      - name: "image1_data"     # Input Name, should algin with pipeline.generate inputs.
        type: "OVTensor"
      - name: "image2_data"
        type: "OVTensor"
        )" << std::endl;
}

ParameterModule::ParameterModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    is_input_module = true;
    // std::cout << "ParameterModule:" << m_desc << std::endl;
}
ParameterModule::~ParameterModule() {}

void ParameterModule::run(ov::AnyMap& inputs) {
    GENAI_INFO("Running module: " + module_desc->name);

    for (auto& output : this->outputs) {
        OPENVINO_ASSERT(inputs.find(output.first) != inputs.end(), "Can't find input data:" + output.first);
        output.second.data = inputs[output.first];
        GENAI_INFO("    Pass " + output.first + " to output port");
    }
}

void ParameterModule::run() {
    OPENVINO_ASSERT(false, "ParameterModule::run() should not be called");
}

void ResultModule::print_static_config() {
    std::cout << R"(
  pipeline_result:          # Module Name
    type: "ResultModule"
    description: "Output result. Supported DataType: [OVTensor, VecOVTensor, String, VecString]"
    device: "CPU"
    inputs:
      - name: "raw_data"
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
    )" << std::endl;
}

ResultModule::ResultModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    is_output_module = true;
}

ResultModule::~ResultModule() {}

void ResultModule::run(ov::AnyMap& outputs) {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();
    
    for (auto& port_name : module_desc->inputs) {
        auto raw_data = this->inputs[port_name.source_module_out_name].data;
        outputs[port_name.source_module_out_name] = raw_data;
    }
}

void ResultModule::run() {
    OPENVINO_ASSERT(false, "ResultModule::run() should not be called");
}

}  // namespace module
}  // namespace genai
}  // namespace ov
