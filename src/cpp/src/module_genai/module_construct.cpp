// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module.hpp"
#include "module_genai/module_factory.hpp"

namespace ov {
namespace genai {
namespace module {

void module_connect(PipelineModuleInstance& pipeline_instance) {
    std::unordered_map<std::string, IBaseModule::PTR> module_map;
    for (const auto& module_ptr : pipeline_instance) {
        // Process inputs
        for (auto& input : module_ptr->module_desc->inputs) {
            auto it = std::find_if(std::begin(pipeline_instance),
                                   std::end(pipeline_instance),
                                   [&](const IBaseModule::PTR& ptr) {
                                       return ptr->get_module_name() == input.source_module_name;
                                   });
            OPENVINO_ASSERT(it != std::end(pipeline_instance), "Can't find module[" + input.source_module_name + "], please check config yaml.");

            module_ptr->inputs[input.name].module_ptr = *it;

            // IBaseModule::OutputModule outp_module = {module_ptr};
            auto& module_ptrs = (*it)->outputs[input.source_module_out_name].module_ptrs;  
            if (std::find(module_ptrs.begin(), 
                          module_ptrs.end(),
                          module_ptr) == module_ptrs.end()) {
                module_ptrs.push_back(module_ptr);
            }
        }
    }
}

void construct_pipeline(const PipelineModulesDesc& pipeline_modules_desc,
                        PipelineModuleInstance& pipeline_instance,
                        const PipelineDesc::PTR& pipeline_desc) {
    for (auto& module_desc : pipeline_modules_desc) {
        const auto type = module_desc.second->type;
        IBaseModule::PTR module_ptr = ModuleFactory::instance().create(type,
                                                                       module_desc.second,
                                                                       pipeline_desc);
        OPENVINO_ASSERT(module_ptr,
                        "No registered creator for module type: ",
                        ModuleTypeConverter::toString(type),
                        ". Add a registration in the corresponding module .cpp (e.g. GENAI_REGISTER_MODULE_SAME(",
                        ModuleTypeConverter::toString(type),
                        ")) and ensure that translation unit is built/linked.");
        pipeline_instance.push_back(module_ptr);
    }
    module_connect(pipeline_instance);
}

}  // namespace module
}  // namespace genai
}  // namespace ov