// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module_genai/module_print_config.hpp"
#include "module_genai/module_type.hpp"
#include "module_genai/module_desc.hpp"
#include "openvino/core/any.hpp"
#include "openvino/genai/visibility.hpp"
#include "visual_language/vision_encoder.hpp"

namespace ov {
namespace genai {
namespace module {

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
    IBaseModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc);

    virtual void prepare_inputs();

    virtual void run() = 0;

    const std::string& get_module_name() const;

    // Port name -> InputModule
    std::map<std::string, InputModule> inputs;
    std::map<std::string, OutputModule> outputs;
    IBaseModuleDesc::PTR module_desc = nullptr;
    PipelineDesc::PTR pipeline_desc = nullptr;
    bool is_input_module = false;
    bool is_output_module = false;

protected:
    bool exists_input(const std::string& input_name) {
        return inputs.find(input_name) != inputs.end();
    }
};

#ifndef DeclareModuleConstructor
#    define DeclareModuleConstructor(class_name)                                                      \
    protected:                                                                                        \
        class_name() = delete;                                                                        \
        class_name(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc);         \
                                                                                                      \
    public:                                                                                           \
        ~class_name();                                                                                \
                                                                                                      \
        void run() override;                                                                          \
                                                                                                      \
        using PTR = std::shared_ptr<class_name>;                                                      \
        static PTR create(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc) { \
            return PTR(new class_name(desc, pipeline_desc));                                          \
        }                                                                                             \
        static void print_static_config()
#endif

}  // namespace module
}  // namespace genai
}  // namespace ov
