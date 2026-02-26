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
    bool is_input() const { return is_input_module; }
    bool is_output() const { return is_output_module; }
    const std::string& get_name() const {
        return module_desc->name;
    }

    std::shared_ptr<ov::Model> m_ov_model = nullptr;
    std::shared_ptr<ov::Model> get_ov_model_from_cfg_models_map(const std::string& param_name, bool required = false);
    // Check if exist input in inputs map.
    bool exists_input(const std::string& input_name);
    ov::Any& get_input(const std::string& input_name);

    // Get param value from module_desc->params.
    // Return empty string if param not found.
    std::string get_param(const std::string& param_item);
    std::string get_optional_param(const std::string& param_item);

protected:
    bool is_input_module = false;
    bool is_output_module = false;

    bool m_dynamic_load_weights = false;  // After inference with larger models, the weights need to be released to free
                                          // up space for inference with other models.
    void check_dynamic_load_weights();    // "dynamic_load_weights" depends on params: "cache_dir"
    std::string m_cache_dir = std::string();
    void check_cache_dir();
    bool m_splitted_model = false;
    void check_splitted_model();

    bool check_bool_param(const std::string& param_name, const bool& default_value);

    // Initialize ov::Model from config models_map with param_name: "ov_model"
    void init_ov_model();
};

#ifndef DeclareModuleConstructorImpl
#    define DeclareModuleConstructorImpl(class_name, dummy_impl, dummy_base_impl)                                   \
    protected:                                                                                                      \
        class_name() = delete;                                                                                      \
        class_name(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc) dummy_base_impl dummy_impl \
                                                                                                                    \
    public:                                                                                                         \
        ~class_name() dummy_impl                                                                                    \
                                                                                                                    \
                 using PTR = std::shared_ptr<class_name>;                                                           \
        static PTR create(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc) {               \
            return PTR(new class_name(desc, pipeline_desc));                                                        \
        }                                                                                                           \
        static void print_static_config() dummy_impl
#endif  // DeclareModuleConstructorImpl

// Module constructor and coommon function declaration macro.
#ifndef DeclareModuleConstructor
#    define DeclareModuleConstructor(class_name) \
    public:                                      \
        void run() override;                     \
        DeclareModuleConstructorImpl(class_name, ;, )
#endif

// Dummy constructor macro, only for test purpose.
#ifndef DeclareModuleConstructorDummy
#    define DeclareModuleConstructorDummy(class_name) \
        DeclareModuleConstructorImpl(class_name, {}, : IBaseModule(desc, pipeline_desc))
#endif

}  // namespace module
}  // namespace genai
}  // namespace ov
