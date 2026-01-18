// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <openvino/genai/module_genai/pipeline.hpp>
#include <openvino/openvino.hpp>
#include <thread>

#include "module_genai/module_base.hpp"
#include "module_genai/module_factory.hpp"
#include "module_genai/pipeline_impl.hpp"
#include "utils/load_image.hpp"
#include "utils/model_yaml.hpp"
#include "utils/ut_modules_base.hpp"
#include "utils/utils.hpp"

// Test for ModulePipeline pass ov::Model as parameter.
// Config yaml only take model_path as param, can't pass ov::Model directly. So we need to
// pass ov::Model in models_map parameter of ModulePipeline constructor.

// Test focus on 3 cases:
// case_1. ModulePipeline with ov::Model passed in models_map, and main module use model from models_map.
// case_2. ModulePipeline with ov::Model passed in models_map, and submodule use model from param.
// case_3. Pass multiple ov::Model;

// Define test parameters:
// bool: Single ov::Model or multiple ov::Model;
// bool: main module or submodule use model from models_map
using test_params = std::tuple<bool, bool>;

// Define Dummy Modules for testing
namespace ov::genai::module {

static bool g_single_ov_model;
static bool g_dummy_module_a_use_model;
static int g_dummy_module_a_got_ovmodel_count = 0;
static int g_dummy_module_b_got_ovmodel_count = 0;

class DummyModuleA : public DummyModuleInterface {
public:
    DummyModuleA() = default;
    static std::string get_name() {
        return "DummyModuleA";
    }
    void init(IBaseModule* p_base_module) override {
        m_base_module = p_base_module;
        if (!g_single_ov_model && g_dummy_module_a_use_model) {
            // Initialize second ov::Model from models_map
            m_ov_model_2 = m_base_module->get_ov_model_from_cfg_models_map("ov_model_2", true);
        }

        // Initialize sub-pipeline
        auto sub_pipeline_name = m_base_module->get_param("sub_module_name");
        m_sub_pipeline_impl =
            init_sub_pipeline(sub_pipeline_name, m_base_module->pipeline_desc, m_base_module->module_desc);
    }

    void run(std::map<std::string, IBaseModule::InputModule>& inputs,
             std::map<std::string, IBaseModule::OutputModule>& outputs) override {
        g_dummy_module_a_got_ovmodel_count += (m_base_module->m_ov_model == nullptr ? 0 : 1);
        g_dummy_module_a_got_ovmodel_count += (m_ov_model_2 == nullptr ? 0 : 1);

        ov::AnyMap sub_inputs;
        sub_inputs["input_data"] = inputs["input_data"].data;
        m_sub_pipeline_impl->generate(sub_inputs);
        outputs["output_data"].data = m_sub_pipeline_impl->get_output("output_data");
    }

private:
    std::shared_ptr<ov::Model> m_ov_model_2 = nullptr;
    ModulePipelineImpl::PTR m_sub_pipeline_impl = nullptr;
};

class DummyModuleB : public DummyModuleInterface {
public:
    DummyModuleB() = default;

    static std::string get_name() {
        return "DummyModuleB";
    }

    void init(IBaseModule* p_base_module) override {
        m_base_module = p_base_module;
        if (!g_single_ov_model && !g_dummy_module_a_use_model) {
            // Initialize second ov::Model from models_map
            m_ov_model_2 = m_base_module->get_ov_model_from_cfg_models_map("ov_model_2", true);
        }
    }

    void run(std::map<std::string, IBaseModule::InputModule>& inputs,
             std::map<std::string, IBaseModule::OutputModule>& outputs) override {
        g_dummy_module_b_got_ovmodel_count += (m_base_module->m_ov_model == nullptr ? 0 : 1);
        g_dummy_module_b_got_ovmodel_count += (m_ov_model_2 == nullptr ? 0 : 1);
    }

private:
    std::shared_ptr<ov::Model> m_ov_model_2 = nullptr;
};

}  // namespace ov::genai::module

using namespace ov::genai::module;
class PipelineTestPassOvModel : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    bool _single_ov_model = true;
    bool _main_module_use_model = true;
    std::string _dummy_module_a_name = ov::genai::module::DummyModuleA::get_name();
    std::string _dummy_module_b_name = ov::genai::module::DummyModuleB::get_name();

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& single_ov_model = std::get<0>(obj.param);
        const auto& main_module_use_model = std::get<1>(obj.param);
        std::string result;
        result += single_ov_model ? "Pass_1_OVModel_" : "Pass_2_OVModel_";
        result += main_module_use_model ? "MainModuleUseOVModel" : "SubModuleUseOVModel";
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();

        auto dmy_module_a_instance = std::make_shared<ov::genai::module::DummyModuleA>();
        auto dmy_module_b_instance = std::make_shared<ov::genai::module::DummyModuleB>();
        REGISTER_DUMMY_MODULE_IMPL(_dummy_module_a_name, dmy_module_a_instance);
        REGISTER_DUMMY_MODULE_IMPL(_dummy_module_b_name, dmy_module_b_instance);
        ov::genai::module::g_dummy_module_a_got_ovmodel_count = 0;
        ov::genai::module::g_dummy_module_b_got_ovmodel_count = 0;

        std::tie(_single_ov_model, _main_module_use_model) = GetParam();
        ov::genai::module::g_single_ov_model = _single_ov_model;
        ov::genai::module::g_dummy_module_a_use_model = _main_module_use_model;

        // load ov model.
        ov::Core core;
        auto model_1 =
            core.read_model(TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4() + "openvino_text_embeddings_model.xml");

        m_models_map.clear();
        std::string module_key = _main_module_use_model ? _dummy_module_a_name : _dummy_module_b_name;
        m_models_map[module_key].clear();
        m_models_map[module_key]["ov_model"] = model_1;
        if (!_single_ov_model) {
            auto model_2 =
                core.read_model(TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4() + "openvino_vision_embeddings_model.xml");
            m_models_map[module_key]["ov_model_2"] = model_2;
        }
    }

    void TearDown() override {
        CLEAR_DUMMY_MODULE_IMPLS();
    }

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "DummyModel";

        YAML::Node pipeline_modules = config["pipeline_modules"];
        // Modules graph
        /*          input_module
         *               |
         *         dummy_module_a --- submodule --> dummy_module_b
         *               |
         *          output_module
         */

        {
            YAML::Node input_module;
            input_module["type"] = "ParameterModule";
            YAML::Node outputs;
            outputs.push_back(output_node("input_data", "OVTensor"));
            input_module["outputs"] = outputs;
            pipeline_modules["input_node"] = input_module;
        }

        {
            YAML::Node dummy_module_a;
            dummy_module_a["type"] = "DummyModule";
            YAML::Node inputs_a;
            inputs_a.push_back(input_node("input_data", "OVTensor", "input_node.input_data"));
            dummy_module_a["inputs"] = inputs_a;
            YAML::Node outputs_a;
            outputs_a.push_back(output_node("output_data", "OVTensor"));
            dummy_module_a["outputs"] = outputs_a;
            YAML::Node params;
            params["sub_module_name"] = "submodule_dummy_module_b";
            dummy_module_a["params"] = params;
            pipeline_modules[_dummy_module_a_name] = dummy_module_a;
        }

        {
            YAML::Node output_module;
            output_module["type"] = "ResultModule";
            YAML::Node inputs;
            inputs.push_back(input_node("output_data", "OVTensor", _dummy_module_a_name + ".output_data"));
            output_module["inputs"] = inputs;
            pipeline_modules["output_node"] = output_module;
        }

        // submodules is a list; each entry is a map with a name and a list of modules.
        YAML::Node submodules(YAML::NodeType::Sequence);
        YAML::Node submodule(YAML::NodeType::Map);
        submodule["name"] = "submodule_dummy_module_b";

        {
            YAML::Node dummy_module_b;
            dummy_module_b["type"] = "DummyModule";
            YAML::Node inputs;
            inputs.push_back(input_node("input_data", "OVTensor"));
            dummy_module_b["inputs"] = inputs;
            YAML::Node outputs;
            outputs.push_back(output_node("output_data", "OVTensor"));
            dummy_module_b["outputs"] = outputs;

            submodule[_dummy_module_b_name] = dummy_module_b;
        }

        submodules.push_back(submodule);
        config["sub_modules"] = submodules;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["input_data"] = ov::Tensor();
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        // Check DummyModuleA got ov::Model count
        int expected_model_count = _single_ov_model ? 1 : 2;
        if (_main_module_use_model) {
            EXPECT_EQ(ov::genai::module::g_dummy_module_a_got_ovmodel_count, expected_model_count);
            EXPECT_EQ(ov::genai::module::g_dummy_module_b_got_ovmodel_count, 0);
        } else {
            EXPECT_EQ(ov::genai::module::g_dummy_module_a_got_ovmodel_count, 0);
            EXPECT_EQ(ov::genai::module::g_dummy_module_b_got_ovmodel_count, expected_model_count);
        }
    }
};

TEST_P(PipelineTestPassOvModel, ModuleTest) {
    run();
}

static std::vector<test_params> g_test_params = {{true, true}, {true, false}, {false, true}};

INSTANTIATE_TEST_SUITE_P(PipelineTestSuite,
                         PipelineTestPassOvModel,
                         ::testing::ValuesIn(g_test_params),
                         PipelineTestPassOvModel::get_test_case_name);
