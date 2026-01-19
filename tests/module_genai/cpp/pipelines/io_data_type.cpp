// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <openvino/genai/module_genai/pipeline.hpp>
#include <typeindex>
#include <thread>

#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "module_genai/module_base.hpp"
#include "module_genai/module_factory.hpp"
#include "utils/load_image.hpp"
#include "utils/model_yaml.hpp"
#include "utils/utils.hpp"

// Test for supported all IO data types in ModulePipeline.

// Define Dummy Modules for testing
namespace ov::genai::module {
namespace io_data_type_test {

class DummyModuleA : public DummyModuleInterface {
public:
    DummyModuleA() = default;
    static std::string get_name() {
        return "DummyModuleA";
    }
    void init(IBaseModule* p_base_module) override {
        m_base_module = p_base_module;
    }
    void run(std::map<std::string, IBaseModule::InputModule>& inputs,
             std::map<std::string, IBaseModule::OutputModule>& outputs) override {
        auto input_data = inputs["input_data"].data;
        outputs["output_data"].data = input_data;
    }
};

class DummyModuleB : public DummyModuleInterface {
public:
    DummyModuleB() = default;
    static std::string get_name() {
        return "DummyModuleB";
    }
    void init(IBaseModule* p_base_module) override {
        m_base_module = p_base_module;
    }
    void run(std::map<std::string, IBaseModule::InputModule>& inputs,
             std::map<std::string, IBaseModule::OutputModule>& outputs) override {
        auto input_data = inputs["input_data"].data;
        outputs["output_data"].data = input_data;
    }
};

}  // namespace io_data_type_test
}  // namespace ov::genai::module

using namespace ov::genai::module;

// Define test parameters:
// string: data_type
// string: device[CPU, GPU]
using test_params = std::tuple<DataType, std::type_index, ov::Any, std::string>;

class IODataTypeTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    DataType m_data_type;
    std::type_index m_type_index{typeid(void)};
    ov::Any m_input_data;
public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get param from parameters
        DataType data_type = std::get<0>(obj.param);
        std::type_index type_index = std::get<1>(obj.param);
        ov::Any input_data = std::get<2>(obj.param);
        std::string device = std::get<3>(obj.param);
        std::string result;
        result += std::string("DataType_") + to_string(data_type);
        result += std::string("_Device_") + device;
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_data_type, m_type_index, m_input_data, m_device) = GetParam();
        auto dummy_module_a_instance = std::make_shared<ov::genai::module::io_data_type_test::DummyModuleA>();
        auto dummy_module_b_instance = std::make_shared<ov::genai::module::io_data_type_test::DummyModuleB>();
        REGISTER_DUMMY_MODULE_IMPL(ov::genai::module::io_data_type_test::DummyModuleA::get_name(),
                                   dummy_module_a_instance);
        REGISTER_DUMMY_MODULE_IMPL(ov::genai::module::io_data_type_test::DummyModuleB::get_name(),
                                   dummy_module_b_instance);
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
         *          dummy_module_a
         *               |
         *          dummy_module_b
         *               |
         *          output_module
         */

        {
            YAML::Node input_module;
            input_module["type"] = "ParameterModule";
            YAML::Node outputs;
            outputs.push_back(output_node("input_data", to_string(m_data_type)));
            input_module["outputs"] = outputs;
            pipeline_modules["input_node"] = input_module;
        }

        {
            YAML::Node dummy_module_a;
            dummy_module_a["type"] = "DummyModule";
            YAML::Node inputs_a;
            inputs_a.push_back(input_node("input_data", to_string(m_data_type), "input_node.input_data"));
            dummy_module_a["inputs"] = inputs_a;
            YAML::Node outputs_a;
            outputs_a.push_back(output_node("output_data", to_string(m_data_type)));
            dummy_module_a["outputs"] = outputs_a;
            YAML::Node params;
            params["device"] = m_device;
            dummy_module_a["params"] = params;
            pipeline_modules[ov::genai::module::io_data_type_test::DummyModuleA::get_name()] = dummy_module_a;
        }

        {
            YAML::Node dummy_module_b;
            dummy_module_b["type"] = "DummyModule";
            YAML::Node inputs_b;
            inputs_b.push_back(input_node("input_data", to_string(m_data_type), "input_node.input_data"));
            dummy_module_b["inputs"] = inputs_b;
            YAML::Node outputs_b;
            outputs_b.push_back(output_node("output_data", to_string(m_data_type)));
            dummy_module_b["outputs"] = outputs_b;
            YAML::Node params;
            params["device"] = m_device;
            dummy_module_b["params"] = params;
            pipeline_modules[ov::genai::module::io_data_type_test::DummyModuleB::get_name()] = dummy_module_b;
        }

        {
            YAML::Node output_module;
            output_module["type"] = "ResultModule";
            YAML::Node inputs;
            inputs.push_back(
                input_node("output_data",
                           to_string(m_data_type),
                           ov::genai::module::io_data_type_test::DummyModuleB::get_name() + ".output_data"));
            output_module["inputs"] = inputs;
            pipeline_modules["output_node"] = output_module;
        }

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["input_data"] = m_input_data;
        return inputs;
    }

    void check_outputs(ModulePipeline& pipe) override {
        auto output = pipe.get_output("output_data");
        EXPECT_EQ(std::type_index(output.type_info()), m_type_index);
    }
};

TEST_P(IODataTypeTest, FunctionTest) {
    run();
}

static std::vector<test_params> g_test_params = {
    {DataType::OVTensor, std::type_index(typeid(ov::Tensor)), ov::Tensor(), "CPU"},
    {DataType::VecOVTensor, std::type_index(typeid(std::vector<ov::Tensor>)), std::vector<ov::Tensor>(), "CPU"},
    {DataType::OVRemoteTensor, std::type_index(typeid(ov::RemoteTensor)), ov::RemoteTensor(), "CPU"},
    {DataType::OVRemoteTensor, std::type_index(typeid(ov::RemoteTensor)), ov::RemoteTensor(), "GPU"},
    {DataType::VecOVRemoteTensor, std::type_index(typeid(std::vector<ov::RemoteTensor>)), std::vector<ov::RemoteTensor>(), "CPU"},
    {DataType::VecOVRemoteTensor, std::type_index(typeid(std::vector<ov::RemoteTensor>)), std::vector<ov::RemoteTensor>(), "GPU"},
    {DataType::String, std::type_index(typeid(std::string)), std::string("test"), "CPU"},
    {DataType::VecString, std::type_index(typeid(std::vector<std::string>)), std::vector<std::string>{"test1", "test2"}, "CPU"},
    {DataType::Int, std::type_index(typeid(int)), int(1), "CPU"},
    {DataType::VecInt, std::type_index(typeid(std::vector<int>)), std::vector<int>{1, 2}, "CPU"},
    {DataType::VecVecInt, std::type_index(typeid(std::vector<std::vector<int>>)), std::vector<std::vector<int>>{{1, 2}, {3, 4}}, "CPU"},
    {DataType::Float, std::type_index(typeid(float)), float(1.0), "CPU"},
    {DataType::VecFloat, std::type_index(typeid(std::vector<float>)), std::vector<float>{1.0f, 2.0f}, "CPU"}
};
INSTANTIATE_TEST_SUITE_P(PipelineTestSuite,
                         IODataTypeTest,
                         ::testing::ValuesIn(g_test_params),
                         IODataTypeTest::get_test_case_name);
