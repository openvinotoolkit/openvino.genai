// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <openvino/genai/module_genai/pipeline.hpp>
#include <thread>

#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "module_genai/module_base.hpp"
#include "module_genai/module_factory.hpp"
#include "utils/load_image.hpp"
#include "utils/model_yaml.hpp"
#include "utils/utils.hpp"

// Test for ModulePipeline generate_async function with different thread modes.
// The test verifies that modules configured to run in SYNC mode execute on the same thread,
// while those in ASYNC mode may run on different threads(Checked by comparing thread ids).

// Define Dummy Modules for testing
namespace ov::genai::module {
namespace PipelineGenerateAsyncTest {
static std::thread::id dummy_module_a_thread_id;
static std::thread::id dummy_module_b_thread_id;

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
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        dummy_module_a_thread_id = std::this_thread::get_id();
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
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        dummy_module_b_thread_id = std::this_thread::get_id();
    }
};

}  // namespace PipelineGenerateAsyncTest
}  // namespace ov::genai::module

// Define test parameters:
// bool: generate_async or generate;
// vector<int>: module ids of sync execution order when run: generate_async();
using test_params = std::tuple<bool, std::vector<std::string>>;

class PipelineGenerateAsyncTest : public ov::genai::module::ModuleTestBase,
                                  public ::testing::TestWithParam<test_params> {
private:
    std::vector<std::string> _sync_execution_module_names;
    bool not_in_sync_execution_modules(const std::string& module_name) const {
        return std::find(_sync_execution_module_names.begin(), _sync_execution_module_names.end(), module_name) ==
               _sync_execution_module_names.end();
    }

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        // Get image paths and device from parameters
        const auto& async = std::get<0>(obj.param);
        const auto& sync_exec_module_names = std::get<1>(obj.param);
        std::string result;
        result += std::string("ThreadMode_") + (async ? "Async" : "Sync");
        if (async) {
            result += "_SyncModules_";
            for (size_t i = 0; i < sync_exec_module_names.size(); ++i) {
                result += sync_exec_module_names[i];
                if (i < sync_exec_module_names.size() - 1)
                    result += "_";
            }
            if (sync_exec_module_names.size() == 0) {
                result += "None";
            }
        }
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_async, _sync_execution_module_names) = GetParam();

        auto dummy_module_a_instance = std::make_shared<ov::genai::module::PipelineGenerateAsyncTest::DummyModuleA>();
        auto dummy_module_b_instance = std::make_shared<ov::genai::module::PipelineGenerateAsyncTest::DummyModuleB>();
        REGISTER_DUMMY_MODULE_IMPL(ov::genai::module::PipelineGenerateAsyncTest::DummyModuleA::get_name(), dummy_module_a_instance);
        REGISTER_DUMMY_MODULE_IMPL(ov::genai::module::PipelineGenerateAsyncTest::DummyModuleB::get_name(), dummy_module_b_instance);
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
         *         /            \
         *  dummy_module_a    dummy_module_b
         *         \            /
         *          \          /
         *          output_module
         */

        {
            YAML::Node input_module;
            input_module["type"] = "ParameterModule";
            YAML::Node outputs;
            outputs.push_back(output_node("input_data_1", "OVTensor"));
            outputs.push_back(output_node("input_data_2", "OVTensor"));
            input_module["outputs"] = outputs;
            pipeline_modules["input_node"] = input_module;
        }

        {
            YAML::Node dummy_module_a;
            dummy_module_a["type"] = "DummyModule";
            YAML::Node inputs_a;
            inputs_a.push_back(input_node("input_data", "OVTensor", "input_node.input_data_1"));
            dummy_module_a["inputs"] = inputs_a;
            YAML::Node outputs_a;
            outputs_a.push_back(output_node("output_data", "OVTensor"));
            dummy_module_a["outputs"] = outputs_a;
            pipeline_modules[ov::genai::module::PipelineGenerateAsyncTest::DummyModuleA::get_name()] = dummy_module_a;

            dummy_module_a["thread_mode"] =
                (m_async &&
                 not_in_sync_execution_modules(ov::genai::module::PipelineGenerateAsyncTest::DummyModuleA::get_name()))
                    ? "ASYNC"
                    : "SYNC";
        }

        {
            YAML::Node dummy_module_b;
            dummy_module_b["type"] = "DummyModule";
            YAML::Node inputs_b;
            inputs_b.push_back(input_node("input_data", "OVTensor", "input_node.input_data_2"));
            dummy_module_b["inputs"] = inputs_b;
            YAML::Node outputs_b;
            outputs_b.push_back(output_node("output_data", "OVTensor"));
            dummy_module_b["outputs"] = outputs_b;
            pipeline_modules[ov::genai::module::PipelineGenerateAsyncTest::DummyModuleB::get_name()] = dummy_module_b;

            dummy_module_b["thread_mode"] =
                (m_async &&
                 not_in_sync_execution_modules(ov::genai::module::PipelineGenerateAsyncTest::DummyModuleB::get_name()))
                    ? "ASYNC"
                    : "SYNC";
        }

        {
            YAML::Node output_module;
            output_module["type"] = "ResultModule";
            YAML::Node inputs;
            inputs.push_back(
                input_node("output_data_1",
                           "OVTensor",
                           ov::genai::module::PipelineGenerateAsyncTest::DummyModuleA::get_name() + ".output_data"));
            inputs.push_back(
                input_node("output_data_2",
                           "OVTensor",
                           ov::genai::module::PipelineGenerateAsyncTest::DummyModuleB::get_name() + ".output_data"));
            output_module["inputs"] = inputs;
            pipeline_modules["output_node"] = output_module;
        }

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["input_data_1"] = ov::Tensor();
        inputs["input_data_2"] = ov::Tensor();
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        if (m_async && _sync_execution_module_names.size() == 0) {
            EXPECT_NE(ov::genai::module::PipelineGenerateAsyncTest::dummy_module_a_thread_id,
                      ov::genai::module::PipelineGenerateAsyncTest::dummy_module_b_thread_id);
        } else {
            EXPECT_EQ(ov::genai::module::PipelineGenerateAsyncTest::dummy_module_a_thread_id,
                      ov::genai::module::PipelineGenerateAsyncTest::dummy_module_b_thread_id);
        }
    }
};

TEST_P(PipelineGenerateAsyncTest, ModuleTest) {
    run();
}

static std::vector<test_params> g_test_params = {
    {true,
     {ov::genai::module::PipelineGenerateAsyncTest::DummyModuleA::get_name(),
      ov::genai::module::PipelineGenerateAsyncTest::DummyModuleB::get_name()}},
    {true, {}},
    {false, {}}};

INSTANTIATE_TEST_SUITE_P(PipelineTestSuite,
                         PipelineGenerateAsyncTest,
                         ::testing::ValuesIn(g_test_params),
                         PipelineGenerateAsyncTest::get_test_case_name);
