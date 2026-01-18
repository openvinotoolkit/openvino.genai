#pragma once

#include <thread>

#include "module_genai/module_base.hpp"

namespace ov {
namespace genai {
namespace module {

// A specific Module can't be registered multiple times. So we use this interface to register different
// DummyModule implementations for different test modules via register table(g_dummy_impl_instances_map).

class DummyModuleInterface {
public:
    DummyModuleInterface() = default;
    virtual ~DummyModuleInterface() = default;

    // Initialize dummy module with IBaseModule pointer.
    virtual void init(IBaseModule* p_base_module) = 0;

    // Run function called by DummyModuleImpl.
    virtual void run(std::map<std::string, IBaseModule::InputModule>& inputs, std::map<std::string, IBaseModule::OutputModule>& outputs) = 0;
    using PTR = std::shared_ptr<DummyModuleInterface>;

protected:
    // Passed from init function. Don't own the pointer.
    IBaseModule* m_base_module = nullptr;
};

// Guide to register different DummyModule implementations for different test modules.
/****************************
    Refer: openvino.genai/tests/module_genai/cpp/pipelines/generate_async_function.cpp
    Step 1: Define your own DummyModuleInterface implementation.
    For example:
```
class DummyModuleA : public DummyModuleInterface {
public:
    DummyModuleA() = default;
    static std::string get_name() {
        return "DummyModuleA";
    }
    void init(IBaseModule* p_base_module) override {
        // Note: we must store the IBaseModule pointer for later use.
        m_base_module = p_base_module;
    }
    void run(std::map<std::string, IBaseModule::InputModule>& inputs,
             std::map<std::string, IBaseModule::OutputModule>& outputs) override {
        // Implement module logic here.
    }
};
```

    Step 2: Register your DummyModuleInterface implementation with a unique module name in SetUp().
    For example:
```
void SetUp() override {
    REGISTER_TEST_NAME();

    // Note: use the same module name as defined in YAML config.
    auto dummy_module_a_instance = std::make_shared<ov::genai::module::PipelineGenerateAsyncTest::DummyModuleA>();
    REGISTER_DUMMY_MODULE_IMPL(ov::genai::module::PipelineGenerateAsyncTest::DummyModuleA::get_name(), dummy_module_a_instance);
}
```
    step 3: Clear the registered DummyModuleInterface implementations in TearDown().
    For example:
```
void TearDown() override {
    CLEAR_DUMMY_MODULE_IMPLS();
}
```
****************************/

}  // namespace module
}  // namespace genai
}  // namespace ov
