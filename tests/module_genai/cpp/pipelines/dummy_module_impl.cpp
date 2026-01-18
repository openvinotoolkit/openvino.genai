// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dummy_module_impl.hpp"

#include <chrono>
#include <memory>
#include <thread>

#include "module_genai/module_base.hpp"
#include "module_genai/module_factory.hpp"

namespace ov::genai::module {

std::map<std::string, DummyModuleInterface::PTR> g_dummy_impl_instances_map;

class DummyModuleImpl : public IBaseModule {
protected:
    DummyModuleImpl() = delete;
    DummyModuleImpl(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
        : IBaseModule(desc, pipeline_desc) {
        OPENVINO_ASSERT(g_dummy_impl_instances_map.find(get_name()) != g_dummy_impl_instances_map.end(),
                        "No DummyModuleInterface instance registered for module: " + get_name());

        // Initialize the corresponding DummyModule instance
        g_dummy_impl_instances_map[get_name()]->init(this);
    }

public:
    ~DummyModuleImpl() {}

    using PTR = std::shared_ptr<DummyModuleImpl>;

    static PTR create(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc) {
        return PTR(new DummyModuleImpl(desc, pipeline_desc));
    }

    static void print_static_config() {}

    void run() override {
        prepare_inputs();
        // Call the corresponding DummyModule instance
        g_dummy_impl_instances_map[get_name()]->run(this->inputs, this->outputs);
    }
};

REGISTER_MODULE_CONFIG(DummyModuleImpl);
GENAI_REGISTER_MODULE(ov::genai::module::ModuleType::DummyModule, DummyModuleImpl);

}  // namespace ov::genai::module
