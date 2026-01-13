// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_fake.hpp"

namespace ov::genai::module {

#define FAKE_MODULE_IMPL(FM_NAME, TM)                                                          \
    std::thread::id FM_NAME##_thread_id;                                                       \
    void FM_NAME::print_static_config() {}                                                     \
    FM_NAME::FM_NAME(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc) \
        : IBaseModule(desc, pipeline_desc) {}                                                  \
    FM_NAME::~FM_NAME() {}                                                                     \
    void FM_NAME::run() {                                                                      \
        FM_NAME##_thread_id = std::this_thread::get_id();                  \
        std::this_thread::sleep_for(std::chrono::milliseconds(TM));                            \
    }

FAKE_MODULE_IMPL(FakeModuleA, 2);
FAKE_MODULE_IMPL(FakeModuleB, 3);
FAKE_MODULE_IMPL(FakeModuleC, 4);
FAKE_MODULE_IMPL(FakeModuleD, 2);

}  // namespace ov::genai::module