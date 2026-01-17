// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <mutex>
#include <unordered_map>

#include "module_genai/module_type.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace genai {
namespace module {

class IBaseModule;
class IBaseModuleDesc;
class PipelineDesc;

namespace detail {
struct EnumClassHash {
    template <typename T>
    std::size_t operator()(T v) const noexcept {
        return std::hash<std::underlying_type_t<T>>{}(static_cast<std::underlying_type_t<T>>(v));
    }
};
}  // namespace detail

class ModuleFactory {
public:
    using Creator = std::function<IBaseModule::PTR(const IBaseModuleDesc::PTR&, const PipelineDesc::PTR&)>;

    static ModuleFactory& instance() {
        static ModuleFactory inst;
        return inst;
    }

    void register_creator(ModuleType type, Creator creator) {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(m_creators.find(type) == m_creators.end(),
                        "Duplicate module registration for type: ",
                        ModuleTypeConverter::toString(type),
                        ". Please ensure only one .cpp registers this ModuleType.");
        m_creators.emplace(type, std::move(creator));
    }

    IBaseModule::PTR create(ModuleType type,
                            const IBaseModuleDesc::PTR& module_desc,
                            const PipelineDesc::PTR& pipeline_desc) const {
        Creator creator;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_creators.find(type);
            if (it == m_creators.end()) {
                return nullptr;
            }
            creator = it->second;
        }
        return creator(module_desc, pipeline_desc);
    }

private:
    ModuleFactory() = default;

    mutable std::mutex m_mutex;
    std::unordered_map<ModuleType, Creator, detail::EnumClassHash> m_creators;
};

template <class ModuleClass>
class ModuleRegistrar {
public:
    explicit ModuleRegistrar(ModuleType type) {
        ModuleFactory::instance().register_creator(
            type,
            [](const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc) {
                return ModuleClass::create(desc, pipeline_desc);
            });
    }
};

// 在模块自己的 .cpp 里调用：GENAI_REGISTER_MODULE(ov::genai::module::ModuleType::VAEDecoderModule, VAEDecoderModule);
#define GENAI_REGISTER_MODULE(module_type_enum, module_class)                                        \
    namespace {                                                                                      \
    const ::ov::genai::module::ModuleRegistrar<::ov::genai::module::module_class>                    \
        ov_genai_module_registrar_##module_class(module_type_enum); \
    }

// 类名与枚举名一致时可用：GENAI_REGISTER_MODULE_SAME(VAEDecoderModule);
#define GENAI_REGISTER_MODULE_SAME(name) GENAI_REGISTER_MODULE(::ov::genai::module::ModuleType::name, name)

}  // namespace module
}  // namespace genai
}  // namespace ov
