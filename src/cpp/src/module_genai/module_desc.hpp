// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <any>
#include <functional>
#include <memory>

#include "module_genai/module_data_type.hpp"
#include "module_genai/module_print_config.hpp"
#include "module_genai/module_type.hpp"
#include "openvino/core/any.hpp"
#include "openvino/genai/visibility.hpp"
#include "visual_language/vision_encoder.hpp"

namespace ov {
namespace genai {
namespace module {

struct OutputPort {
    std::string name;
    DataType dt_type;
};

struct InputPort {
    std::string name;
    DataType dt_type;
    std::string source_module_name;
    std::string source_module_out_name;
};

class IBaseModuleDesc {
public:
    virtual ~IBaseModuleDesc() = default;

    std::string name = "Unknown";
    ModuleType type = ModuleType::Unknown;
    std::vector<InputPort> inputs;
    std::vector<OutputPort> outputs;
    std::string device = "CPU"; // default CPU
    std::string description;
    ThreadMode thread_mode = ThreadMode::AUTO;
    std::unordered_map<std::string, std::string> params;
    std::string model_type;

    using PTR = std::shared_ptr<IBaseModuleDesc>;
    static PTR create() {
        return std::make_shared<IBaseModuleDesc>();
    }

    std::string get_full_path(const std::string& fn);
    std::filesystem::path config_root_path = ".";  // default to current directory
};

// map: module name -> module desc
using PipelineModulesDesc = std::unordered_map<std::string, IBaseModuleDesc::PTR>;

// Pipeline-scoped resource cache for sharing compiled models across modules
// Resources are automatically released when PipelineDesc is destroyed
class PipelineResourceCache {
public:
    PipelineResourceCache() = default;
    ~PipelineResourceCache() = default;

    // Non-copyable
    PipelineResourceCache(const PipelineResourceCache&) = delete;
    PipelineResourceCache& operator=(const PipelineResourceCache&) = delete;

    // Get or create a resource with the given key
    // Returns {resource, was_cached} - the bool indicates whether the resource was already in the cache
    template<typename T>
    std::pair<std::shared_ptr<T>, bool> get_or_create(
        std::size_t cache_key,
        std::function<std::shared_ptr<T>()> creator) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache.find(cache_key);
        if (it != m_cache.end()) {
            auto typed = std::any_cast<std::shared_ptr<T>>(&it->second);
            if (!typed) {
                OPENVINO_THROW("PipelineResourceCache: cache key collision - entry exists with a different type for key ",
                               cache_key);
            }
            return {*typed, true};
        }
        auto resource = creator();
        m_cache[cache_key] = resource;
        return {resource, false};
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.clear();
    }

private:
    std::mutex m_mutex;
    std::unordered_map<std::size_t, std::any> m_cache;
};

class PipelineDesc {
protected:
    PipelineDesc();
    PipelineDesc(const PipelineDesc&) = delete;
    PipelineDesc& operator=(const PipelineDesc&) = delete;

public:
    ~PipelineDesc();
    // global_context;
    std::string model_type;

    // main pipeline desc
    PipelineModulesDesc main_pipeline_desc;
    // sub-pipeline name -> sub-pipeline desc
    std::vector<std::pair<std::string, PipelineModulesDesc>> sub_pipeline_descs;

    void setConfigModelsMap(const ConfigModelsMap& models_map) {
        m_models_map = models_map;
    }
    const ConfigModelsMap& getConfigModelsMap() const {
        return m_models_map;
    }

    // Pipeline-scoped resource cache for modules to share resources
    // Resources are released when PipelineDesc is destroyed
    PipelineResourceCache& get_resource_cache();

    using PTR = std::shared_ptr<PipelineDesc>;
    static PTR create() {
        return std::shared_ptr<PipelineDesc>(new PipelineDesc());
    }

private:
    ConfigModelsMap m_models_map;
    std::unique_ptr<PipelineResourceCache> m_resource_cache;
};

}  // namespace module
}  // namespace genai
}  // namespace ov
