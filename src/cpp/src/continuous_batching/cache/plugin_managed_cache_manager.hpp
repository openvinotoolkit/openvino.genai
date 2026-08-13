// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <memory>
#include <string>

#include "continuous_batching/cache/i_cache_manager.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tp_gpu/paged_attention_cache_controller.hpp"

namespace ov::genai {

/**
 * @brief KV cache owned by the plugin rather than by the pipeline.
 *
 * A tensor-parallel model splits its cache by KV head across several devices,
 * so it is neither one allocation nor one device and cannot be built through a
 * single remote context. Such a plugin hands out a controller instead, and this
 * manager is the thin forwarder to it: the scheduler keeps seeing one cache
 * with one block size, and the plugin keeps the sharding to itself.
 */
class PluginManagedCacheManager : public ICacheManager {
public:
    /**
     * @brief The controller a compiled model offers, or null when it offers
     *        none and the pipeline has to allocate the cache itself.
     */
    static ov::tp_gpu::PagedAttentionCacheControllerPtr find_controller(const ov::CompiledModel& compiled_model) {
        const auto& key = ov::tp_gpu::paged_attention_cache_controller.name();
        const auto supported = compiled_model.get_property(ov::supported_properties);
        if (std::find(supported.begin(), supported.end(), key) == supported.end()) {
            return nullptr;
        }
        return compiled_model.get_property(ov::tp_gpu::paged_attention_cache_controller);
    }

    PluginManagedCacheManager(ov::tp_gpu::PagedAttentionCacheControllerPtr controller,
                              std::vector<std::string> devices)
        : m_controller(std::move(controller)),
          m_devices(std::move(devices)) {
        OPENVINO_ASSERT(m_controller, "Plugin-managed cache manager built without a controller");
        OPENVINO_ASSERT(!m_devices.empty(), "Plugin-managed cache manager built without devices");
    }

    void allocate_cache_if_needed(size_t num_blocks) override {
        m_controller->allocate_cache_if_needed(num_blocks);
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override {
        m_controller->copy_blocks(block_copy_map);
    }

    void zero_blocks(const std::set<size_t>& block_indices) override {
        m_controller->zero_blocks(block_indices);
    }

    void clear() override {
        m_controller->clear();
    }

    size_t get_num_layers() const override {
        return m_controller->get_num_layers();
    }

    size_t get_num_cache_tensors() const override {
        return m_controller->get_num_cache_tensors();
    }

    size_t get_block_size() const override {
        return m_controller->get_block_size();
    }

    std::string get_device() const override {
        return m_devices.front();
    }

    std::vector<std::string> get_devices() const override {
        return m_devices;
    }

    size_t get_block_size_in_bytes() const override {
        return m_controller->get_block_size_in_bytes();
    }

    size_t get_num_allocated_blocks() const override {
        return m_controller->get_num_allocated_blocks();
    }

private:
    ov::tp_gpu::PagedAttentionCacheControllerPtr m_controller;
    /// Every device the cache is spread over. A block spans all of them, so
    /// the memory budget is their sum, not any single one's.
    std::vector<std::string> m_devices;
};

}  // namespace ov::genai
