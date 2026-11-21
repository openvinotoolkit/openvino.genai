// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "continuous_batching/cache/i_cache_manager.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/plugin_managed_cache.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai {

/**
 * @brief KV cache whose storage the plugin manages (ov::plugin_managed_cache).
 *
 * A tensor-parallel model splits its cache by KV head across several devices, so it is neither one allocation
 * nor on one device and cannot be built through a single remote context. The plugin keeps the split to itself
 * and answers in model-level terms -- block sizes, shapes and free memory as if the model were not split -- so
 * the scheduler keeps seeing one cache.
 */
class PluginManagedKVCacheManager : public IKVCacheManager {
public:
    /**
     * @brief The plugin-managed cache a compiled model offers, or an empty facade when the pipeline has to
     *        allocate the cache itself.
     */
    static ov::PluginManagedCache find(const ov::CompiledModel& compiled_model) {
        const auto supported = compiled_model.get_property(ov::supported_properties);
        if (std::find(supported.begin(), supported.end(), ov::plugin_managed_cache.name()) == supported.end()) {
            return {};
        }
        return compiled_model.get_property(ov::plugin_managed_cache);
    }

    PluginManagedKVCacheManager(ov::PluginManagedCache cache, std::string device) : m_cache(std::move(cache)), m_device(std::move(device)) {
        OPENVINO_ASSERT(m_cache, "Plugin-managed KV cache manager built without a cache");
        const auto groups = m_cache.get_groups();
        OPENVINO_ASSERT(groups.size() == 1 && groups[0].kind == ov::CacheKind::KEY_VALUE,
                        "Only a single plugin-managed KEY_VALUE cache group is supported");
        m_group = groups[0];

        std::set<size_t> layers;
        for (const auto& tensor : m_group.tensors) {
            layers.insert(tensor.layer);
        }
        m_num_layers = layers.size();
    }

    void allocate_cache_if_needed(size_t num_blocks) override {
        const size_t allocated = m_cache.get_num_allocated_blocks(kv_group);
        if (allocated >= num_blocks) {
            return;
        }
        if (allocated == 0) {
            m_cache.allocate(kv_group, num_blocks, false);
        } else {
            m_cache.reallocate(kv_group, num_blocks, false);
        }
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) override {
        m_cache.copy_blocks(kv_group, block_copy_map);
    }

    void zero_blocks(const std::set<size_t>& block_indices) override {
        m_cache.zero_blocks(kv_group, block_indices);
    }

    void clear() override {
        m_cache.deallocate(kv_group);
    }

    size_t get_num_layers() const override {
        return m_num_layers;
    }

    size_t get_num_cache_tensors() const override {
        return m_group.tensors.size();
    }

    size_t get_block_size() const override {
        return m_group.block_size;
    }

    std::string get_device() const override {
        return m_device;
    }

    std::optional<size_t> get_available_memory() const override {
        return m_cache.get_available_memory();
    }

    size_t get_block_size_in_bytes() const override {
        return m_group.block_size_in_bytes;
    }

    size_t get_num_allocated_blocks() const override {
        return m_cache.get_num_allocated_blocks(kv_group);
    }

    size_t get_v_head_size(size_t layer_id) const override {
        const auto name = "value_cache." + std::to_string(layer_id);
        for (const auto& tensor : m_group.tensors) {
            if (tensor.name == name) {
                return tensor.shape[3].get_length();
            }
        }
        OPENVINO_THROW("The plugin-managed cache has no ", name);
    }

private:
    static constexpr size_t kv_group = 0;

    ov::PluginManagedCache m_cache;
    ov::CacheGroupInfo m_group;
    std::string m_device;
    size_t m_num_layers = 0;
};

}  // namespace ov::genai
