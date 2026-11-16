// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "content_hash.hpp"
#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_offload_cache.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_offload_cache.hpp"
#include "continuous_batching/cache/store_registry.hpp"
#include "logger.hpp"
#include "openvino/genai/cache_offload.hpp"

namespace ov::genai {

/**
 * @brief Owns the offload tiers of the KV and linear-attention caches and everything that connects them to a store:
 * the namespace a store is opened under, the observer registration on the block managers and the warm-up of the
 * in-memory prefix cache from the tiers.
 *
 * The block managers and cache managers stay with the CacheOrchestrator, which only decides where blocks live. The
 * coordinator must be destroyed before the block managers it registered an observer on.
 */
class CacheOffloadCoordinator {
public:
    explicit CacheOffloadCoordinator(std::string model_identity = {}) : m_model_identity(std::move(model_identity)) {}

    CacheOffloadCoordinator(const CacheOffloadCoordinator&) = delete;
    CacheOffloadCoordinator& operator=(const CacheOffloadCoordinator&) = delete;

    ~CacheOffloadCoordinator() {
        // The tiers are registered as observers on their block managers, so detach before either is gone.
        if (m_kv_block_manager != nullptr) {
            m_kv_block_manager->set_overwritten_block_observer(nullptr);
        }
        if (m_la_block_manager != nullptr) {
            m_la_block_manager->set_overwritten_block_observer(nullptr);
        }
    }

    bool enabled() const {
        return m_kv != nullptr || m_la != nullptr;
    }

    /**
     * @brief Creates the disk offload cache for KV blocks and attaches it to the KV block manager.
     */
    void enable_kv(KVCacheManager& kv_manager,
                   BlockManager& block_manager,
                   const CacheOffloadConfig& offload_config,
                   const std::string& device) {
        const size_t kv_block_tokens = block_manager.get_block_size();
        auto store = open_kv_cache_store(kv_manager.get_block_layout(),
                                         offload_config,
                                         device,
                                         group_name("kv", kv_manager.get_block_layout(), kv_block_tokens, device,
                                                    kv_manager.get_layout_signature()),
                                         kv_block_tokens);
        GENAI_INFO("[KV_TRACE] kv_cache_offload slot_bytes=%zu capacity_blocks=%zu",
                   store->get_block_size(),
                   store->get_capacity_blocks());

        m_kv = std::make_unique<KVCacheOffloadCache>(kv_manager, std::move(store), offload_config.buffer_slots);
        block_manager.set_overwritten_block_observer(m_kv.get());
        m_kv_block_manager = &block_manager;
        m_kv->set_store_completed_blocks(offload_config.store_completed_blocks);
    }

    /**
     * @brief Creates the disk offload cache for linear attention state blocks and attaches it to the linear attention
     * block manager. Separate file/cache from the KV one, since the slot sizes differ.
     */
    void enable_linear_attention(LinearAttentionCacheManager& la_manager,
                                 BlockManager& block_manager,
                                 const CacheOffloadConfig& offload_config,
                                 const std::string& device) {
        CacheOffloadConfig la_backend_config = offload_config;
        la_backend_config.capacity_bytes = offload_config.linear_attention_capacity_bytes;
        // The LA block manager's block size is the checkpoint interval in tokens.
        const size_t la_interval = block_manager.get_block_size();
        auto store = open_kv_cache_store(la_manager.get_block_layout(),
                                         la_backend_config,
                                         device,
                                         group_name("la", la_manager.get_block_layout(), la_interval, device,
                                                    la_manager.get_layout_signature()),
                                         la_interval);
        GENAI_INFO("[KV_TRACE] la_cache_offload slot_bytes=%zu capacity_blocks=%zu",
                   store->get_block_size(),
                   store->get_capacity_blocks());

        m_la = std::make_unique<LinearAttentionOffloadCache>(la_manager, std::move(store), offload_config.buffer_slots);
        block_manager.set_overwritten_block_observer(m_la.get());
        m_la_block_manager = &block_manager;
        m_la->set_store_completed_blocks(offload_config.store_completed_blocks);
    }

    /**
     * @brief Namespace the offload store of one cache type is opened under.
     *
     * Without a model identity this is just `base` ("kv" / "la"). With one, the namespace also carries a digest of
     * the identity and the cache layout, so a persistent plugin keeps entries of different models, devices and
     * layouts apart. `layout_signature` carries each segment's precision and shape, so two models whose aggregate
     * block byte size happens to match (e.g. FP16 vs BF16) still open different namespaces.
     */
    std::string group_name(const std::string& base,
                           const KVCacheDiskLayout& layout,
                           size_t block_tokens,
                           const std::string& device,
                           const std::string& layout_signature = "") const {
        if (m_model_identity.empty()) {
            return base;
        }
        const std::string key = m_model_identity + '|' + device + '|' + std::to_string(layout.get_slot_size()) + '|' +
                                std::to_string(layout.get_num_layers()) + '|' + std::to_string(block_tokens) + '|' +
                                layout_signature;
        const ContentHash digest = compute_content_hash(key.data(), key.size());
        char hex[33];
        std::snprintf(hex, sizeof(hex), "%016llx%016llx", static_cast<unsigned long long>(digest.hi),
                      static_cast<unsigned long long>(digest.lo));
        return base + ':' + hex;
    }

    /**
     * @brief Refills the in-memory prefix cache of the registered block managers from the offload tiers, so that the
     * prefix restore that follows sees those blocks as cached. A manager without a tier is skipped.
     *
     * @param kv_block_manager Block manager of the KV cache, or null when the model has none.
     * @param la_block_manager Block manager of the linear attention cache, or null when the model has none.
     * @param begin_device_phase Called before a warm-up writes into the device cache; see DevicePhaseGuard.
     */
    void warm(BlockManager* kv_block_manager,
              BlockManager* la_block_manager,
              const SequenceGroup::Ptr& sequence_group,
              size_t max_processed_tokens,
              const DevicePhaseGuard& begin_device_phase) {
        // The reclamation pause and the prefetched blocks of a tier belong to one warm-up at a time.
        const std::lock_guard<std::mutex> warm_lock(m_warm_mutex);
        if (m_kv != nullptr && kv_block_manager != nullptr) {
            KVCacheOffloadCache::ScopedReclamationPause keep_entries(*m_kv);
            kv_block_manager->warm_prefix_cache(sequence_group,
                                                *m_kv,
                                                std::numeric_limits<size_t>::max(),
                                                begin_device_phase);
        }
        if (m_la != nullptr && la_block_manager != nullptr) {
            // A hybrid restore never goes past the last prompt token or the KV prefix that is in memory, so a
            // checkpoint beyond that would be read for nothing and could crowd out the one that can be used.
            size_t la_warm_ceiling = max_processed_tokens;
            if (kv_block_manager != nullptr) {
                const auto kv_plan = kv_block_manager->get_prefix_restore_plan(sequence_group, max_processed_tokens);
                const size_t last_usable = sequence_group->get_prompt_len() > 0 ? sequence_group->get_prompt_len() - 1 : 0;
                la_warm_ceiling = kv_plan.empty() ? 0 : std::min(kv_plan.cache_token_position, last_usable);
            }
            if (la_warm_ceiling > 0) {
                LinearAttentionOffloadCache::ScopedReclamationPause keep_entries(*m_la);
                la_block_manager->warm_prefix_cache(sequence_group, *m_la, la_warm_ceiling, begin_device_phase);
            }
        }
    }

private:
    std::mutex m_warm_mutex;
    std::string m_model_identity;
    std::unique_ptr<KVCacheOffloadCache> m_kv;
    std::unique_ptr<LinearAttentionOffloadCache> m_la;
    BlockManager* m_kv_block_manager = nullptr;
    BlockManager* m_la_block_manager = nullptr;
};

}  // namespace ov::genai
