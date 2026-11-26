// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "content_hash.hpp"
#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_offload_cache.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_offload_cache.hpp"
#include "continuous_batching/cache/store_registry.hpp"
#include "logger.hpp"
#include "continuous_batching/cache/offload_settings.hpp"

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
     * @brief Opens the store of the KV cache and, for a hybrid model, of the linear attention cache together, creates
     * their offload tiers and attaches each to its block manager.
     *
     * The stores are opened as one set (see open_grouped_kv_cache_store), so a plugin that serves all groups from one
     * instance sees the KV blocks and the linear attention checkpoints of a request in one place. The slot sizes
     * differ, so each group keeps its own layout and capacity.
     *
     * @param la_manager Linear attention cache manager, or null when that cache is not offloaded.
     */
    void enable(KVCacheManager& kv_manager,
                BlockManager& kv_block_manager,
                LinearAttentionCacheManager* la_manager,
                BlockManager* la_block_manager,
                const OffloadSettings& offload_config,
                const std::string& device) {
        OPENVINO_ASSERT((la_manager == nullptr) == (la_block_manager == nullptr),
                        "The linear attention cache manager and its block manager go together");
        const size_t kv_block_tokens = kv_block_manager.get_block_size();
        std::vector<StoreGroupSpec> specs;
        specs.push_back(StoreGroupSpec{
            "kv",
            group_name("kv", kv_manager.get_block_layout(), kv_block_tokens, device, kv_manager.get_layout_signature()),
            kv_manager.get_block_layout(),
            kv_block_tokens,
            offload_config.capacity_bytes});
        if (la_manager != nullptr) {
            // The LA block manager's block size is the checkpoint interval in tokens.
            const size_t la_interval = la_block_manager->get_block_size();
            specs.push_back(StoreGroupSpec{
                "la",
                group_name("la", la_manager->get_block_layout(), la_interval, device, la_manager->get_layout_signature()),
                la_manager->get_block_layout(),
                la_interval,
                offload_config.linear_attention_capacity_bytes});
        }

        GENAI_INFO("KV cache offload plan: %s", offload_config.to_string().c_str());
        auto stores = open_grouped_kv_cache_store(specs, offload_config, device, shared_namespace(specs));
        auto kv_store = stores->group(0);
        GENAI_INFO("[KV_TRACE] kv_cache_offload slot_bytes=%zu capacity_blocks=%zu",
                   kv_store->block_bytes(),
                   kv_store->capacity_blocks_hint());
        const size_t device_blocks = kv_block_manager.get_total_block_count();
        if (offload_config.store_completed_blocks && device_blocks > 0 && kv_store->capacity_blocks_hint() <= device_blocks) {
            GENAI_WARN("KV cache offload with WRITE_THROUGH holds %zu blocks, no more than the %zu blocks of the device cache: "
                       "within one run it adds nothing, only a later run profits from what it stores",
                       kv_store->capacity_blocks_hint(),
                       device_blocks);
        }
        m_kv = std::make_unique<KVCacheOffloadCache>(kv_manager, std::move(kv_store), OffloadSettings::DEFAULT_BUFFER_SLOTS);
        kv_block_manager.set_overwritten_block_observer(m_kv.get());
        m_kv_block_manager = &kv_block_manager;
        m_kv->set_store_completed_blocks(offload_config.store_completed_blocks);

        if (la_manager != nullptr) {
            auto la_store = stores->group(1);
            GENAI_INFO("[KV_TRACE] la_cache_offload slot_bytes=%zu capacity_blocks=%zu",
                       la_store->block_bytes(),
                       la_store->capacity_blocks_hint());
            m_la = std::make_unique<LinearAttentionOffloadCache>(*la_manager, std::move(la_store), OffloadSettings::DEFAULT_BUFFER_SLOTS);
            la_block_manager->set_overwritten_block_observer(m_la.get());
            m_la_block_manager = la_block_manager;
            m_la->set_store_completed_blocks(offload_config.store_completed_blocks);
            // The LA block manager's block size is the checkpoint interval in tokens.
            m_la->enable_prefix_tokens(la_block_manager->get_block_size());
        }
    }

    /// Namespace of a store instance that serves all groups of @p specs: the namespaces the groups would have apart.
    std::string shared_namespace(const std::vector<StoreGroupSpec>& specs) const {
        if (m_model_identity.empty()) {
            return "kv+la";
        }
        std::string key;
        for (const StoreGroupSpec& spec : specs) {
            key += spec.own_namespace;
            key += '|';
        }
        const ContentHash digest = compute_content_hash(key.data(), key.size());
        char hex[33];
        std::snprintf(hex, sizeof(hex), "%016llx%016llx", static_cast<unsigned long long>(digest.hi),
                      static_cast<unsigned long long>(digest.lo));
        return std::string("kv+la:") + hex;
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

    /// What a warm-up found in the KV and in the linear-attention cache; a cache without a tier reports zeros.
    struct WarmupReport {
        WarmupStats kv;
        WarmupStats la;
    };

    /**
     * @brief Refills the in-memory prefix cache of the registered block managers from the offload tiers, so that the
     * prefix restore that follows sees those blocks as cached. A manager without a tier is skipped.
     *
     * @param kv_block_manager Block manager of the KV cache, or null when the model has none.
     * @param la_block_manager Block manager of the linear attention cache, or null when the model has none.
     * @param begin_device_phase Called before a warm-up writes into the device cache; see DevicePhaseGuard.
     * @return How many tokens of each cache were already in memory and how many the warm-up brought back.
     */
    WarmupReport warm(BlockManager* kv_block_manager,
                      BlockManager* la_block_manager,
                      const SequenceGroup::Ptr& sequence_group,
                      size_t max_processed_tokens,
                      const DevicePhaseGuard& begin_device_phase) {
        WarmupReport report;
        // The reclamation pause and the prefetched blocks of a tier belong to one warm-up at a time.
        const std::lock_guard<std::mutex> warm_lock(m_warm_mutex);
        if (m_kv != nullptr && kv_block_manager != nullptr) {
            KVCacheOffloadCache::ScopedReclamationPause keep_entries(*m_kv);
            kv_block_manager->warm_prefix_cache(sequence_group, *m_kv, max_processed_tokens, begin_device_phase,
                                                &report.kv);
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
                la_block_manager->warm_prefix_cache(sequence_group, *m_la, la_warm_ceiling, begin_device_phase,
                                                    &report.la);
            }
        }
        return report;
    }

    /**
     * @brief Tells each tier's store what the restore of one request found, once the restore is decided. A store that
     * takes the report reports the reuse metrics itself; for any other the host logs one summary line.
     *
     * @param reused_tokens Prefix length the request finally uses, the same for both caches of a hybrid model.
     */
    void report_restore(const SequenceGroup::Ptr& sequence_group, const WarmupReport& found, size_t reused_tokens) {
        const size_t prompt_tokens = sequence_group->get_prompt_len();
        const auto report_tier = [&](auto& tier, const WarmupStats& stats, const char* cache_name) {
            if (tier == nullptr) {
                return;
            }
            const RestoreStats restore_stats{prompt_tokens, stats.device_hit_tokens, stats.store_hit_tokens, reused_tokens};
            if (!tier->report_restore_stats(restore_stats)) {
                GENAI_INFO("%s prefix restore: request=%llu prompt=%zu device_hit=%zu store_hit=%zu reused=%zu",
                           cache_name,
                           static_cast<unsigned long long>(sequence_group->get_request_id()),
                           restore_stats.prompt_tokens,
                           restore_stats.device_hit_tokens,
                           restore_stats.store_hit_tokens,
                           restore_stats.reused_tokens);
            }
        };
        report_tier(m_kv, found.kv, "KV cache");
        report_tier(m_la, found.la, "Linear attention cache");
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
