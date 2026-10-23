// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/session_snapshot.hpp"

#include <cstring>
#include <fstream>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/store_registry.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {

constexpr char MAGIC[8] = {'O', 'V', 'G', 'S', 'E', 'S', 'S', '1'};
constexpr uint32_t FORMAT_VERSION = 3;

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];
    uint32_t format_version;
    uint32_t device_len;
    uint8_t kv_present;
    uint8_t la_present;
    uint8_t is_embeddings;
    // When set, KV/linear-attention block payloads were delegated to a third-party store plugin
    // at save() time (SessionSnapshot::save's store_config) instead of being embedded below -
    // load() then requires a matching store_config to read them back. Never derived from file
    // contents: which plugin library to load is always caller-supplied, never data-driven.
    uint8_t store_backed;
    uint32_t kv_num_layers;
    uint64_t kv_block_size_in_bytes;
    uint64_t la_block_size_in_bytes;
    uint64_t num_tokens;  // number of positions (prompt + generated-so-far), for both sequence group types
    uint64_t num_kv_blocks;
    uint64_t kv_slot_bytes;
    uint64_t la_slot_bytes;
    uint64_t kv_payload_checksum;
    uint64_t la_payload_checksum;
    // Only meaningful when is_embeddings is set.
    uint64_t hidden_size;
    uint64_t position_dims;  // number of i64 values per position (3 for M-RoPE, 1 otherwise)
    int64_t rope_delta;
    uint64_t embeds_payload_checksum;
    uint64_t position_ids_payload_checksum;
};
#pragma pack(pop)

uint64_t fnv1a(const uint8_t* data, size_t size, uint64_t seed = 1469598103934665603ULL) {
    uint64_t hash = seed;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t checksum_blocks(const std::vector<std::vector<uint8_t>>& blocks) {
    uint64_t hash = 1469598103934665603ULL;
    for (const auto& block : blocks) {
        hash = fnv1a(block.data(), block.size(), hash);
    }
    return hash;
}

void write_exact(std::ofstream& out, const void* data, size_t size) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    OPENVINO_ASSERT(out.good(), "Failed to write session snapshot file");
}

void read_exact(std::ifstream& in, void* data, size_t size) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
    OPENVINO_ASSERT(in.good(), "Session snapshot file is truncated or unreadable");
}

/// @return The physical block indices for `seq_id`'s shared (layer-0) block table, in logical order.
std::vector<size_t> logical_block_indices(const BlockManager& block_mgr, uint64_t seq_id) {
    const auto& block_tables = block_mgr.get_block_tables(seq_id);
    OPENVINO_ASSERT(!block_tables.empty(), "Sequence has no block table for this cache type");
    std::vector<size_t> indices;
    indices.reserve(block_tables[0].size());
    for (const auto& block : block_tables[0]) {
        indices.push_back(static_cast<size_t>(block->get_index()));
    }
    return indices;
}

/// Flattens the prompt + generated-so-far embeddings into one contiguous [num_positions * hidden_size] f32 buffer.
std::vector<uint8_t> flatten_embeds(const std::vector<std::vector<float>>& prompt_embeds,
                                    const std::vector<std::vector<float>>& generated_embeds,
                                    size_t hidden_size) {
    const size_t total_positions = prompt_embeds.size() + generated_embeds.size();
    std::vector<uint8_t> buffer(total_positions * hidden_size * sizeof(float));
    float* dst = reinterpret_cast<float*>(buffer.data());
    for (const auto& position : prompt_embeds) {
        OPENVINO_ASSERT(position.size() == hidden_size, "Inconsistent embedding hidden size");
        std::memcpy(dst, position.data(), hidden_size * sizeof(float));
        dst += hidden_size;
    }
    for (const auto& position : generated_embeds) {
        OPENVINO_ASSERT(position.size() == hidden_size, "Inconsistent embedding hidden size");
        std::memcpy(dst, position.data(), hidden_size * sizeof(float));
        dst += hidden_size;
    }
    return buffer;
}

/// Flattens one small tensor per position (as produced by Sequence::append_position_ids) into one
/// contiguous [num_positions * position_dims] i64 buffer.
std::vector<uint8_t> flatten_position_ids(const std::vector<ov::Tensor>& position_ids_list, size_t position_dims) {
    std::vector<uint8_t> buffer(position_ids_list.size() * position_dims * sizeof(int64_t));
    int64_t* dst = reinterpret_cast<int64_t*>(buffer.data());
    for (const auto& position : position_ids_list) {
        OPENVINO_ASSERT(position.get_size() == position_dims, "Inconsistent position ids element size");
        std::memcpy(dst, position.data<const int64_t>(), position_dims * sizeof(int64_t));
        dst += position_dims;
    }
    return buffer;
}

bool store_plugin_requested(const CacheOffloadConfig& store_config) {
    return store_config.allow_untrusted_store_plugin && !store_config.store_library_path.empty();
}

/// Writes `blocks` (each exactly `slot_bytes`) through a required store plugin, keyed by their
/// plain position in `blocks` - session snapshot addresses blocks by sequence order, not content
/// hash, so this never shares a key namespace with the opportunistic offload cache's content-hash
/// keys even when both point at the same plugin/directory (see the distinct `group`).
void write_blocks_through_plugin(const std::vector<std::vector<uint8_t>>& blocks,
                                 size_t slot_bytes,
                                 const CacheOffloadConfig& store_config,
                                 const std::string& device,
                                 const std::string& group) {
    if (blocks.empty()) {
        return;
    }
    CacheOffloadConfig scoped_config = store_config;
    scoped_config.capacity_bytes = blocks.size() * slot_bytes;
    const KVCacheDiskLayout layout(std::vector<size_t>{slot_bytes}, std::vector<size_t>{0});
    auto store = open_required_kv_cache_store_plugin(layout, scoped_config, device, group);
    for (size_t i = 0; i < blocks.size(); ++i) {
        OPENVINO_ASSERT(store->write_block(i, blocks[i]),
                        "Session snapshot KV cache store plugin ran out of capacity while saving block ", i,
                        " of ", blocks.size(), " (group '", group, "')");
    }
}

/// Reads back `num_blocks` blocks (each `slot_bytes`) previously written by write_blocks_through_plugin.
void read_blocks_through_plugin(std::vector<std::vector<uint8_t>>& blocks,
                                size_t num_blocks,
                                size_t slot_bytes,
                                const CacheOffloadConfig& store_config,
                                const std::string& device,
                                const std::string& group) {
    blocks.assign(num_blocks, {});
    if (num_blocks == 0) {
        return;
    }
    CacheOffloadConfig scoped_config = store_config;
    scoped_config.capacity_bytes = num_blocks * slot_bytes;
    const KVCacheDiskLayout layout(std::vector<size_t>{slot_bytes}, std::vector<size_t>{0});
    auto store = open_required_kv_cache_store_plugin(layout, scoped_config, device, group);
    for (size_t i = 0; i < num_blocks; ++i) {
        OPENVINO_ASSERT(store->read_block(i, blocks[i]),
                        "Session snapshot KV cache store plugin reports block ", i, " of ", num_blocks,
                        " missing (group '", group, "') - the snapshot may be corrupted, or was saved with a "
                        "different store plugin configuration");
    }
}

}  // namespace

void SessionSnapshot::save(CacheOrchestrator& orchestrator,
                           const SequenceGroup::CPtr& seq_group,
                           const Sequence::CPtr& sequence,
                           const std::filesystem::path& path,
                           const CacheOffloadConfig& store_config) {
    const bool has_kv = orchestrator.has_kv_cache();
    const bool has_la = orchestrator.has_linear_attention_cache();
    OPENVINO_ASSERT(has_kv, "Session snapshot requires a model with KV cache inputs");
    const bool store_backed = store_plugin_requested(store_config);

    const uint64_t seq_id = sequence->get_id();
    const bool is_embeddings = seq_group->get_sequence_group_type() == SequenceGroupType::EMBEDDINGS;

    std::vector<int64_t> token_ids;
    std::vector<uint8_t> embeds_bytes;
    std::vector<uint8_t> position_ids_bytes;
    uint64_t hidden_size = 0;
    uint64_t position_dims = 0;
    uint64_t num_positions = 0;
    int64_t rope_delta = 0;

    if (is_embeddings) {
        // V1 scope: text-only history. Any of these being set means images/video/other multimodal
        // state is part of the conversation, which this format does not yet capture. The tensor
        // fields are default-constructed (empty `_impl`) when unused, so check with `operator!()`
        // rather than calling get_size()/get_shape() on them, which throw on an unset tensor.
        OPENVINO_ASSERT(!seq_group->get_token_type_ids().has_value() &&
                        !seq_group->get_deepstack_visual_embeds() &&
                        !seq_group->get_visual_pos_masks().has_value() &&
                        !seq_group->get_per_layer_inputs(),
                        "Session snapshot only supports text-only requests for embeddings-routed models "
                        "(e.g. no images/video in the conversation history)");

        hidden_size = seq_group->get_hidden_size();
        const auto& prompt_embeds = seq_group->get_input_embeds();
        const auto& generated_embeds = sequence->get_generated_ids_embeds();
        num_positions = prompt_embeds.size() + generated_embeds.size();
        embeds_bytes = flatten_embeds(prompt_embeds, generated_embeds, hidden_size);

        const auto& position_ids_list = sequence->get_position_ids_list();
        OPENVINO_ASSERT(position_ids_list.size() == num_positions,
                        "Internal error: position ids count does not match the embeddings count");
        position_dims = position_ids_list.empty() ? 0 : position_ids_list.front().get_size();
        position_ids_bytes = flatten_position_ids(position_ids_list, position_dims);
        rope_delta = sequence->get_rope_delta();
    } else {
        token_ids = seq_group->get_prompt_ids();
        const auto& generated = sequence->get_generated_ids();
        token_ids.insert(token_ids.end(), generated.begin(), generated.end());
        num_positions = token_ids.size();
    }

    std::vector<std::vector<uint8_t>> kv_blocks;
    uint64_t kv_block_size_in_bytes = 0;
    uint64_t kv_slot_bytes = 0;
    uint32_t kv_num_layers = 0;
    {
        auto& kv_manager = static_cast<KVCacheManager&>(orchestrator.get_cache_manager(CacheType::KV_CACHE));
        kv_block_size_in_bytes = kv_manager.get_block_size_in_bytes();
        kv_num_layers = static_cast<uint32_t>(kv_manager.get_num_layers());
        kv_slot_bytes = kv_manager.get_block_layout().get_slot_size();

        const auto indices = logical_block_indices(orchestrator.get_block_manager(CacheType::KV_CACHE), seq_id);
        kv_blocks.reserve(indices.size());
        for (size_t physical_index : indices) {
            kv_blocks.push_back(kv_manager.read_block(physical_index));
        }
    }

    std::vector<std::vector<uint8_t>> la_blocks;
    uint64_t la_block_size_in_bytes = 0;
    uint64_t la_slot_bytes = 0;
    if (has_la) {
        auto& la_manager =
            static_cast<LinearAttentionCacheManager&>(orchestrator.get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE));
        la_block_size_in_bytes = la_manager.get_block_size_in_bytes();
        la_slot_bytes = la_manager.get_block_layout().get_slot_size();

        const auto indices =
            logical_block_indices(orchestrator.get_block_manager(CacheType::LINEAR_ATTENTION_CACHE), seq_id);
        OPENVINO_ASSERT(!indices.empty(), "Linear attention cache has no block allocated for this sequence");
        // Only the latest checkpoint is needed: the recurrent state already summarizes all prior history.
        la_blocks.push_back(la_manager.read_block(indices.back()));
    }

    FileHeader header{};
    std::memcpy(header.magic, MAGIC, sizeof(MAGIC));
    header.format_version = FORMAT_VERSION;
    const std::string device = orchestrator.get_device();
    header.device_len = static_cast<uint32_t>(device.size());
    header.kv_present = has_kv ? 1 : 0;
    header.la_present = has_la ? 1 : 0;
    header.is_embeddings = is_embeddings ? 1 : 0;
    header.store_backed = store_backed ? 1 : 0;
    header.kv_num_layers = kv_num_layers;
    header.kv_block_size_in_bytes = kv_block_size_in_bytes;
    header.la_block_size_in_bytes = la_block_size_in_bytes;
    header.num_tokens = num_positions;
    header.num_kv_blocks = kv_blocks.size();
    header.kv_slot_bytes = kv_slot_bytes;
    header.la_slot_bytes = la_slot_bytes;
    header.kv_payload_checksum = checksum_blocks(kv_blocks);
    header.la_payload_checksum = has_la ? checksum_blocks(la_blocks) : 0;
    header.hidden_size = hidden_size;
    header.position_dims = position_dims;
    header.rope_delta = rope_delta;
    header.embeds_payload_checksum = is_embeddings ? fnv1a(embeds_bytes.data(), embeds_bytes.size()) : 0;
    header.position_ids_payload_checksum = is_embeddings ? fnv1a(position_ids_bytes.data(), position_ids_bytes.size()) : 0;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    OPENVINO_ASSERT(out.is_open(), "Failed to create session snapshot file at '", path.string(), "'");
    write_exact(out, &header, sizeof(header));
    write_exact(out, device.data(), device.size());
    if (is_embeddings) {
        write_exact(out, embeds_bytes.data(), embeds_bytes.size());
        write_exact(out, position_ids_bytes.data(), position_ids_bytes.size());
    } else {
        write_exact(out, token_ids.data(), token_ids.size() * sizeof(int64_t));
    }
    if (store_backed) {
        const std::string device_for_store = orchestrator.get_device();
        write_blocks_through_plugin(kv_blocks, kv_slot_bytes, store_config, device_for_store,
                                    "session_snapshot_kv:" + path.string());
        if (has_la) {
            write_blocks_through_plugin(la_blocks, la_slot_bytes, store_config, device_for_store,
                                        "session_snapshot_la:" + path.string());
        }
    } else {
        for (const auto& block : kv_blocks) {
            write_exact(out, block.data(), block.size());
        }
        for (const auto& block : la_blocks) {
            write_exact(out, block.data(), block.size());
        }
    }
}

SessionSnapshot::Loaded SessionSnapshot::load(const CacheOrchestrator& orchestrator,
                                              const std::filesystem::path& path,
                                              const CacheOffloadConfig& store_config) {
    std::ifstream in(path, std::ios::binary);
    OPENVINO_ASSERT(in.is_open(), "Session snapshot file not found at '", path.string(), "'");

    FileHeader header{};
    read_exact(in, &header, sizeof(header));
    OPENVINO_ASSERT(std::memcmp(header.magic, MAGIC, sizeof(MAGIC)) == 0,
                    "File at '", path.string(), "' is not a valid OpenVINO GenAI session snapshot");
    OPENVINO_ASSERT(header.format_version == FORMAT_VERSION,
                    "Session snapshot format version ", header.format_version,
                    " is not supported (expected ", FORMAT_VERSION, ")");

    std::string device(header.device_len, '\0');
    read_exact(in, device.data(), header.device_len);

    OPENVINO_ASSERT(header.kv_present && orchestrator.has_kv_cache(),
                    "Session snapshot/model mismatch: snapshot KV cache presence differs from the loaded model");
    OPENVINO_ASSERT(static_cast<bool>(header.la_present) == orchestrator.has_linear_attention_cache(),
                    "Session snapshot/model mismatch: snapshot linear-attention cache presence differs from the "
                    "loaded model");
    OPENVINO_ASSERT(device == orchestrator.get_device(),
                    "Session snapshot was captured on device '", device, "' but the current pipeline uses '",
                    orchestrator.get_device(), "'");

    {
        const auto& kv_manager = static_cast<const KVCacheManager&>(orchestrator.get_cache_manager(CacheType::KV_CACHE));
        OPENVINO_ASSERT(header.kv_num_layers == kv_manager.get_num_layers() &&
                        header.kv_block_size_in_bytes == kv_manager.get_block_size_in_bytes() &&
                        header.kv_slot_bytes == kv_manager.get_block_layout().get_slot_size(),
                        "Session snapshot was captured with an incompatible KV cache layout for the currently "
                        "loaded model");
    }
    if (header.la_present) {
        const auto& la_manager = static_cast<const LinearAttentionCacheManager&>(
            orchestrator.get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE));
        OPENVINO_ASSERT(header.la_block_size_in_bytes == la_manager.get_block_size_in_bytes() &&
                        header.la_slot_bytes == la_manager.get_block_layout().get_slot_size(),
                        "Session snapshot was captured with an incompatible linear-attention cache layout for the "
                        "currently loaded model");
    }

    Loaded result;
    result.is_embeddings = header.is_embeddings != 0;
    if (result.is_embeddings) {
        result.hidden_size = header.hidden_size;
        result.rope_delta = header.rope_delta;

        std::vector<uint8_t> embeds_bytes(header.num_tokens * header.hidden_size * sizeof(float));
        read_exact(in, embeds_bytes.data(), embeds_bytes.size());
        OPENVINO_ASSERT(fnv1a(embeds_bytes.data(), embeds_bytes.size()) == header.embeds_payload_checksum,
                        "Session snapshot embeddings payload failed its integrity check (file may be corrupted)");
        result.input_embeds = ov::Tensor(ov::element::f32, ov::Shape{1, header.num_tokens, header.hidden_size});
        std::memcpy(result.input_embeds.data(), embeds_bytes.data(), embeds_bytes.size());

        std::vector<uint8_t> position_ids_bytes(header.num_tokens * header.position_dims * sizeof(int64_t));
        read_exact(in, position_ids_bytes.data(), position_ids_bytes.size());
        OPENVINO_ASSERT(fnv1a(position_ids_bytes.data(), position_ids_bytes.size()) == header.position_ids_payload_checksum,
                        "Session snapshot position ids payload failed its integrity check (file may be corrupted)");
        const ov::Shape position_ids_shape = header.position_dims > 1
                                                  ? ov::Shape{header.position_dims, 1, header.num_tokens}
                                                  : ov::Shape{1, header.num_tokens};
        result.position_ids = ov::Tensor(ov::element::i64, position_ids_shape);
        std::memcpy(result.position_ids.data(), position_ids_bytes.data(), position_ids_bytes.size());
    } else {
        result.token_ids.resize(header.num_tokens);
        read_exact(in, result.token_ids.data(), header.num_tokens * sizeof(int64_t));
    }

    std::vector<std::vector<uint8_t>> kv_blocks;
    if (header.store_backed) {
        read_blocks_through_plugin(kv_blocks, header.num_kv_blocks, header.kv_slot_bytes, store_config,
                                   orchestrator.get_device(), "session_snapshot_kv:" + path.string());
    } else {
        kv_blocks.resize(header.num_kv_blocks);
        for (auto& block : kv_blocks) {
            block.resize(header.kv_slot_bytes);
            read_exact(in, block.data(), block.size());
        }
    }
    OPENVINO_ASSERT(checksum_blocks(kv_blocks) == header.kv_payload_checksum,
                    "Session snapshot KV cache payload failed its integrity check (file may be corrupted)");
    result.blocks_per_type[CacheType::KV_CACHE] = std::move(kv_blocks);

    if (header.la_present) {
        std::vector<std::vector<uint8_t>> la_blocks;
        if (header.store_backed) {
            read_blocks_through_plugin(la_blocks, 1, header.la_slot_bytes, store_config,
                                       orchestrator.get_device(), "session_snapshot_la:" + path.string());
        } else {
            la_blocks.assign(1, {});
            la_blocks[0].resize(header.la_slot_bytes);
            read_exact(in, la_blocks[0].data(), la_blocks[0].size());
        }
        OPENVINO_ASSERT(checksum_blocks(la_blocks) == header.la_payload_checksum,
                        "Session snapshot linear-attention cache payload failed its integrity check (file may be "
                        "corrupted)");
        result.blocks_per_type[CacheType::LINEAR_ATTENTION_CACHE] = std::move(la_blocks);
    }

    return result;
}

}  // namespace ov::genai
