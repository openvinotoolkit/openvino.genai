// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/session_snapshot.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string_view>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/store_registry.hpp"
#include "logger.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {

constexpr char MAGIC[8] = {'O', 'V', 'G', 'S', 'E', 'S', 'S', '1'};
constexpr uint32_t FORMAT_VERSION = 3;

// v4: no block payloads, only the token history and the prefix hashes of the blocks that were stored
// in the pipeline's prefix-caching offload store.
constexpr uint32_t FORMAT_VERSION_PREFIX_OFFLOAD = 4;
constexpr const char* FORMAT_ENV = "OV_GENAI_SESSION_SNAPSHOT_FORMAT";

#pragma pack(push, 1)
struct PrefixOffloadHeader {
    char magic[8];
    uint32_t format_version;
    uint32_t device_len;
    uint8_t kv_present;
    uint8_t la_present;
    uint8_t is_embeddings;
    uint8_t reserved;
    uint32_t kv_num_layers;
    uint64_t kv_block_size_in_bytes;
    uint64_t kv_slot_bytes;
    uint64_t la_block_size_in_bytes;
    uint64_t la_slot_bytes;
    uint64_t kv_block_tokens;
    uint64_t la_cache_interval;
    // Only meaningful when is_embeddings is set.
    uint64_t hidden_size;
    uint64_t position_dims;
    int64_t rope_delta;
    uint64_t num_tokens;
    uint64_t num_kv_entries;
    uint64_t checksum;
};
#pragma pack(pop)

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
    // The snapshot is reported as saved once this returns, so the plugin must have made it durable.
    store->flush();
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

/// The conversation history a snapshot carries: token ids, or for embeddings-routed models the input
/// embeddings and position ids (the sequence keeps no token ids for those, so they cannot be recomputed).
struct SnapshotHistory {
    bool is_embeddings = false;
    std::vector<int64_t> token_ids;
    std::vector<uint8_t> embeds_bytes;
    std::vector<uint8_t> position_ids_bytes;
    uint64_t hidden_size = 0;
    uint64_t position_dims = 0;
    uint64_t num_positions = 0;
    int64_t rope_delta = 0;
};

SnapshotHistory capture_history(const SequenceGroup::CPtr& seq_group, const Sequence::CPtr& sequence) {
    SnapshotHistory history;
    history.is_embeddings = seq_group->get_sequence_group_type() == SequenceGroupType::EMBEDDINGS;
    if (history.is_embeddings) {
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

        history.hidden_size = seq_group->get_hidden_size();
        const auto& prompt_embeds = seq_group->get_input_embeds();
        const auto& generated_embeds = sequence->get_generated_ids_embeds();
        history.num_positions = prompt_embeds.size() + generated_embeds.size();
        history.embeds_bytes = flatten_embeds(prompt_embeds, generated_embeds, history.hidden_size);

        const auto& position_ids_list = sequence->get_position_ids_list();
        OPENVINO_ASSERT(position_ids_list.size() == history.num_positions,
                        "Internal error: position ids count does not match the embeddings count");
        history.position_dims = position_ids_list.empty() ? 0 : position_ids_list.front().get_size();
        history.position_ids_bytes = flatten_position_ids(position_ids_list, history.position_dims);
        history.rope_delta = sequence->get_rope_delta();
    } else {
        history.token_ids = seq_group->get_prompt_ids();
        const auto& generated = sequence->get_generated_ids();
        history.token_ids.insert(history.token_ids.end(), generated.begin(), generated.end());
        history.num_positions = history.token_ids.size();
    }
    return history;
}

struct ByteSpan {
    const uint8_t* data;
    size_t size;
};

uint64_t prefix_offload_checksum(PrefixOffloadHeader header,
                                 const std::string& device,
                                 const std::vector<ByteSpan>& history,
                                 const std::vector<SessionSnapshot::PrefixEntry>& entries) {
    header.checksum = 0;
    uint64_t hash = fnv1a(reinterpret_cast<const uint8_t*>(&header), sizeof(header));
    hash = fnv1a(reinterpret_cast<const uint8_t*>(device.data()), device.size(), hash);
    for (const ByteSpan& span : history) {
        hash = fnv1a(span.data, span.size, hash);
    }
    return fnv1a(reinterpret_cast<const uint8_t*>(entries.data()),
                 entries.size() * sizeof(SessionSnapshot::PrefixEntry),
                 hash);
}

void save_prefix_offload_format(CacheOrchestrator& orchestrator,
                                const SequenceGroup::CPtr& seq_group,
                                const Sequence::CPtr& sequence,
                                const std::filesystem::path& path) {
    OPENVINO_ASSERT(orchestrator.has_kv_cache(), "Session snapshot requires a model with KV cache inputs");
    auto* kv_offload = orchestrator.get_kv_offload_cache();
    OPENVINO_ASSERT(kv_offload != nullptr, "Session snapshot format 4 requires KV cache offload to be enabled");
    const bool has_la = orchestrator.has_linear_attention_cache();
    auto* la_offload = has_la ? orchestrator.get_la_offload_cache() : nullptr;
    OPENVINO_ASSERT(!has_la || la_offload != nullptr,
                    "Session snapshot format 4 requires linear-attention cache offload to be enabled "
                    "(cache_offload_config.linear_attention_capacity_bytes)");

    const SnapshotHistory history = capture_history(seq_group, sequence);
    const size_t num_positions = history.num_positions;
    OPENVINO_ASSERT(seq_group->get_num_processed_tokens() + 1 >= num_positions,
                    "Session snapshot cannot be saved while the request is still prefilling: ",
                    seq_group->get_num_processed_tokens(), " of ", num_positions,
                    " positions have been processed. Step the pipeline until prefill completes first");

    auto& kv_manager = static_cast<KVCacheManager&>(orchestrator.get_cache_manager(CacheType::KV_CACHE));
    const size_t block_tokens = orchestrator.get_block_size(CacheType::KV_CACHE);
    // The last position was only sampled, not forwarded, so its KV does not exist.
    const size_t computed_len = num_positions - 1;
    const auto indices = logical_block_indices(orchestrator.get_block_manager(CacheType::KV_CACHE), sequence->get_id());
    // get_hash memoizes the sequence's prefix hashes, hence the mutable access.
    const auto mutable_sequence = std::const_pointer_cast<Sequence>(sequence);

    std::vector<SessionSnapshot::PrefixEntry> entries;
    std::vector<size_t> pinned_hashes;
    std::vector<size_t> pinned_la_hashes;
    try {
        for (size_t block = 0; block * block_tokens < computed_len; ++block) {
            OPENVINO_ASSERT(block < indices.size(), "Sequence has fewer KV blocks than its computed length needs");
            const size_t content_len = std::min((block + 1) * block_tokens, computed_len);
            const size_t hash = mutable_sequence->get_hash(content_len, block_tokens);
            OPENVINO_ASSERT(kv_offload->persist(hash, indices[block]),
                            "Session snapshot could not store KV block ", block,
                            " in the prefix-caching offload store: no slot is free and every entry is pinned. "
                            "Increase cache_offload_config.capacity_bytes");
            pinned_hashes.push_back(hash);
            entries.push_back({content_len, hash});
        }
        const size_t num_kv_entries = entries.size();

        // The recurrent state at the last computed position summarizes the whole history, so one checkpoint
        // is enough. It is stored last in the entry list.
        size_t la_interval = 0;
        if (has_la) {
            la_interval = orchestrator.get_block_size(CacheType::LINEAR_ATTENTION_CACHE);
            const auto la_indices =
                logical_block_indices(orchestrator.get_block_manager(CacheType::LINEAR_ATTENTION_CACHE), sequence->get_id());
            OPENVINO_ASSERT(!la_indices.empty(), "Linear attention cache has no block allocated for this sequence");
            const size_t la_hash = mutable_sequence->get_hash(computed_len, la_interval);
            OPENVINO_ASSERT(la_offload->persist(la_hash, la_indices.back()),
                            "Session snapshot could not store the linear-attention checkpoint in the prefix-caching "
                            "offload store: no slot is free and every entry is pinned. "
                            "Increase cache_offload_config.linear_attention_capacity_bytes");
            pinned_la_hashes.push_back(la_hash);
            entries.push_back({computed_len, la_hash});
        }
        // The snapshot is reported as saved once this returns, so the store must have made it durable.
        kv_offload->flush_durable();
        if (has_la) {
            la_offload->flush_durable();
        }

        const std::string device = orchestrator.get_device();
        PrefixOffloadHeader header{};
        std::memcpy(header.magic, MAGIC, sizeof(MAGIC));
        header.format_version = FORMAT_VERSION_PREFIX_OFFLOAD;
        header.device_len = static_cast<uint32_t>(device.size());
        header.kv_present = 1;
        header.la_present = has_la ? 1 : 0;
        header.kv_num_layers = static_cast<uint32_t>(kv_manager.get_num_layers());
        header.kv_block_size_in_bytes = kv_manager.get_block_size_in_bytes();
        header.kv_slot_bytes = kv_manager.get_block_layout().get_slot_size();
        header.kv_block_tokens = block_tokens;
        if (has_la) {
            const auto& la_manager = static_cast<const LinearAttentionCacheManager&>(
                orchestrator.get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE));
            header.la_block_size_in_bytes = la_manager.get_block_size_in_bytes();
            header.la_slot_bytes = la_manager.get_block_layout().get_slot_size();
            header.la_cache_interval = la_interval;
        }
        header.num_tokens = num_positions;
        header.num_kv_entries = num_kv_entries;
        header.is_embeddings = history.is_embeddings ? 1 : 0;
        header.hidden_size = history.hidden_size;
        header.position_dims = history.position_dims;
        header.rope_delta = history.rope_delta;
        const std::vector<ByteSpan> history_spans =
            history.is_embeddings
                ? std::vector<ByteSpan>{{history.embeds_bytes.data(), history.embeds_bytes.size()},
                                        {history.position_ids_bytes.data(), history.position_ids_bytes.size()}}
                : std::vector<ByteSpan>{{reinterpret_cast<const uint8_t*>(history.token_ids.data()),
                                         history.token_ids.size() * sizeof(int64_t)}};
        header.checksum = prefix_offload_checksum(header, device, history_spans, entries);

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        OPENVINO_ASSERT(out.is_open(), "Failed to create session snapshot file at '", path.string(), "'");
        write_exact(out, &header, sizeof(header));
        write_exact(out, device.data(), device.size());
        for (const ByteSpan& span : history_spans) {
            write_exact(out, span.data, span.size);
        }
        write_exact(out, entries.data(), entries.size() * sizeof(SessionSnapshot::PrefixEntry));
    } catch (...) {
        for (const size_t hash : pinned_hashes) {
            kv_offload->unpin(hash);
        }
        for (const size_t hash : pinned_la_hashes) {
            la_offload->unpin(hash);
        }
        throw;
    }
}

SessionSnapshot::Loaded load_prefix_offload_format(const CacheOrchestrator& orchestrator,
                                                   const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    OPENVINO_ASSERT(in.is_open(), "Session snapshot file not found at '", path.string(), "'");

    PrefixOffloadHeader header{};
    read_exact(in, &header, sizeof(header));

    // Bound every size read from the file by the file itself before allocating anything for it.
    const uint64_t file_size = std::filesystem::file_size(path);
    OPENVINO_ASSERT(header.la_present <= 1 && header.is_embeddings <= 1, "Session snapshot file is truncated or corrupted");
    const uint64_t num_entries = header.num_kv_entries + header.la_present;
    const bool is_embeddings = header.is_embeddings != 0;
    uint64_t embeds_size = 0;
    uint64_t position_ids_size = 0;
    uint64_t token_ids_size = 0;
    bool sizes_in_range = header.device_len <= file_size && header.num_tokens <= file_size &&
                          header.num_kv_entries <= file_size / sizeof(SessionSnapshot::PrefixEntry);
    if (sizes_in_range && is_embeddings) {
        sizes_in_range = header.hidden_size >= 1 && header.hidden_size <= file_size && header.position_dims <= 8 &&
                         header.num_tokens <= file_size / (header.hidden_size * sizeof(float));
        if (sizes_in_range) {
            embeds_size = header.num_tokens * header.hidden_size * sizeof(float);
            position_ids_size = header.num_tokens * header.position_dims * sizeof(int64_t);
        }
    } else if (sizes_in_range) {
        token_ids_size = header.num_tokens * sizeof(int64_t);
    }
    OPENVINO_ASSERT(sizes_in_range && sizeof(header) + header.device_len + token_ids_size + embeds_size +
                                              position_ids_size + num_entries * sizeof(SessionSnapshot::PrefixEntry) ==
                                          file_size,
                    "Session snapshot file is truncated or corrupted");

    std::string device(header.device_len, '\0');
    read_exact(in, device.data(), device.size());
    SessionSnapshot::Loaded result;
    result.prefix_offload_format = true;
    result.is_embeddings = is_embeddings;
    std::vector<uint8_t> embeds_bytes(embeds_size);
    std::vector<uint8_t> position_ids_bytes(position_ids_size);
    std::vector<ByteSpan> history_spans;
    if (is_embeddings) {
        read_exact(in, embeds_bytes.data(), embeds_bytes.size());
        read_exact(in, position_ids_bytes.data(), position_ids_bytes.size());
        history_spans = {{embeds_bytes.data(), embeds_bytes.size()},
                         {position_ids_bytes.data(), position_ids_bytes.size()}};
    } else {
        result.token_ids.resize(header.num_tokens);
        read_exact(in, result.token_ids.data(), result.token_ids.size() * sizeof(int64_t));
        history_spans = {{reinterpret_cast<const uint8_t*>(result.token_ids.data()), token_ids_size}};
    }
    std::vector<SessionSnapshot::PrefixEntry> entries(num_entries);
    read_exact(in, entries.data(), entries.size() * sizeof(SessionSnapshot::PrefixEntry));
    OPENVINO_ASSERT(prefix_offload_checksum(header, device, history_spans, entries) == header.checksum,
                    "Session snapshot failed its integrity check (file may be corrupted)");
    if (is_embeddings) {
        result.hidden_size = header.hidden_size;
        result.rope_delta = header.rope_delta;
        result.input_embeds = ov::Tensor(ov::element::f32, ov::Shape{1, header.num_tokens, header.hidden_size});
        std::memcpy(result.input_embeds.data(), embeds_bytes.data(), embeds_bytes.size());
        const ov::Shape position_ids_shape = header.position_dims > 1
                                                  ? ov::Shape{header.position_dims, 1, header.num_tokens}
                                                  : ov::Shape{1, header.num_tokens};
        result.position_ids = ov::Tensor(ov::element::i64, position_ids_shape);
        std::memcpy(result.position_ids.data(), position_ids_bytes.data(), position_ids_bytes.size());
    }
    result.kv_entries.assign(entries.begin(), entries.begin() + header.num_kv_entries);
    if (header.la_present) {
        result.la_entry = entries.back();
    }

    OPENVINO_ASSERT(header.kv_present && orchestrator.has_kv_cache(),
                    "Session snapshot/model mismatch: snapshot KV cache presence differs from the loaded model");
    OPENVINO_ASSERT(static_cast<bool>(header.la_present) == orchestrator.has_linear_attention_cache(),
                    "Session snapshot/model mismatch: snapshot linear-attention cache presence differs from the "
                    "loaded model");
    OPENVINO_ASSERT(device == orchestrator.get_device(),
                    "Session snapshot was captured on device '", device, "' but the current pipeline uses '",
                    orchestrator.get_device(), "'");

    const auto& kv_manager = static_cast<const KVCacheManager&>(orchestrator.get_cache_manager(CacheType::KV_CACHE));
    OPENVINO_ASSERT(header.kv_num_layers == kv_manager.get_num_layers() &&
                        header.kv_block_size_in_bytes == kv_manager.get_block_size_in_bytes() &&
                        header.kv_slot_bytes == kv_manager.get_block_layout().get_slot_size(),
                    "Session snapshot was captured with an incompatible KV cache layout for the currently "
                    "loaded model");
    // Block hashes chain over whole blocks, so a different block size makes every stored hash unreachable.
    OPENVINO_ASSERT(header.kv_block_tokens == orchestrator.get_block_size(CacheType::KV_CACHE),
                    "Session snapshot was captured with a KV block size of ", header.kv_block_tokens,
                    " tokens but the current pipeline uses ", orchestrator.get_block_size(CacheType::KV_CACHE));
    if (header.la_present) {
        const auto& la_manager = static_cast<const LinearAttentionCacheManager&>(
            orchestrator.get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE));
        OPENVINO_ASSERT(header.la_block_size_in_bytes == la_manager.get_block_size_in_bytes() &&
                            header.la_slot_bytes == la_manager.get_block_layout().get_slot_size(),
                        "Session snapshot was captured with an incompatible linear-attention cache layout for the "
                        "currently loaded model");
        OPENVINO_ASSERT(header.la_cache_interval == orchestrator.get_block_size(CacheType::LINEAR_ATTENTION_CACHE),
                        "Session snapshot was captured with a linear-attention checkpoint interval of ",
                        header.la_cache_interval, " tokens but the current pipeline uses ",
                        orchestrator.get_block_size(CacheType::LINEAR_ATTENTION_CACHE));
    }

    OPENVINO_ASSERT(header.num_tokens >= 1, "Session snapshot holds no tokens");
    uint64_t previous_len = 0;
    for (const auto& entry : result.kv_entries) {
        OPENVINO_ASSERT(entry.content_len > previous_len && entry.content_len <= header.num_tokens - 1,
                        "Session snapshot has an invalid prefix entry (file may be corrupted)");
        previous_len = entry.content_len;
    }
    OPENVINO_ASSERT(!result.la_entry || result.la_entry->content_len == header.num_tokens - 1,
                    "Session snapshot has an invalid linear-attention entry (file may be corrupted)");
    return result;
}

}  // namespace

bool SessionSnapshot::prefix_offload_format_requested(const CacheOrchestrator& orchestrator) {
    const char* value = std::getenv(FORMAT_ENV);
    if (value == nullptr || std::string_view(value) != "4") {
        return false;
    }
    auto* kv_offload = orchestrator.get_kv_offload_cache();
    if (kv_offload == nullptr || (orchestrator.has_linear_attention_cache() && orchestrator.get_la_offload_cache() == nullptr)) {
        GENAI_WARN("%s=4 needs KV (and for hybrid models linear-attention) cache offload to be enabled; falling back to "
                   "the self-contained snapshot format",
                   FORMAT_ENV);
        return false;
    }
    return true;
}

void SessionSnapshot::save(CacheOrchestrator& orchestrator,
                           const SequenceGroup::CPtr& seq_group,
                           const Sequence::CPtr& sequence,
                           const std::filesystem::path& path,
                           const CacheOffloadConfig& store_config) {
    if (prefix_offload_format_requested(orchestrator)) {
        save_prefix_offload_format(orchestrator, seq_group, sequence, path);
        return;
    }

    const bool has_kv = orchestrator.has_kv_cache();
    const bool has_la = orchestrator.has_linear_attention_cache();
    OPENVINO_ASSERT(has_kv, "Session snapshot requires a model with KV cache inputs");
    const bool store_backed = store_plugin_requested(store_config);

    const uint64_t seq_id = sequence->get_id();
    SnapshotHistory history = capture_history(seq_group, sequence);
    const bool is_embeddings = history.is_embeddings;
    std::vector<int64_t>& token_ids = history.token_ids;
    const std::vector<uint8_t>& embeds_bytes = history.embeds_bytes;
    const std::vector<uint8_t>& position_ids_bytes = history.position_ids_bytes;
    const uint64_t hidden_size = history.hidden_size;
    const uint64_t position_dims = history.position_dims;
    const uint64_t num_positions = history.num_positions;
    const int64_t rope_delta = history.rope_delta;

    std::vector<std::vector<uint8_t>> kv_blocks;
    uint64_t kv_block_size_in_bytes = 0;
    uint64_t kv_slot_bytes = 0;
    uint32_t kv_num_layers = 0;
    // restore_state() treats every position but the last as already having its KV computed, so a
    // request still in the middle of a chunked prefill (or not scheduled yet) would be saved with
    // unwritten cache blocks and silently resume from garbage.
    OPENVINO_ASSERT(seq_group->get_num_processed_tokens() + 1 >= num_positions,
                    "Session snapshot cannot be saved while the request is still prefilling: ",
                    seq_group->get_num_processed_tokens(), " of ", num_positions,
                    " positions have been processed. Step the pipeline until prefill completes first");
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

    {
        // Peek at the version to pick the format; anything else falls through to the v3 checks below.
        char magic[sizeof(MAGIC)] = {};
        uint32_t version = 0;
        in.read(magic, sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        const bool is_prefix_offload_format = in.good() && std::memcmp(magic, MAGIC, sizeof(MAGIC)) == 0 &&
                                              version == FORMAT_VERSION_PREFIX_OFFLOAD;
        in.clear();
        in.seekg(0);
        if (is_prefix_offload_format) {
            return load_prefix_offload_format(orchestrator, path);
        }
    }

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
