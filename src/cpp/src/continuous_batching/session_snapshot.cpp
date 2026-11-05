// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/session_snapshot.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {

constexpr char MAGIC[8] = {'O', 'V', 'G', 'S', 'E', 'S', 'S', '1'};

// No block payloads, only the token history and the prefix hashes of the blocks that were stored
// in the pipeline's prefix-caching offload store.
constexpr uint32_t FORMAT_VERSION = 4;

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

uint64_t fnv1a(const uint8_t* data, size_t size, uint64_t seed = 1469598103934665603ULL) {
    uint64_t hash = seed;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
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
    OPENVINO_ASSERT(kv_offload != nullptr,
                    "Session snapshot requires scheduler_config.enable_prefix_caching and scheduler_config.use_cache_offload: "
                    "the snapshot keeps its KV blocks in the prefix-caching offload store");
    const bool has_la = orchestrator.has_linear_attention_cache();
    auto* la_offload = has_la ? orchestrator.get_la_offload_cache() : nullptr;
    OPENVINO_ASSERT(!has_la || la_offload != nullptr,
                    "Session snapshot of a hybrid model requires linear-attention cache offload "
                    "(cache_offload_config.linear_attention_capacity_bytes)");
    // A plugin that failed to load leaves the pipeline on the temporary in-process store; the snapshot would then
    // look saved but lose its blocks as soon as this pipeline ends.
    OPENVINO_ASSERT(!orchestrator.offload_plugin_requested() ||
                        (kv_offload->is_plugin_backed() && (!has_la || la_offload->is_plugin_backed())),
                    "Session snapshot was asked to keep its blocks in a store plugin, but the plugin did not "
                    "load and this pipeline fell back to the temporary in-process store, which would lose them when "
                    "the pipeline ends. Check the plugin path and the 'Falling back to the default file store' warning");

    const SnapshotHistory history = capture_history(seq_group, sequence);
    const size_t num_positions = history.num_positions;
    OPENVINO_ASSERT(seq_group->get_num_processed_tokens() + 1 >= seq_group->get_prompt_len(),
                    "Session snapshot cannot be saved while the request is still prefilling: ",
                    seq_group->get_num_processed_tokens(), " of ", seq_group->get_prompt_len(),
                    " prompt positions have been processed. Step the pipeline until prefill completes first");

    auto& kv_manager = static_cast<KVCacheManager&>(orchestrator.get_cache_manager(CacheType::KV_CACHE));
    const size_t block_tokens = orchestrator.get_block_size(CacheType::KV_CACHE);
    // The last position was only sampled, not forwarded, so its KV does not exist. A speculative draft can lag
    // further behind; the positions it has not forwarded are recomputed after a restore.
    const size_t computed_len = std::min<size_t>(num_positions - 1, seq_group->get_num_processed_tokens());
    OPENVINO_ASSERT(!has_la || computed_len == num_positions - 1,
                    "Session snapshot of a hybrid model needs every position but the last to be computed");
    const auto indices = logical_block_indices(orchestrator.get_block_manager(CacheType::KV_CACHE), sequence->get_id());
    // get_hash memoizes the sequence's prefix hashes, hence the mutable access.
    const auto mutable_sequence = std::const_pointer_cast<Sequence>(sequence);

    std::vector<SessionSnapshot::PrefixEntry> entries;
    std::vector<size_t> pinned_hashes;
    std::vector<size_t> pinned_la_hashes;
    try {
        // The store is asked about the whole chain once, from block 0 in order, before any block is written.
        const size_t num_kv_blocks = (computed_len + block_tokens - 1) / block_tokens;
        std::vector<size_t> kv_chain;
        kv_chain.reserve(num_kv_blocks);
        for (size_t block = 0; block < num_kv_blocks; ++block) {
            kv_chain.push_back(mutable_sequence->get_hash(std::min((block + 1) * block_tokens, computed_len), block_tokens));
        }
        kv_offload->probe(kv_chain);

        for (size_t block = 0; block * block_tokens < computed_len; ++block) {
            OPENVINO_ASSERT(block < indices.size(), "Sequence has fewer KV blocks than its computed length needs");
            const size_t content_len = std::min((block + 1) * block_tokens, computed_len);
            const size_t hash = kv_chain[block];
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
            std::vector<size_t> la_chain;
            for (size_t len = la_interval; len < computed_len; len += la_interval) {
                la_chain.push_back(mutable_sequence->get_hash(len, la_interval));
            }
            la_chain.push_back(la_hash);
            la_offload->probe(la_chain);
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
        header.format_version = FORMAT_VERSION;
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
    OPENVINO_ASSERT(std::memcmp(header.magic, MAGIC, sizeof(MAGIC)) == 0,
                    "File at '", path.string(), "' is not a valid OpenVINO GenAI session snapshot");
    OPENVINO_ASSERT(header.format_version == FORMAT_VERSION,
                    "Session snapshot format version ", header.format_version,
                    " is not supported (expected ", FORMAT_VERSION, ")");

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

void SessionSnapshot::save(CacheOrchestrator& orchestrator,
                           const SequenceGroup::CPtr& seq_group,
                           const Sequence::CPtr& sequence,
                           const std::filesystem::path& path) {
    save_prefix_offload_format(orchestrator, seq_group, sequence, path);
}

SessionSnapshot::Loaded SessionSnapshot::load(const CacheOrchestrator& orchestrator,
                                              const std::filesystem::path& path) {
    return load_prefix_offload_format(orchestrator, path);
}

}  // namespace ov::genai
