// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string_view>
#include "content_hash.hpp"
#include "prefix_digest_registry.hpp"
#include "prefix_token_registry.hpp"
#include "sequence_group.hpp"

namespace ov {
namespace genai {

std::mutex Sequence::m_counter_mutex;

size_t Sequence::_make_hash(size_t content_length, size_t block_size) {
    OPENVINO_ASSERT(content_length > 0, "Hash computation requires positive content length");
    OPENVINO_ASSERT(block_size > 0, "Hash computation requires positive block size");
    auto& prefix_hashes = m_prefix_hashes.at(block_size);
    size_t block_start_idx = content_length - (content_length % block_size);
    if (block_start_idx == content_length) {
        block_start_idx -= block_size;
    }

    // hash of current block depends on prefix hashes
    std::vector<int64_t> content;
    size_t filled_blocks_count = block_start_idx / block_size;
    OPENVINO_ASSERT(filled_blocks_count <= prefix_hashes.size());
    if (filled_blocks_count > 0) {
        content.emplace_back(prefix_hashes[filled_blocks_count - 1]);
    }

    _append_block_content(content, block_start_idx, content_length, /*exact_embeddings=*/false);
    const char* data = reinterpret_cast<const char*>(content.data());
    std::size_t size = content.size() * sizeof(content[0]);
    return std::hash<std::string_view>{}(std::string_view(data, size));
}

ContentHash Sequence::_make_digest(size_t content_length, size_t block_size) {
    OPENVINO_ASSERT(content_length > 0, "Digest computation requires positive content length");
    OPENVINO_ASSERT(block_size > 0, "Digest computation requires positive block size");
    size_t block_start_idx = content_length - (content_length % block_size);
    if (block_start_idx == content_length) {
        block_start_idx -= block_size;
    }

    // The digest of the current block chains the digest of the preceding block, like _make_hash() does with hashes.
    auto& digests = m_prefix_digests[block_size];
    const size_t filled_blocks_count = block_start_idx / block_size;
    while (digests.size() < filled_blocks_count) {
        digests.push_back(_make_digest((digests.size() + 1) * block_size, block_size));
    }
    std::vector<int64_t> content;
    if (filled_blocks_count > 0) {
        const ContentHash& previous = digests[filled_blocks_count - 1];
        content.push_back(static_cast<int64_t>(previous.lo));
        content.push_back(static_cast<int64_t>(previous.hi));
    }
    _append_block_content(content, block_start_idx, content_length, /*exact_embeddings=*/true);
    return compute_content_hash(content.data(), content.size() * sizeof(content[0]));
}

ContentHash Sequence::get_digest(size_t content_length, size_t block_size) {
    OPENVINO_ASSERT(content_length > 0, "Digest computation requires positive content length");
    OPENVINO_ASSERT(block_size > 0, "Digest computation requires positive block size");
    if (content_length % block_size != 0) {
        return _make_digest(content_length, block_size);
    }
    const size_t block_idx = content_length / block_size - 1;
    auto& digests = m_prefix_digests[block_size];
    while (digests.size() <= block_idx) {
        digests.push_back(_make_digest((digests.size() + 1) * block_size, block_size));
    }
    return digests[block_idx];
}

void Sequence::_append_block_content(std::vector<int64_t>& content, size_t block_start_idx, size_t content_length, bool exact_embeddings) {
    auto sequence_group = get_sequence_group_ptr();
    // get tokens corresponding to current block
    if (sequence_group->get_sequence_group_type() == SequenceGroupType::TOKENS) {
        const auto& prompt_ids = sequence_group->get_prompt_ids();
        OPENVINO_ASSERT(content_length <= prompt_ids.size() + m_generated_ids.size());
        if (block_start_idx < prompt_ids.size()) {
            content.insert(content.end(), prompt_ids.begin() + block_start_idx, prompt_ids.begin() + std::min(prompt_ids.size(), content_length));
        }
        if (content_length > prompt_ids.size()) {
            size_t start = block_start_idx < prompt_ids.size() ? 0 : block_start_idx - prompt_ids.size();
            // Use parentheses around (content_length - prompt_ids.size()) to suppress MSVC debug assert: "cannot seek vector iterator after end"
            content.insert(content.end(), m_generated_ids.begin() + start, m_generated_ids.begin() + (content_length - prompt_ids.size()));
        }
    }
    else if (sequence_group->get_sequence_group_type() == SequenceGroupType::EMBEDDINGS) {
        const auto& input_embeds = sequence_group->get_input_embeds();
        const auto& generated_embeds = m_generated_ids_embeds;
        OPENVINO_ASSERT(content_length <= input_embeds.size() + generated_embeds.size());

        // get inputs embeddings
        if (block_start_idx < input_embeds.size()) {
            for (size_t idx = block_start_idx; idx < std::min(input_embeds.size(), content_length); idx++) {
                if (m_prefix_publication_limit != std::numeric_limits<size_t>::max()) {
                    const auto& prompt_ids = sequence_group->get_prompt_ids();
                    OPENVINO_ASSERT(prompt_ids.size() == input_embeds.size(),
                                    "Prompt-only embedding prefix caching requires one token ID per embedding");
                    content.push_back(prompt_ids[idx]);
                }
                auto embed = exact_embeddings ? _exact_embedding(input_embeds[idx]) : _reduce_embedding(input_embeds[idx]);
                content.insert(content.end(), embed.begin(), embed.end());
            }
        }

        // get generated ids embeddings
        if (content_length > input_embeds.size()) {
            size_t start = block_start_idx < input_embeds.size() ? 0 : block_start_idx - input_embeds.size();
            for (size_t idx = start; idx < content_length - input_embeds.size(); idx++) {
                auto embed = exact_embeddings ? _exact_embedding(generated_embeds[idx]) : _reduce_embedding(generated_embeds[idx]);
                content.insert(content.end(), embed.begin(), embed.end());
            }
        }
    }
    else {
        OPENVINO_THROW("Hash calculation is not supported for this sequence type.");
    }
}

std::vector<int64_t> Sequence::_reduce_embedding(const std::vector<float>& embedding) {
    size_t res_size = std::min((size_t)ceil(float(embedding.size()) / m_embeddings_hash_calculation_stride), m_embeddings_hash_max_num_values);
    std::vector<int64_t> res(res_size, 0);
    for (size_t i = 0, idx=0; idx < res_size; i+= m_embeddings_hash_calculation_stride, idx++) {
        std::memcpy(&(res[idx]), &(embedding[i]), sizeof(embedding[i]));
    }
    return res;
}

// A block of image or audio embeddings must not be taken for another one because a few sampled values agree, once
// a store keeps blocks for later runs, so the digest covers every value.
std::vector<int64_t> Sequence::_exact_embedding(const std::vector<float>& embedding) {
    std::vector<int64_t> res((embedding.size() * sizeof(float) + sizeof(int64_t) - 1) / sizeof(int64_t), 0);
    std::memcpy(res.data(), embedding.data(), embedding.size() * sizeof(float));
    return res;
}

size_t Sequence::get_prefix_identity_length() const {
    const auto sequence_group = get_sequence_group_ptr();
    return sequence_group->get_prompt_len() + (sequence_group->get_sequence_group_type() == SequenceGroupType::TOKENS
                                                   ? m_generated_ids.size()
                                                   : m_generated_ids_embeds.size());
}

void Sequence::truncate_prefix_hashes() {
    if (m_prefix_hashes.empty() && m_prefix_digests.empty()) {
        return;
    }
    const size_t identity_length = get_prefix_identity_length();
    for (auto& [block_size, prefix_hashes] : m_prefix_hashes) {
        prefix_hashes.resize(std::min(prefix_hashes.size(), identity_length / block_size));
    }
    for (auto& [block_size, digests] : m_prefix_digests) {
        digests.resize(std::min(digests.size(), identity_length / block_size));
    }
}

// Each KV block can be uniquely identified by 
// the tokens within the block and the tokens in the prefix before the block.
// hash(prefix tokens + block tokens) <--> KV Block
size_t Sequence::get_hash(size_t content_length, size_t block_size) {

    if (m_prefix_hash_provider) {
        OPENVINO_ASSERT(content_length > 0 && block_size > 0 && content_length <= m_prefix_publication_limit,
                        "Prefix identity is outside the configured publication range");
        return m_prefix_hash_provider(content_length, block_size);
    }
    auto sequence_group = get_sequence_group_ptr();
    OPENVINO_ASSERT(sequence_group, "Hash computation requires setting of sequence_group ptr.");
    OPENVINO_ASSERT(content_length > 0, "Hash computation requires positive content length");
    OPENVINO_ASSERT(block_size > 0, "Hash computation requires positive block size");
    auto& prefix_hashes = m_prefix_hashes[block_size];
    size_t cur_content = block_size * (prefix_hashes.size() + 1);
    while (cur_content <= content_length)
    {
        prefix_hashes.push_back(_make_hash(cur_content, block_size));
        cur_content += block_size;
    }
    if (content_length % block_size == 0) {
        const size_t hash = prefix_hashes[content_length / block_size - 1];
        record_digest(hash, content_length, block_size);
        return hash;
    }
    
    const size_t hash = _make_hash(content_length, block_size);
    record_digest(hash, content_length, block_size);
    return hash;
}

void Sequence::record_digest(size_t hash, size_t content_length, size_t block_size) {
    if (PrefixDigestRegistry::enabled()) {
        PrefixDigestRegistry::instance().record(hash, get_digest(content_length, block_size));
    }
    // A prefix that is already known is not copied again. Tokens are recorded for the part of the prefix that lies in the
    // prompt: a sequence of embeddings keeps the ids of its prompt only when the caller supplied them.
    if (PrefixTokenRegistry::wants(block_size) && PrefixTokenRegistry::instance().find(hash) == nullptr) {
        const auto sequence_group = get_sequence_group_ptr();
        const auto& prompt_ids = sequence_group->get_prompt_ids();
        std::shared_ptr<std::vector<int64_t>> token_ids;
        if (sequence_group->get_sequence_group_type() == SequenceGroupType::TOKENS) {
            token_ids = std::make_shared<std::vector<int64_t>>();
            token_ids->reserve(content_length);
            _append_block_content(*token_ids, 0, content_length, /*exact_embeddings=*/false);
        } else if (content_length <= prompt_ids.size()) {
            token_ids = std::make_shared<std::vector<int64_t>>(prompt_ids.begin(), prompt_ids.begin() + content_length);
        }
        if (token_ids != nullptr) {
            PrefixTokenRegistry::instance().record(hash, std::move(token_ids));
        }
    }
}

size_t Sequence::get_hash(size_t block_size) {
    auto sequence_group = get_sequence_group_ptr();
    OPENVINO_ASSERT(sequence_group, "Hash computation requires setting of sequence_group ptr.");
    return get_hash(sequence_group->get_context_len(), block_size);
}
}  // namespace genai
}  // namespace ov
