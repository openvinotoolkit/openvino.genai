// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string_view>
#include "sequence_group.hpp"

namespace ov {
namespace genai {

std::mutex Sequence::m_counter_mutex;

size_t Sequence::_make_hash(size_t content_length) {
        auto sequence_group = get_sequence_group_ptr();
        auto block_size = sequence_group->get_block_size();
        size_t block_start_idx = content_length - (content_length % block_size);
        if (block_start_idx == content_length) {
            block_start_idx -= block_size;
        }

        // hash of current block depends on prefix hashes
        std::vector<int64_t> content;
        size_t prefix_hashes_needed_count = block_start_idx / block_size;
        OPENVINO_ASSERT(prefix_hashes_needed_count <= m_prefix_hashes.size());
        content.insert(content.end(), m_prefix_hashes.begin(), m_prefix_hashes.begin() + prefix_hashes_needed_count);

        // get tokens corresponding to current block
        const auto prompt_ids = sequence_group->get_prompt_ids();
        OPENVINO_ASSERT(content_length <= prompt_ids.size() + m_generated_ids.size());
        if (block_start_idx < prompt_ids.size()) {
            content.insert(content.end(), prompt_ids.begin() + block_start_idx, prompt_ids.begin() + std::min(prompt_ids.size(), content_length));
        }
        if (content_length > prompt_ids.size()) {
            size_t start = block_start_idx < prompt_ids.size() ? 0 : block_start_idx - prompt_ids.size();
            content.insert(content.end(), m_generated_ids.begin() + start, m_generated_ids.begin() + content_length - prompt_ids.size());
        }
        const char* data = reinterpret_cast<const char*>(content.data());
        std::size_t size = content.size() * sizeof(content[0]);
        return std::hash<std::string_view>{}(std::string_view(data, size));
}

// Each KV block can be uniquely identified by
// the tokens within the block and the tokens in the prefix before the block.
// hash(prefix tokens + block tokens) <--> KV Block
size_t Sequence::get_hash(size_t content_length) {

    auto sequence_group = get_sequence_group_ptr();
    OPENVINO_ASSERT(sequence_group, "Hash computation requires setting of sequence_group ptr.");
    auto content_len = content_length == 0 ? sequence_group->get_context_len() : content_length;
    auto block_size = sequence_group->get_block_size();
    size_t cur_content = block_size * (m_prefix_hashes.size() + 1);
    while (cur_content <= content_len)
    {
        m_prefix_hashes.push_back(_make_hash(cur_content));
        cur_content += block_size;
    }
    if (content_len % block_size == 0) {
        return m_prefix_hashes[content_len / block_size - 1];
    }

    return _make_hash(content_len);
}
}  // namespace genai
}  // namespace ov
