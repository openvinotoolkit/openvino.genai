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
        size_t filled_blocks_count = block_start_idx / block_size;
        OPENVINO_ASSERT(filled_blocks_count <= m_prefix_hashes.size());
        if (filled_blocks_count > 0) {
            content.emplace_back(m_prefix_hashes[filled_blocks_count - 1]);
        }

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
                    auto embed = _reduce_embedding(input_embeds[idx]);
                    content.insert(content.end(), embed.begin(), embed.end());
                }
            }

            // get generated ids embeddings
            if (content_length > input_embeds.size()) {
                size_t start = block_start_idx < input_embeds.size() ? 0 : block_start_idx - input_embeds.size();
                for (size_t idx = start; idx < content_length - input_embeds.size(); idx++) {
                    auto embed = _reduce_embedding(generated_embeds[idx]);
                    content.insert(content.end(), embed.begin(), embed.end());
                }
            }
        }
        else {
            OPENVINO_THROW("Hash calculation is not supported for this sequence type.");
        }
        const char* data = reinterpret_cast<const char*>(content.data());
        std::size_t size = content.size() * sizeof(content[0]);
        return std::hash<std::string_view>{}(std::string_view(data, size));
}

std::vector<int64_t> Sequence::_reduce_embedding(const std::vector<float>& embedding) {
    size_t res_size = std::min((size_t)ceil(float(embedding.size()) / m_embeddings_hash_calculation_stride), m_embeddings_hash_max_num_values);
    std::vector<int64_t> res(res_size, 0);
    for (size_t i = 0, idx=0; idx < res_size; i+= m_embeddings_hash_calculation_stride, idx++) {
        std::memcpy(&(res[idx]), &(embedding[i]), sizeof(embedding[i]));
    }
    return res;
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