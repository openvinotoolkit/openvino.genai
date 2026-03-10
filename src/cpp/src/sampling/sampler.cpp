// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <future>

#include "sampling/sampler.hpp"
#include "tokenizer/tokenizer_impl.hpp"

namespace ov::genai {
// Modified Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurrence in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, const std::vector<int64_t>& needle) {
    if (needle.empty()) {  // no_repeat_ngram_size == 1, ban every token
        return {haystack.begin(), haystack.end()};
    }
    std::vector<int> partial_match_table(needle.size() + 1, -1);
    int cnd = 0;
    for (size_t pos = 1; pos < needle.size(); ++pos) {
        if (needle.at(pos) == needle.at(size_t(cnd))) {
            partial_match_table.at(pos) = partial_match_table.at(size_t(cnd));
        } else {
            partial_match_table.at(pos) = cnd;
            while (cnd >= 0 && needle.at(pos) != needle.at(size_t(cnd))) {
                cnd = partial_match_table.at(size_t(cnd));
            }
        }
        ++cnd;
    }
    partial_match_table.back() = cnd;
    std::vector<int64_t> res;
    size_t haystack_id = 0;
    int needle_id = 0;
    while (haystack_id < haystack.size() - 1) {
        if (needle.at(size_t(needle_id)) == haystack.at(haystack_id)) {
            ++haystack_id;
            ++needle_id;
            if (needle_id == int(needle.size())) {
                res.push_back(haystack.at(haystack_id));
                needle_id = partial_match_table.at(size_t(needle_id));
            }
        } else {
            needle_id = partial_match_table.at(size_t(needle_id));
            if (needle_id < 0) {
                ++haystack_id;
                ++needle_id;
            }
        }
    }
    return res;
}

std::vector<Token> log_softmax(const ov::Tensor& logits, size_t batch_idx) {
    ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3);
    size_t batch = shape[0], seq_len = shape[1], vocab_size = shape[2];
    OPENVINO_ASSERT(batch_idx < batch, "Logits batch size doesn't match the number of beams");

    size_t batch_offset = batch_idx * seq_len * vocab_size, sequence_offset = (seq_len - 1) * vocab_size;
    const float* beam_logits = logits.data<const float>() + batch_offset + sequence_offset;
    float max_logit = *std::max_element(beam_logits, beam_logits + vocab_size);
    float log_sum = std::log(std::accumulate(
        beam_logits, beam_logits + vocab_size, 0.0f, [max_logit](float accumulated, float to_add) {
            return accumulated + std::exp(to_add - max_logit);
    }));

    std::vector<Token> tokens;
    tokens.reserve(vocab_size);
    for (size_t idx = 0; idx < vocab_size; ++idx)
        tokens.push_back({beam_logits[idx] - max_logit - log_sum, int64_t(idx)});

    return tokens;
}

std::vector<int64_t> wrap_tokens(const std::vector<int64_t>& tokens, const std::vector<int64_t>& prefix_tokens, const std::vector<int64_t>& suffix_tokens) {
    std::vector<int64_t> all_tokens = prefix_tokens;
    all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
    all_tokens.insert(all_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    return all_tokens;
}

std::string clean_wrapped_text(const std::string& wrapped_text, const std::string& prefix, const std::string& suffix) {
    auto prefix_pos = wrapped_text.find(prefix);
    OPENVINO_ASSERT(prefix_pos != std::string::npos);
    auto suffix_pos = wrapped_text.rfind(suffix);
    OPENVINO_ASSERT(suffix_pos != std::string::npos);
    auto clean_text_start = prefix_pos + prefix.size();
    auto clean_text_length = suffix_pos - clean_text_start;
    std::string clean_text = wrapped_text.substr(clean_text_start, clean_text_length);
    return clean_text;
}

std::vector<int64_t> encode_and_process_string(const std::string& stop_string, ov::genai::Tokenizer& tokenizer) {
    // encode stop_string
    std::string stop_string_copy = stop_string;
    ov::Tensor ov_encoded_stop_string = tokenizer.encode(stop_string_copy, ov::genai::add_special_tokens(false)).input_ids;
    size_t tensor_size = ov_encoded_stop_string.get_size();
    std::vector<int64_t> encoded_stop_string(tensor_size);
    std::copy_n(ov_encoded_stop_string.data<int64_t>(), tensor_size, encoded_stop_string.begin());
    return encoded_stop_string;
}

struct MatchStopStringResult {
    size_t to_remove = 0;
    // int64_t last_token_id = 0;
    // bool is_to_update_last_token = false;
    bool is_matched = false;
};

// Return number of last tokens that match one of the stop_strings. If there's no match 0 is returned.
MatchStopStringResult match_stop_string(Tokenizer& tokenizer,
                      const TokenIds& generated_tokens,
                      const std::pair<size_t, std::set<std::string>>& stop_strings,
                      bool is_include_to_output,
                      size_t draft_generated_tokens = 0) {
    MatchStopStringResult result;
    if (generated_tokens.size() >= stop_strings.first) {
        // draft_generated_tokens is to handle case with >= 1 generated tokens per step
        size_t offset = generated_tokens.size() - draft_generated_tokens;
        if (offset < stop_strings.first) {
            return result;
        }
        offset -= stop_strings.first;
        TokenIds buffer(generated_tokens.begin() + offset, generated_tokens.end());
        std::string decoded_buffer = tokenizer.decode(buffer);
        for (const auto& stop_string : stop_strings.second) {
            auto pos = decoded_buffer.find(stop_string);
            if (pos != std::string::npos) {
                result.is_matched = true;

                auto stop_string_len = is_include_to_output ? stop_string.length() : 0;
                decoded_buffer = decoded_buffer.substr(0, pos + stop_string_len);
                // to remove word splitting symbols from tail
                while (decoded_buffer.back() == ' ' || decoded_buffer.back() == '\n') {
                    decoded_buffer.pop_back();
                }
                if (decoded_buffer.empty()) {
                    result.to_remove = buffer.size();
                    return result;
                }

                // find token cnt to be removed from sequence by decoding token by token
                std::string decoded_partially_string;
                for (size_t i = 0; i < buffer.size(); ++i) {
                    decoded_partially_string = tokenizer.decode(TokenIds{buffer.begin(), buffer.begin() + i + 1});
                    if (decoded_partially_string.find(decoded_buffer) != std::string::npos) {
                        result.to_remove = buffer.size() - i - 1;
                        break;
                    }
                }
                return result;
            }
        }
    }
    return result;
}

// Return number of last tokens that match one of the stop_strings. If there's no match 0 is returned.
// Number of tokens might not be exact as if there's no direct token match, we decode generated tokens incrementally expanding decoding scope
// with 4 next tokens with each iteration until we check all tokens.
int match_stop_string2(Tokenizer & tokenizer, const TokenIds & generated_tokens, const std::set<std::string> & stop_strings) {
    for (const auto& stop_string: stop_strings) {
        auto stop_tokens_ov = tokenizer.encode(stop_string).input_ids;
        size_t num_tokens = stop_tokens_ov.get_size();
        if(num_tokens > generated_tokens.size())
            continue;

        // Check direct token match
        std::vector<int64_t> stop_tokens(stop_tokens_ov.data<int64_t>(), stop_tokens_ov.data<int64_t>() + num_tokens);
        std::vector<int64_t> last_generated_tokens(generated_tokens.end()-num_tokens, generated_tokens.end());
        if (stop_tokens == last_generated_tokens)
            return num_tokens;
        
        // Continue checking chunks of 4 tokens
        num_tokens += 4;
        while (num_tokens <= generated_tokens.size()) {
            std::vector<int64_t> last_generated_tokens(generated_tokens.end()-num_tokens, generated_tokens.end());
            std::string decoded_last_tokens = tokenizer.decode(last_generated_tokens);
            if (decoded_last_tokens.find(stop_string) != std::string::npos) {
                return num_tokens;
            }
            num_tokens += 4;
        }
    }
    return 0;
}

void Sampler::GroupBeamSearcher::finalize(SamplerOutput& sampler_output) {
    for (Group& group : m_groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                uint64_t sequence_id = beam.m_sequence->get_id();

                int64_t preempted_id = group.finish(beam, m_parameters);
                if (preempted_id >= 0) {
                    // remove preempted one
                    m_sequence_group->remove_sequence(preempted_id);
                }

                // mark current sequence as finished
                beam.m_sequence->set_status(SequenceStatus::FINISHED);
                // Setting length since this function is used when sequence generated tokens number reaches max_new_tokens 
                beam.m_sequence->set_finish_reason(GenerationFinishReason::LENGTH);
                // we also need to drop add ongoing / forked sequences from scheduler
                sampler_output.m_dropped_sequences.push_back(sequence_id);
            }
        }
    }
}

Sampler::GroupBeamSearcher::GroupBeamSearcher(SequenceGroup::Ptr sequence_group, Tokenizer tokenizer)
    : Sampler::Searcher(sequence_group, tokenizer),
      m_groups{m_parameters.num_beam_groups} {
    OPENVINO_ASSERT(m_sequence_group->num_running_seqs() == 1);
    assert(m_parameters.num_beams % m_parameters.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = m_parameters.num_beams / m_parameters.num_beam_groups;

    for (Group& group : m_groups) {
        group.ongoing.reserve(group_size);
        // initially we just add our "base" sequence to beams inside each group
        for (size_t i = 0; i < group_size; ++i)
            group.ongoing.push_back(Beam((*sequence_group)[0]));
        // to avoid selecting the same tokens for beams within group, let's just initialize score
        // for the front one
        group.ongoing.front().m_score = 0.0f;
    }
}


std::map<size_t, int32_t> Sampler::GroupBeamSearcher::get_beam_idxs() {
    std::map<size_t, int32_t> next_beams;

    for (Group& group : m_groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                next_beams.insert({beam.m_sequence->get_id(), beam.m_global_beam_idx});
            }
        }
    }

    return next_beams;
}

std::pair<std::map<std::string, float>, std::vector<float>> Sampler::get_structured_output_times() {
    if (m_tokenizer.m_pimpl != nullptr && m_tokenizer.m_pimpl->get_structured_output_controller()) {
        return m_tokenizer.m_pimpl->get_structured_output_controller()->get_times();
    } else {
        // If compiled without structured output support, return empty times
        return {{}, {}};
    }
}

void Sampler::clear_structured_output_compile_times() {
    if (m_tokenizer.m_pimpl != nullptr && m_tokenizer.m_pimpl->get_structured_output_controller()) {
        m_tokenizer.m_pimpl->get_structured_output_controller()->clear_compile_times();
    }
}

void Sampler::GroupBeamSearcher::select_next_tokens(const ov::Tensor& logits,
    SamplerOutput& sampler_output,
    const std::pair<size_t, std::set<std::string>>& stop_strings) {
    assert(m_parameters.num_beams % m_parameters.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = m_parameters.num_beams / m_parameters.num_beam_groups;
    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;
    next_tokens.reserve(m_parameters.num_beams);
    next_beams.reserve(m_parameters.num_beams);

    // parent sequence ID -> number of child sequences
    std::map<uint64_t, uint64_t> parent_2_num_childs_map;

    for (Group& group : m_groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                uint64_t parent_seq_id = beam.m_sequence->get_id();

                // here we need to map index of sequence in beam search group(s) and sequence group
                beam.m_global_beam_idx = [this] (uint64_t seq_id) -> size_t {
                    std::vector<Sequence::Ptr> running_seqs = m_sequence_group->get_running_sequences();
                    for (size_t seq_global_index = 0; seq_global_index < running_seqs.size(); ++seq_global_index) {
                        if (seq_id == running_seqs[seq_global_index]->get_id())
                            return seq_global_index;
                    }
                    OPENVINO_THROW("Internal error in beam search: should not be here");
                } (parent_seq_id);

                // zero out all parent forks counts
                parent_2_num_childs_map[parent_seq_id] = 0;
            }
        }
    }

    auto try_to_finish_candidate = [&] (Group& group, Beam& candidate, bool include_candidate_token = true) -> void {
        uint64_t seq_id = candidate.m_sequence->get_id();
        // try to finish candidate
        int64_t preempted_seq_id = group.finish(candidate, m_parameters);

        // if candidate has lower score than others finished
        if (preempted_seq_id == seq_id) {
            // do nothing and just ignore current finished candidate
        } else {
            if (preempted_seq_id >= 0) {
                m_sequence_group->remove_sequence(preempted_seq_id);
            }

            // need to insert candidate to a sequence group
            Sequence::Ptr forked_sequence = m_sequence_group->fork_sequence(candidate.m_sequence);
            // and finish immediately
            forked_sequence->set_status(SequenceStatus::FINISHED);
            // Setting stop since this function is used when sequence generated eos token
            forked_sequence->set_finish_reason(GenerationFinishReason::STOP);

            // TODO: make it simpler
            // currently, we finish sequence and then fork it in current code
            {
                for (size_t i = 0; i < group.min_heap.size(); ++i) {
                    if (group.min_heap[i].m_sequence->get_id() == seq_id) {
                        group.min_heap[i].m_sequence = forked_sequence;
                        break;
                    }
                }
            }

            // append token from candidate to actual sequence
            if (include_candidate_token)
                forked_sequence->append_token(candidate.m_token_id, candidate.m_log_prob);
        }
    };

    // group ID => child beams
    std::map<int, std::vector<Beam>> child_beams_per_group;

    for (size_t group_id = 0; group_id < m_groups.size(); ++group_id) {
        Group & group = m_groups[group_id];
        if (group.done)
            continue;

        std::vector<Beam> candidates;
        candidates.reserve(group_size * 2 * group_size);
        for (const Beam& beam : group.ongoing) {
            std::vector<Token> tokens = log_softmax(logits, beam.m_global_beam_idx);

            // apply diversity penalty
            for (auto prev_group_id = 0; prev_group_id < group_id; ++prev_group_id) {
                for (const Beam& prev_beam : child_beams_per_group[prev_group_id]) {
                    tokens[prev_beam.m_token_id].m_log_prob -= m_parameters.diversity_penalty;
                }
            }

            // apply n_gramm
            std::vector<int64_t> full_text{m_sequence_group->get_prompt_ids()};
            full_text.insert(full_text.end(), beam.m_sequence->get_generated_ids().begin(), beam.m_sequence->get_generated_ids().end());
            if (full_text.size() > 1 && full_text.size() >= m_parameters.no_repeat_ngram_size) {
                auto tail_start = full_text.end() - ptrdiff_t(m_parameters.no_repeat_ngram_size) + 1;
                for (int64_t banned_token : kmp_search(full_text, {tail_start, full_text.end()})) {
                    tokens[banned_token].m_log_prob = -std::numeric_limits<float>::infinity();
                }
            }

            // sort tokens
            std::sort(tokens.begin(), tokens.end(), [](Token left, Token right) {
                return left.m_log_prob > right.m_log_prob;  // Most probable tokens in front
            });

            size_t add_count = 0;
            for (Token token : tokens) {
                Beam new_candidate = beam;
                new_candidate.m_score += new_candidate.m_log_prob = token.m_log_prob;
                new_candidate.m_token_id = token.m_index;

                // TODO: fix it
                // and ensure cumulative_log prob is used
                if (/* m_parameters.early_finish(new_candidate) */ false) {
                    try_to_finish_candidate(group, new_candidate);
                } else {
                    candidates.push_back(new_candidate);
                    if (++add_count == 2 * group_size) {
                        break;
                    }
                }
            }
        }

        // Sample 2 * group_size highest score tokens to get at least 1 non EOS token per beam
        OPENVINO_ASSERT(candidates.size() >= 2 * group_size, "No beams left to search");

        auto to_sort = candidates.begin() + ptrdiff_t(2 * group_size);
        std::partial_sort(candidates.begin(), to_sort, candidates.end(), greater);

        for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
            Beam & candidate = candidates[cand_idx];
            if (is_stop_token_id_hit(candidate.m_token_id, m_sequence_group->get_sampling_parameters().stop_token_ids)) {
                // If beam_token does not belong to top num_beams tokens, it should not be added
                if (cand_idx >= group_size)
                    continue;

                // try to finish candidate
                try_to_finish_candidate(group, candidate);
                continue;
            }

            if (!m_parameters.stop_strings.empty()) {
                // We need to include candidate token to already generated tokens to check if stop string has been generated
                // There's probably a better way to do that, than copying whole vector...
                std::vector<int64_t> token_ids = candidate.m_sequence->get_generated_ids();
                token_ids.push_back(candidate.m_token_id);
                auto match_result = match_stop_string(m_tokenizer, token_ids, stop_strings, m_parameters.include_stop_str_in_output);
                if (match_result.is_matched) {
                    // If beam_token does not belong to top num_beams tokens, it should not be added
                    if (cand_idx >= group_size)
                        continue;

                    // remove tokens that match stop_string from output (last token is not included in candidate.m_sequence at this point)
                    candidate.m_sequence->remove_last_tokens(match_result.to_remove);

                    // try to finish candidate
                    try_to_finish_candidate(group, candidate);
                    continue;
                }
            }

            parent_2_num_childs_map[candidate.m_sequence->get_id()] += 1;
            child_beams_per_group[group_id].push_back(candidate);

            // if num children are enough
            if (child_beams_per_group[group_id].size() == group_size) {
                break;
            }
        }

        // check whether group has finished
        group.is_done();

        // group cannot continue if there are no valid child beams
        if (child_beams_per_group[group_id].size() == 0) {
            group.done = true;
        }

        if (group.done) {
            // group has finished, group all running sequences
            for (const Beam& beam : group.ongoing) {
                uint64_t seq_id = beam.m_sequence->get_id();
                m_sequence_group->remove_sequence(seq_id);
                sampler_output.m_dropped_sequences.push_back(seq_id);
            }
            group.ongoing.clear();
        }
    }

    // fork child sequences for non-finished groups

    for (size_t group_id = 0; group_id < m_groups.size(); ++group_id) {
        Group & group = m_groups[group_id];

        if (!group.done) {
            for (Beam& child_beam : child_beams_per_group[group_id]) {
                uint64_t parent_sequence_id = child_beam.m_sequence->get_id();
                uint64_t& num_childs = parent_2_num_childs_map[parent_sequence_id];

                // if current beam is forked multiple times
                if (num_childs > 1) {
                    child_beam.m_sequence = m_sequence_group->fork_sequence(child_beam.m_sequence);
                    child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);

                    // reduce forks count, since fork already happened and next loop iteration
                    // will go by the second branch (num_childs == 1)
                    --num_childs;

                    // fill out sampler output
                    sampler_output.m_forked_sequences[parent_sequence_id].push_back(child_beam.m_sequence->get_id());
                } else if (num_childs == 1) {
                    // keep current sequence going and add a new token
                    child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);
                }
            }

            // drop beams which are not forked by current group
            for (const Beam& beam : group.ongoing) {
                size_t num_childs = parent_2_num_childs_map[beam.m_sequence->get_id()];
                if (num_childs == 0) {
                    // drop sequence as not forked
                    sampler_output.m_dropped_sequences.push_back(beam.m_sequence->get_id());
                    m_sequence_group->remove_sequence(beam.m_sequence->get_id());
                }
            }

            // child become parents
            group.ongoing = child_beams_per_group[group_id];
        }
    }
}
/* tree generator logic */

// ------------------------------------------------------------------
// EagleCandidateGraph
// ------------------------------------------------------------------

EagleCandidateGraph::EagleCandidateGraph(int64_t root_token_id, float root_score, int max_tokens, int max_depth)
    : m_max_candidate_nodes(max_tokens),
      m_max_depth(max_depth) {
    InternalNode root;
    root.data = {0u, root_token_id, root_score, 0};
    root.parent_id = 0u;
    root.ancestor_ids = {0u};  // root is its own ancestor (self-inclusive convention)
    m_nodes[0u] = std::move(root);
}

uint64_t EagleCandidateGraph::add_node(int64_t token_id, float score, uint64_t parent_node_id) {
    const auto parent_it = m_nodes.find(parent_node_id);
    OPENVINO_ASSERT(parent_it != m_nodes.end(), "EagleCandidateGraph: parent node not found");
    const InternalNode& parent = parent_it->second;
    OPENVINO_ASSERT(parent.data.tree_layer < m_max_depth,
                    "EagleCandidateGraph::add_node: parent node is at max_depth (",
                    m_max_depth,
                    "), cannot add child");

    // Snapshot parent fields before inserting into m_nodes: operator[] may rehash and invalidate refs.
    const int parent_layer = parent.data.tree_layer;
    std::unordered_set<uint64_t> inherited_ancestors = parent.ancestor_ids;

    const uint64_t new_id = m_next_node_id++;
    InternalNode new_node;
    new_node.data = {new_id, token_id, score, parent_layer + 1};
    new_node.parent_id = parent_node_id;
    new_node.ancestor_ids = std::move(inherited_ancestors);
    new_node.ancestor_ids.insert(new_id);  // add self
    m_nodes[new_id] = std::move(new_node);
    m_nodes[parent_node_id].child_ids.push_back(new_id);
    return new_id;
}

std::vector<EagleCandidateGraph::Node> EagleCandidateGraph::select_candidate_nodes() const {
    if (m_max_candidate_nodes <= 0)
        return {m_nodes.at(0u).data};

    // Strict total order for deterministic top-k selection.
    // Better node: higher score, then lower tree layer, then lower id.
    const auto is_better = [](const Node& a, const Node& b) {
        if (a.score != b.score)
            return a.score > b.score;
        if (a.tree_layer != b.tree_layer)
            return a.tree_layer < b.tree_layer;
        return a.id < b.id;
    };

    // Min-heap over kept nodes where top() is the current worst node by is_better.
    std::priority_queue<Node, std::vector<Node>, decltype(is_better)> min_heap(is_better);

    for (const auto& [id, node] : m_nodes) {
        if (id == 0u)
            continue;
        if (min_heap.size() < static_cast<size_t>(m_max_candidate_nodes)) {
            min_heap.push(node.data);
        } else if (is_better(node.data, min_heap.top())) {
            min_heap.pop();
            min_heap.push(node.data);
        }
    }

    std::vector<Node> result;
    result.reserve(min_heap.size() + 1);
    result.push_back(m_nodes.at(0u).data);  // root is always included
    while (!min_heap.empty()) {
        result.push_back(min_heap.top());
        min_heap.pop();
    }

    // Sort by tree layer; break ties by score (descending) then id for full determinism.
    std::sort(result.begin(), result.end(), [](const Node& a, const Node& b) {
        if (a.tree_layer != b.tree_layer)
            return a.tree_layer < b.tree_layer;
        if (a.score != b.score)
            return a.score > b.score;
        return a.id < b.id;
    });
    return result;
}

std::vector<EagleCandidateGraph::Node> EagleCandidateGraph::get_leaf_nodes(const std::vector<Node>& selected) const {
    std::unordered_set<uint64_t> selected_ids;
    selected_ids.reserve(selected.size());
    for (const Node& n : selected)
        selected_ids.insert(n.id);

    std::vector<Node> leaves;
    for (const Node& n : selected) {
        const InternalNode& internal = m_nodes.at(n.id);
        const bool has_selected_child =
            std::any_of(internal.child_ids.begin(), internal.child_ids.end(), [&selected_ids](uint64_t child_id) {
                return selected_ids.count(child_id) > 0;
            });
        if (!has_selected_child)
            leaves.push_back(n);
    }
    return leaves;
}

std::vector<uint64_t> EagleCandidateGraph::get_path_to_node(uint64_t node_id) const {
    OPENVINO_ASSERT(m_nodes.count(node_id) > 0, "EagleCandidateGraph::get_path_to_node: node_id not found in graph");
    std::vector<uint64_t> path;
    auto it = m_nodes.find(node_id);
    while (it != m_nodes.end()) {
        path.push_back(it->second.data.id);
        if (it->second.data.id == 0u)
            break;  // reached root
        it = m_nodes.find(it->second.parent_id);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

bool EagleCandidateGraph::can_expand(uint64_t node_id) const {
    const auto it = m_nodes.find(node_id);
    OPENVINO_ASSERT(it != m_nodes.end(), "EagleCandidateGraph::can_expand: node_id not found");
    return it->second.data.tree_layer < m_max_depth;
}

bool EagleCandidateGraph::is_ancestor(uint64_t ancestor_id, uint64_t node_id) const {
    const auto it = m_nodes.find(node_id);
    if (it == m_nodes.end())
        return false;
    return it->second.ancestor_ids.count(ancestor_id) > 0;
}

// ------------------------------------------------------------------
// TreeSearcher
// ------------------------------------------------------------------

Sampler::TreeSearcher::TreeSearcher(SequenceGroup::Ptr sequence_group, ov::Tensor d2t)
    : Sampler::Searcher(sequence_group),
      m_d2t_tensor(std::move(d2t)) {
    OPENVINO_ASSERT(m_sequence_group->num_running_seqs() == 1,
                    "TreeSearcher must be initialized with exactly one running sequence");
    // m_original_grouped_id is set authoritatively in tree_reset() at the start of each draft round.
}

void Sampler::TreeSearcher::tree_reset() {
    const size_t total_tokens = m_parameters.eagle_tree_params.total_tokens;
    OPENVINO_ASSERT(total_tokens > 1,
                    "eagle_tree_params.total_tokens must be greater than 1, got ", total_tokens);
    const int max_non_root_candidate_nodes = static_cast<int>(total_tokens - 1);
    m_candidate_graph.emplace(-1, 0.0f, max_non_root_candidate_nodes, m_parameters.eagle_tree_params.tree_depth);

    const std::vector<Sequence::Ptr> running = m_sequence_group->get_running_sequences();
    OPENVINO_ASSERT(!running.empty(), "tree_reset: sequence group has no running sequences");
    Sequence::Ptr root_seq = running[0];
    m_original_grouped_id = root_seq->get_grouped_id();
    m_frontier = {{0u, std::move(root_seq), 0.0f}};  // root: node_id=0, cumulative score=0
}

auto Sampler::TreeSearcher::build_top_k_frontier(const ov::Tensor& logits) -> std::vector<CandidateBeam> {
    OPENVINO_ASSERT(m_candidate_graph.has_value(), "build_top_k_frontier called before tree_reset()");

    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3,
                    "Logits tensor must be 3-dimensional [batch, seq_len, vocab], got rank ",
                    shape.size());
    const size_t batch = shape[0], seq_len = shape[1], vocab_size = shape[2];
    OPENVINO_ASSERT(batch > 0, "Logits tensor has zero batch size");
    OPENVINO_ASSERT(seq_len > 0, "Logits tensor has zero sequence length");
    OPENVINO_ASSERT(vocab_size > 0, "Logits tensor has zero vocabulary size");

    const size_t branching_factor = m_parameters.eagle_tree_params.branching_factor;
    OPENVINO_ASSERT(branching_factor > 0, "eagle_tree_params.branching_factor must be positive");

    // Build seq_id -> batch-index lookup for the current frontier.
    const std::vector<Sequence::Ptr> running_seqs = m_sequence_group->get_running_sequences();
    std::unordered_map<uint64_t, size_t> seq_id_to_batch_idx;
    seq_id_to_batch_idx.reserve(running_seqs.size());
    for (size_t idx = 0; idx < running_seqs.size(); ++idx)
        seq_id_to_batch_idx[running_seqs[idx]->get_id()] = idx;

    const int64_t* d2t = m_d2t_tensor ? m_d2t_tensor.data<const int64_t>() : nullptr;
    const float* logits_data = logits.data<const float>();

    // Min-heap comparator: top() holds the worst (lowest logit) of the kept candidates.
    using IndexScore = std::pair<float, size_t>;  // {logit, vocab_index}
    const auto min_cmp = [](const IndexScore& a, const IndexScore& b) {
        return a.first > b.first;
    };

    std::vector<CandidateBeam> candidates;
    candidates.reserve(branching_factor * m_frontier.size());

    // Scratch buffer reused across frontier nodes to avoid per-iteration heap-drain allocation.
    std::vector<IndexScore> top_k;
    top_k.reserve(branching_factor);

    for (const DraftBeam& draft : m_frontier) {
        if (!m_candidate_graph->can_expand(draft.node_id))
            continue;  // parent is at max_depth, skip expansion

        const auto idx_it = seq_id_to_batch_idx.find(draft.m_sequence->get_id());
        OPENVINO_ASSERT(idx_it != seq_id_to_batch_idx.end(),
                        "Internal error in tree search: frontier sequence not found in running set");
        const size_t batch_idx = idx_it->second;
        OPENVINO_ASSERT(batch_idx < batch, "Logits batch size doesn't match number of frontier sequences");

        const float* logit_row = logits_data + (batch_idx * seq_len + (seq_len - 1)) * vocab_size;

        // Pass 1: fused max-logit tracking and per-node top-branching_factor selection.
        // A single vocabulary scan both records the running max and maintains the min-heap,
        // avoiding a separate max-finding pass.
        float max_logit = -std::numeric_limits<float>::infinity();
        std::priority_queue<IndexScore, std::vector<IndexScore>, decltype(min_cmp)> min_heap(min_cmp);
        for (size_t i = 0; i < vocab_size; ++i) {
            const float logit = logit_row[i];
            if (logit > max_logit)
                max_logit = logit;
            if (min_heap.size() < branching_factor)
                min_heap.emplace(logit, i);
            else if (logit > min_heap.top().first) {
                min_heap.pop();
                min_heap.emplace(logit, i);
            }
        }

        // Pass 2: compute log-sum-exp for softmax normalization.
        float sum_exp = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i)
            sum_exp += std::exp(logit_row[i] - max_logit);
        const float log_sum = std::log(sum_exp);

        // Drain heap into scratch buffer (ascending logit order), emit candidates in descending order.
        top_k.clear();
        while (!min_heap.empty()) {
            top_k.push_back(min_heap.top());
            min_heap.pop();
        }
        for (auto it = top_k.rbegin(); it != top_k.rend(); ++it) {
            const float log_prob = it->first - max_logit - log_sum;
            const float cum_score = draft.score + log_prob;
            const int64_t token_id = static_cast<int64_t>(it->second) + (d2t ? d2t[it->second] : 0);
            const uint64_t new_node_id = m_candidate_graph->add_node(token_id, cum_score, draft.node_id);
            candidates.push_back({new_node_id, draft.m_sequence, token_id, log_prob, cum_score});
        }
    }

    // Global stage: from the merged per-node top-branching_factor sets,
    // keep only the globally best branching_factor candidates.
    const size_t kept = std::min(branching_factor, candidates.size());
    std::partial_sort(candidates.begin(),
                      candidates.begin() + kept,
                      candidates.end(),
                      [](const CandidateBeam& a, const CandidateBeam& b) {
                          if (a.score != b.score)
                              return a.score > b.score;
                          return a.node_id < b.node_id;
                      });
    candidates.resize(kept);

    return candidates;
}

void Sampler::TreeSearcher::advance_draft_layer(const std::vector<CandidateBeam>& candidates,
                                                SamplerOutput& sampler_output) {
    OPENVINO_ASSERT(!m_frontier.empty(), "advance_draft_layer called with an empty frontier");

    // Count selected children per frontier sequence to decide fork vs. extend-in-place.
    std::unordered_map<uint64_t, uint64_t> children_per_parent;
    children_per_parent.reserve(m_frontier.size());
    for (const DraftBeam& beam : m_frontier)
        children_per_parent[beam.m_sequence->get_id()] = 0;
    for (const CandidateBeam& cand : candidates) {
        OPENVINO_ASSERT(cand.parent_sequence != nullptr, "advance_draft_layer: candidate has a null parent sequence");
        OPENVINO_ASSERT(children_per_parent.count(cand.parent_sequence->get_id()) > 0,
                        "advance_draft_layer: candidate references a parent sequence not in the current frontier");
        children_per_parent[cand.parent_sequence->get_id()] += 1;
    }

    // Materialize each candidate: fork the parent when it has multiple children,
    // extend in-place for its last child to avoid an unnecessary allocation.
    std::vector<DraftBeam> next_frontier;
    next_frontier.reserve(candidates.size());
    for (const CandidateBeam& cand : candidates) {
        const uint64_t parent_seq_id = cand.parent_sequence->get_id();
        uint64_t& remaining = children_per_parent[parent_seq_id];

        Sequence::Ptr seq;
        if (remaining > 1) {
            seq = m_sequence_group->fork_sequence(cand.parent_sequence);
            sampler_output.m_forked_sequences[parent_seq_id].push_back(seq->get_id());
            --remaining;
        } else {
            seq = cand.parent_sequence;  // last (or only) child: reuse in-place
        }
        seq->append_token(cand.token_id, cand.log_prob);
        next_frontier.push_back({cand.node_id, std::move(seq), cand.score});
    }

    // Retire frontier sequences with no selected children (remaining == 0).
    for (const DraftBeam& beam : m_frontier) {
        if (children_per_parent[beam.m_sequence->get_id()] != 0)
            continue;
        if (beam.m_sequence->get_grouped_id() == m_original_grouped_id) {
            // Keep as a suspended anchor for the verification phase.
            beam.m_sequence->set_status(SequenceStatus::SUSPENDED);
        } else {
            sampler_output.m_dropped_sequences.push_back(beam.m_sequence->get_id());
            m_sequence_group->remove_sequence(beam.m_sequence->get_id());
        }
    }

    m_frontier = std::move(next_frontier);
}

void Sampler::TreeSearcher::finalize_tree(SamplerOutput& sampler_output) {
    OPENVINO_ASSERT(m_candidate_graph.has_value(), "finalize_tree called before tree_reset()");

    const std::vector<EagleCandidateGraph::Node> final_nodes = m_candidate_graph->select_candidate_nodes();

    // Position (depth) of each node in final_nodes, forwarded to the verifier.
    std::vector<int64_t> position_ids;
    position_ids.reserve(final_nodes.size());
    for (const EagleCandidateGraph::Node& n : final_nodes)
        position_ids.push_back(static_cast<int64_t>(n.tree_layer));

    // Map node IDs to their indices in final_nodes for path index remapping below.
    std::unordered_map<uint64_t, size_t> node_to_idx;
    node_to_idx.reserve(final_nodes.size());
    for (size_t i = 0; i < final_nodes.size(); ++i)
        node_to_idx[final_nodes[i].id] = i;

    // retrieve_indices: one root-to-leaf path (as final_nodes indices) per leaf.
    // The verifier uses this to extract each candidate token sequence.
    const std::vector<EagleCandidateGraph::Node> leaf_nodes = m_candidate_graph->get_leaf_nodes(final_nodes);
    OPENVINO_ASSERT(!leaf_nodes.empty(),
                    "finalize_tree: no leaf nodes among selected candidates; "
                    "check that tree parameters produce a valid tree structure");
    std::vector<std::vector<int64_t>> retrieve_indices;
    retrieve_indices.reserve(leaf_nodes.size());
    for (const EagleCandidateGraph::Node& leaf : leaf_nodes) {
        const std::vector<uint64_t> raw_path = m_candidate_graph->get_path_to_node(leaf.id);
        std::vector<int64_t> path;
        path.reserve(raw_path.size());
        for (uint64_t nid : raw_path)
            path.push_back(static_cast<int64_t>(node_to_idx.at(nid)));
        retrieve_indices.push_back(std::move(path));
    }

    // tree_mask[i][j] = 1 iff node j is an ancestor of node i (self-inclusive).
    // Enables parallel verification via masked attention over the draft tree.
    std::vector<std::vector<uint8_t>> tree_mask;
    tree_mask.reserve(final_nodes.size());
    for (const EagleCandidateGraph::Node& node : final_nodes) {
        std::vector<uint8_t> row;
        row.reserve(final_nodes.size());
        for (const EagleCandidateGraph::Node& other : final_nodes)
            row.push_back(m_candidate_graph->is_ancestor(other.id, node.id) ? 1u : 0u);
        tree_mask.push_back(std::move(row));
    }

    // Locate the main sequence that carries the verification metadata.
    Sequence::Ptr sequence = nullptr;
    for (const Sequence::Ptr& seq : m_sequence_group->get_sequences()) {
        if (seq->get_grouped_id() == m_original_grouped_id) {
            sequence = seq;
            break;
        }
    }
    OPENVINO_ASSERT(sequence != nullptr,
                    "finalize_tree: sequence with grouped_id ",
                    m_original_grouped_id,
                    " not found");

    // Roll back draft tokens, then re-apply the selected tree nodes (skipping root at index 0).
    OPENVINO_ASSERT(sequence->get_generated_len() >= m_pre_draft_generated_len,
                    "finalize_tree: generated_len (", sequence->get_generated_len(),
                    ") is less than pre-draft length (", m_pre_draft_generated_len,
                    "); sequence state is inconsistent");
    sequence->remove_last_tokens(sequence->get_generated_len() - m_pre_draft_generated_len);
    for (size_t t = 1; t < final_nodes.size(); ++t)
        sequence->append_token(final_nodes[t].token_id, 0.0f);

    // extra_processed_tokens: tree nodes beyond the deepest path (= total - depth - 1 for root).
    // These are processed simultaneously by the verifier and must be counted now.
    OPENVINO_ASSERT(final_nodes.size() > static_cast<size_t>(m_parameters.eagle_tree_params.tree_depth),
                    "finalize_tree: final_nodes.size() (",
                    final_nodes.size(),
                    ") must exceed tree_depth (",
                    m_parameters.eagle_tree_params.tree_depth,
                    ")");
    const size_t extra_processed_tokens =
        final_nodes.size() - static_cast<size_t>(m_parameters.eagle_tree_params.tree_depth) - 1;
    m_sequence_group->update_processed_tokens_num(m_sequence_group->get_num_processed_tokens() +
                                                  extra_processed_tokens);

    m_sequence_group->set_num_validated_tokens(final_nodes.size() - 1);

    sequence->set_eagle_metadata({tree_mask, retrieve_indices, position_ids});
    sequence->set_status(SequenceStatus::RUNNING);

    // Drop all forked draft sequences; only the main sequence proceeds to verification.
    for (const Sequence::Ptr& seq : m_sequence_group->get_running_sequences()) {
        if (seq->get_grouped_id() == m_original_grouped_id)
            continue;
        sampler_output.m_dropped_sequences.push_back(seq->get_id());
        m_sequence_group->remove_sequence(seq->get_id());
    }
}

void Sampler::TreeSearcher::advance_draft_step(const ov::Tensor& logits, SamplerOutput& sampler_output) {
    if (m_phase == Phase::IDLE) {
        OPENVINO_ASSERT(m_sequence_group->num_running_seqs() == 1,
                        "TreeSearcher: unexpected number of running sequences at draft start");
        m_pre_draft_generated_len = m_sequence_group->get_running_sequences()[0]->get_generated_len();
        tree_reset();
        m_phase = Phase::DRAFTING;
    }

    m_current_draft_layer++;
    const std::vector<CandidateBeam> candidates = build_top_k_frontier(logits);

    if (m_current_draft_layer < m_parameters.eagle_tree_params.tree_depth) {
        advance_draft_layer(candidates, sampler_output);
    } else {
        finalize_tree(sampler_output);
        m_phase = Phase::IDLE;
        m_current_draft_layer = 0;
    }
}

Logits Sampler::_get_logit_vector(ov::Tensor logits, size_t batch_idx, size_t token_idx) {
    ov::Shape logits_shape = logits.get_shape();
    size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2];
    OPENVINO_ASSERT(batch_idx <= batch_size);
    OPENVINO_ASSERT(token_idx < seq_len);
    size_t batch_offset = batch_idx * seq_len * vocab_size;
    size_t sequence_offset = (seq_len - token_idx - 1) * vocab_size;
    float* logits_data = logits.data<float>() + batch_offset + sequence_offset;

    return Logits{logits_data, vocab_size};
}

Token Sampler::_greedy_sample(const Logits& logits, size_t top_logprobs) const {
    // For greedy sampling we do not expect sorting or shrinking considered tokens.
    // When logprobs > 0, m_vector is initialized (penalties wrote there, m_data is pristine);
    // scan m_vector so penalty effects influence token selection.
    // Otherwise operate directly on m_data.
    size_t m = std::max(size_t(1), top_logprobs); // ensure m is at least 1
    std::vector<float> top_values(m, -std::numeric_limits<float>::infinity());
    std::vector<size_t> top_indexes(m, 0);

    const bool use_vector = logits.is_vector_initialized();
    auto value_at = [&](size_t i) { return use_vector ? logits.m_vector[i].m_log_prob : logits.m_data[i]; };
    auto index_at = [&](size_t i) -> size_t { return use_vector ? static_cast<size_t>(logits.m_vector[i].m_index) : i; };

    for (size_t i = 0; i < logits.m_size; ++i) {
        if (value_at(i) > top_values.back()) {
            top_values.back() = value_at(i);
            top_indexes.back() = index_at(i);

            for (size_t j = top_values.size() - 1; j > 0 && top_values[j] > top_values[j - 1]; --j) {
                std::swap(top_values[j], top_values[j - 1]);
                std::swap(top_indexes[j], top_indexes[j - 1]);
            }
        }
    }

    size_t max_index = top_indexes.front();
    float max_value = 0.0;

    if (top_logprobs) {
        OPENVINO_ASSERT(!std::isnan(logits.m_full_vocab_log_sum_exp),
            "Internal error: top_logprobs > 0 but m_full_vocab_log_sum_exp was not computed. "
            "FullVocabLogSumExpTransform must run before _greedy_sample when logprobs > 0.");
        // m_data is pristine (FullVocabLogSumExpTransform + CopyLogitsToVectorTransform ran first,
        // so penalties/grammar only wrote to m_vector). Return raw log-probability.
        max_value = logits.m_data[max_index] - logits.m_full_vocab_log_sum_exp;
    }

    return Token(max_value, max_index);
}

std::vector<Token> Sampler::_multinomial_sample(const Logits& logits, size_t num_tokens_per_sequence) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<Token> out_tokens;
    out_tokens.reserve(num_tokens_per_sequence);

    if (logits.m_defer_expf) {
        // --- Deferred / fused expf path ---
        // defer_expf is only set when top_k > 0, so m_vector is always initialized here.
        OPENVINO_ASSERT(logits.is_vector_initialized(),
            "Internal error: m_defer_expf=true but m_vector not initialized. "
            "defer_expf requires top_k > 0 which always populates m_vector via TopKFilter.");
        // m_vector holds K logits in arbitrary heap order — _multinomial_sample does its own max scan.
        float max_val = logits.m_vector[0].m_log_prob;
        for (size_t i = 1; i < logits.m_size; ++i)
            if (logits.m_vector[i].m_log_prob > max_val)
                max_val = logits.m_vector[i].m_log_prob;
        // Precompute weights once to avoid recomputing expf on every draw.
        std::vector<float> weights(logits.m_size);
        float sum_cum = 0.0f;
        for (size_t i = 0; i < logits.m_size; ++i) {
            weights[i] = expf(logits.m_vector[i].m_log_prob - max_val);
            sum_cum += weights[i];
        }
        // Defensive fallback: should not occur in practice (at least one non-masked token is
        // always guaranteed), but if all logits happen to be -inf/NaN, sample uniformly over
        // the K candidates so generation can continue rather than aborting.
        const bool fallback_uniform = !(sum_cum > 0.0f && std::isfinite(sum_cum));
        if (fallback_uniform) {
            std::fill(weights.begin(), weights.end(), 1.0f);
            sum_cum = static_cast<float>(logits.m_size);
        }
        // log_sum_cum = log(Σ exp(v[i])) = log(sum_cum) + max_val (log-sum-exp identity).
        // In the fallback case max_val may be -inf, so use 0 to keep log_sum_cum finite.
        const float log_sum_cum = logf(sum_cum) + (fallback_uniform ? 0.0f : max_val);

        for (size_t token_idx = 0; token_idx < num_tokens_per_sequence; ++token_idx) {
            const float r = sum_cum * u(rng_engine);
            float sum_run = 0.0f;
            size_t sampled_idx = logits.m_size - 1;
            for (size_t i = 0; i < logits.m_size; ++i) {
                sum_run += weights[i];
                if (sum_run > r) { sampled_idx = i; break; }
            }
            // When m_full_vocab_log_sum_exp is set (logprobs > 0), m_data holds the
            // original raw model logits (grammar/penalties only wrote to m_vector).
            // Return raw log-probability: log p_i = raw_logit_i − log(Σ exp(raw_logit_j)).
            const float log_prob = fallback_uniform
                ? std::log(1.0f / static_cast<float>(logits.m_size))
                : (!std::isnan(logits.m_full_vocab_log_sum_exp)
                    ? logits.m_data[logits.m_vector[sampled_idx].m_index] - logits.m_full_vocab_log_sum_exp
                    : logits.m_vector[sampled_idx].m_log_prob - log_sum_cum);
            out_tokens.emplace_back(log_prob, logits.m_vector[sampled_idx].m_index);
        }
    } else {
        // --- Normalised probability path (top_p active, deferring not possible) ---
        // Pre-compute total weight once: TopP truncates without renormalizing, so Σ probs may be < 1.
        // Scaling r by total_weight ensures uniform sampling within the actual probability mass (Fix 1).
        float total_weight = 0.0f;
        if (logits.is_vector_initialized()) {
            for (size_t i = 0; i < logits.m_size; ++i)
                total_weight += logits.m_vector[i].m_log_prob;
        } else {
            for (size_t i = 0; i < logits.m_size; ++i)
                total_weight += logits.m_data[i];
        }
        // Defensive fallback: should not occur in practice (at least one non-masked token is
        // always guaranteed), but recover gracefully by sampling uniformly if all probabilities
        // are zero or NaN rather than aborting generation.
        if (!(total_weight > 0.0f && std::isfinite(total_weight))) {
            std::uniform_int_distribution<size_t> uniform_idx(0, logits.m_size - 1);
            const float uniform_log_prob = std::log(1.0f / logits.m_size);
            for (size_t token_idx = 0; token_idx < num_tokens_per_sequence; ++token_idx) {
                const size_t sampled_idx = uniform_idx(rng_engine);
                if (logits.is_vector_initialized()) {
                    auto logit = logits.m_vector[sampled_idx];
                    logit.m_log_prob = uniform_log_prob;
                    out_tokens.push_back(logit);
                } else {
                    out_tokens.emplace_back(uniform_log_prob, sampled_idx);
                }
            }
        } else {
        // Sampling distributes r = total_weight * U(0,1), drawing from the conditional
        // distribution p_i / total_weight.  The stored log-prob must reflect the same
        // distribution: log(p_i / total_weight) = log(p_i) - log(total_weight).
        // When total_weight == 1 (no TopP truncation) log_total_weight == 0, no-op.
        const float log_total_weight = std::log(total_weight);
        for (size_t token_idx = 0; token_idx < num_tokens_per_sequence; ++token_idx) {
            const float r = total_weight * u(rng_engine);
            float sum_cum = 0.0f;
            if (logits.is_vector_initialized()) {
                size_t sampled_idx = logits.m_size - 1;  // fallback for floating-point rounding
                for (size_t i = 0; i < logits.m_size; ++i) {
                    sum_cum += logits.m_vector[i].m_log_prob;
                    if (sum_cum > r) { sampled_idx = i; break; }
                }
                auto logit = logits.m_vector[sampled_idx];
                // When m_full_vocab_log_sum_exp is set (logprobs > 0), m_data holds the
                // original raw model logits (grammar/penalties only wrote to m_vector).
                // Return raw log-probability: log p_i = raw_logit_i − log(Σ exp(raw_logit_j)).
                logit.m_log_prob = !std::isnan(logits.m_full_vocab_log_sum_exp)
                    ? logits.m_data[logit.m_index] - logits.m_full_vocab_log_sum_exp
                    : std::log(logit.m_log_prob) - log_total_weight;
                out_tokens.push_back(logit);
            } else {
                size_t sampled_idx = logits.m_size - 1;
                for (size_t i = 0; i < logits.m_size; ++i) {
                    sum_cum += logits.m_data[i];
                    if (sum_cum > r) { sampled_idx = i; break; }
                }
                out_tokens.emplace_back(std::log(logits.m_data[sampled_idx]) - log_total_weight, sampled_idx);
            }
        }
        }
    }
    return out_tokens;
}

std::vector<int64_t> Sampler::_try_finish_generation(SequenceGroup::Ptr & sequence_group) {
    const auto& sampling_params = sequence_group->get_sampling_parameters();
    std::vector<int64_t> dropped_seq_ids;
    for (auto& running_sequence : sequence_group->get_running_sequences()) {
        const auto generated_len = running_sequence->get_generated_len();
        if (sequence_group->get_max_new_tokens() <= generated_len || 
            is_stop_token_id_hit(running_sequence->get_generated_ids().back(), sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
            // stop sequence by max_new_tokens or stop token (eos included)
            running_sequence->set_status(SequenceStatus::FINISHED);

            if (is_stop_token_id_hit(running_sequence->get_generated_ids().back(), sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
                running_sequence->set_finish_reason(GenerationFinishReason::STOP);
            } else if (sequence_group->get_max_new_tokens() == generated_len) {
                running_sequence->set_finish_reason(GenerationFinishReason::LENGTH);
            }

            dropped_seq_ids.push_back(running_sequence->get_id());
            continue;
        }

        if (!sampling_params.stop_strings.empty()) {
            auto& stop_strings = m_stop_strings.at(sequence_group->get_request_id());
            auto match_result = match_stop_string(m_tokenizer, running_sequence->get_generated_ids(), stop_strings,
                                                  sampling_params.include_stop_str_in_output, sequence_group->get_num_tokens_to_validate());
            if (match_result.is_matched) {
                running_sequence->remove_last_tokens(match_result.to_remove);

                running_sequence->set_status(SequenceStatus::FINISHED);
                running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                dropped_seq_ids.push_back(running_sequence->get_id());
            }
        }
    }
    return dropped_seq_ids;
}

void register_new_token(const Token& sampled_token,
                        Sequence::Ptr running_sequence,
                        LogitProcessor& logit_processor,
                        bool is_extend_sequence,
                        bool is_validation_mode_enabled) {
    logit_processor.register_new_generated_token(sampled_token.m_index);
    if (is_extend_sequence) {
        running_sequence->append_token(sampled_token.m_index, sampled_token.m_log_prob);
    }
    if (!is_validation_mode_enabled &&
        logit_processor.get_assistant_confidence_threshold() > 0 &&
        (std::fabs(std::exp(sampled_token.m_log_prob)) < logit_processor.get_assistant_confidence_threshold() || sampled_token.m_log_prob == 0)) {
        auto sequence_group = running_sequence->get_sequence_group_ptr();
        sequence_group->pause_generation(true);
    }
};

std::map<size_t, int32_t> Sampler::get_beam_idxs(SequenceGroup::CPtr sequence_group) {
    size_t request_id = sequence_group->get_request_id();
    auto beam_searcher = m_beam_search_info.find(request_id);
    if (beam_searcher == m_beam_search_info.end()) {
        std::map<size_t, int32_t> beams;
        for (auto& seq : sequence_group->get_running_sequences())
            beams.insert({seq->get_id(), 0});
        return beams;
    }

    return beam_searcher->second.get_beam_idxs();
}

std::list<uint64_t>
create_n_forked_sequences(SequenceGroup::Ptr sequence_group,
                          LogitProcessor& logit_processor,
                          const std::vector<Token>& sampled_tokens) {
    const auto& running_sequences = sequence_group->get_running_sequences();
    OPENVINO_ASSERT(running_sequences.size() == 1);
    Sequence::Ptr sequence_to_fork = running_sequences[0];
    if (sequence_to_fork->get_generated_len() > 0) {
        logit_processor.update_generated_len(0);
        sequence_to_fork->remove_last_tokens(sequence_to_fork->get_generated_len());
    }
    std::list<uint64_t> forked_seq_ids;
    for (size_t i = 1; i < sampled_tokens.size(); ++i) {
        const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
        const auto forked_seq_id = forked_sequence->get_id();
        forked_seq_ids.push_back(forked_seq_id);
        register_new_token(sampled_tokens[i], forked_sequence, logit_processor, true, false);
    }
    return forked_seq_ids;
}

void
stop_sample_tokens(Sequence::Ptr running_sequence,
                   size_t token_idx,
                   size_t max_gen_len,
                   size_t& max_removed_tokens_per_request) {
    running_sequence->remove_last_tokens(token_idx);
    max_removed_tokens_per_request = std::max(max_removed_tokens_per_request, token_idx);
}

void
align_all_sequence_len(SequenceGroup::Ptr& sequence_group,
                       size_t min_generated_tokens,
                       LogitProcessor& logit_processor) {
    for (auto& sequence : sequence_group->get_running_sequences()) {
        const auto& generated_token_ids = sequence->get_generated_ids();
        auto generated_len = sequence->get_generated_len();
        if (generated_len > min_generated_tokens) {
            auto removed_token_cnt = generated_len - min_generated_tokens;
            for (size_t i = min_generated_tokens + 1; i < generated_len; ++i) {
                logit_processor.decrease_generated_token_occurance(generated_token_ids[i]);
            }
            sequence->remove_last_tokens(removed_token_cnt);
        }
    }
    logit_processor.update_generated_len(min_generated_tokens);
}

// Verify the draft tree against target-model logits: greedily accept the longest
// matching prefix across all tree paths, then append a bonus token.
// Returns accepted_steps = number of accepted draft tokens + 1 (for the bonus token).
size_t Sampler::verify_draft_tree(Sequence::Ptr& sequence,
                                  const ov::Tensor& sequence_group_logits,
                                  LogitProcessor& logit_processor,
                                  size_t num_tokens_to_validate) {
    OPENVINO_ASSERT(num_tokens_to_validate > 0,
                    "verify_draft_tree: num_tokens_to_validate must be > 0, got ",
                    num_tokens_to_validate);

    auto& eagle_metadata = sequence->get_eagle_metadata();
    auto retrieve_indices = eagle_metadata.retrieve_indices;

    if (retrieve_indices.empty()) {
        return 0;
    }

    // Reconstruct per-path token sequences from retrieve_indices.
    // retrieve_indices[i] is a sequence of flat-tree node indices for path i;
    // candidate_token_ids[i] is the corresponding draft token sequence.
    const auto generated_len = sequence->get_generated_len();
    OPENVINO_ASSERT(generated_len >= num_tokens_to_validate + 1,
                    "verify_draft_tree: generated_len (",
                    generated_len,
                    ") must be >= num_tokens_to_validate + 1 (",
                    num_tokens_to_validate + 1,
                    ")");
    const auto& generated_ids = sequence->get_generated_ids();
    const size_t base_offset = generated_len - num_tokens_to_validate - 1;

    std::vector<std::vector<int64_t>> candidate_token_ids;
    candidate_token_ids.reserve(retrieve_indices.size());
    size_t max_path_len = 0;
    for (const auto& path : retrieve_indices) {
        max_path_len = std::max(max_path_len, path.size());
        std::vector<int64_t> candidate_tokens;
        candidate_tokens.reserve(path.size());
        for (const auto& index : path) {
            OPENVINO_ASSERT(index >= 0 && static_cast<size_t>(index) + base_offset < generated_ids.size(),
                            "verify_draft_tree: retrieve index ",
                            index,
                            " is out of bounds for generated_ids (size=",
                            generated_ids.size(),
                            ")");
            candidate_tokens.push_back(generated_ids[base_offset + index]);
        }
        candidate_token_ids.push_back(std::move(candidate_tokens));
    }

    // path_pos[i]: next unvalidated token index in path i (starts at 1 to skip the root).
    std::vector<size_t> path_pos(candidate_token_ids.size(), 1);
    size_t accepted_steps = 1;                  // 1 accounts for the bonus token always appended at the end
    size_t logit_idx = num_tokens_to_validate;  // index into _get_logit_vector (counts from seq end)
    std::optional<Token> bonus_token;

    // Reusable buffer for the candidate tokens from surviving paths at each depth.
    std::vector<int64_t> frontier_tokens;
    frontier_tokens.reserve(retrieve_indices.size());

    for (size_t step = 1;
         step < max_path_len && !candidate_token_ids.empty() && accepted_steps <= num_tokens_to_validate;
         ++step) {
        // Collect one candidate token per surviving path at this depth.
        frontier_tokens.clear();
        for (size_t i = 0; i < candidate_token_ids.size(); ++i) {
            if (path_pos[i] < candidate_token_ids[i].size()) {
                frontier_tokens.push_back(candidate_token_ids[i][path_pos[i]]);
            }
        }

        // Sample from the target model at this tree depth.
        auto logit_vector = _get_logit_vector(sequence_group_logits, 0, logit_idx);
        logit_processor.apply(logit_vector);
        Token sampled_token = _greedy_sample(logit_vector, 0);
        int64_t pred_token = sampled_token.m_index;

        bool matched = false;

        for (size_t i = 0; i < frontier_tokens.size(); ++i) {
            if (frontier_tokens[i] != pred_token) {
                continue;
            }
            matched = true;

            // Keep paths whose next token matches pred_token; drop the rest.
            // Example: candidate_token_ids = [[5109,20198],[5109,53,3394],[5109,53,1750,52],
            //                                  [5109,53,1404,220],[5109,53,1750,55,55]]
            //          retrieve_indices     = [[0,2],[0,1,3],[0,1,4,7],[0,1,5,8],[0,1,4,6,9]]
            for (size_t j = 0; j < candidate_token_ids.size();) {
                if (path_pos[j] < candidate_token_ids[j].size() && candidate_token_ids[j][path_pos[j]] == pred_token) {
                    path_pos[j]++;
                    ++j;
                } else {
                    retrieve_indices.erase(retrieve_indices.begin() + j);
                    candidate_token_ids.erase(candidate_token_ids.begin() + j);
                    path_pos.erase(path_pos.begin() + j);
                }
            }
            break;
        }

        if (!matched) {
            // Target prediction diverges from all draft paths; record as bonus token and stop.
            bonus_token = sampled_token;
            break;
        }

        OPENVINO_ASSERT(!retrieve_indices.empty(),
                        "verify_draft_tree: all paths eliminated after a successful match; "
                        "this indicates a bug in the path-pruning logic");

        const int64_t retrieve_index = retrieve_indices[0][path_pos[0] - 1];
        OPENVINO_ASSERT(retrieve_index >= 0,
                        "verify_draft_tree: retrieve_index must be non-negative, got ",
                        retrieve_index);
        const size_t retrieve_index_u = static_cast<size_t>(retrieve_index);
        OPENVINO_ASSERT(retrieve_index_u <= num_tokens_to_validate,
                        "verify_draft_tree: retrieve_index (",
                        retrieve_index_u,
                        ") must not exceed num_tokens_to_validate (",
                        num_tokens_to_validate,
                        ")");
        logit_idx = num_tokens_to_validate - retrieve_index_u;
        accepted_steps++;

        if (accepted_steps + generated_len - num_tokens_to_validate ==
            sequence->get_sequence_group_ptr()->get_max_new_tokens()) {
            break;
        }
    }

    // Retain only the accepted prefix of the surviving path.
    OPENVINO_ASSERT(!retrieve_indices.empty(), "verify_draft_tree: retrieve_indices is empty at commit stage");
    auto& accepted_path = retrieve_indices[0];
    auto& accepted_tokens = candidate_token_ids[0];
    if (accepted_path.size() > accepted_steps) {
        accepted_path.erase(accepted_path.begin() + accepted_steps, accepted_path.end());
        accepted_tokens.erase(accepted_tokens.begin() + accepted_steps, accepted_tokens.end());
    }

    // Commit: remove all draft tokens from the sequence, then re-append only accepted ones.
    sequence->remove_last_tokens(num_tokens_to_validate);
    for (size_t i = 1; i < accepted_steps; ++i) {
        sequence->append_token(accepted_tokens[i], 0.0f);
        logit_processor.register_new_generated_token(accepted_tokens[i]);
    }

    if (!bonus_token.has_value()) {
        // All draft tokens accepted; sample one extra token from the target model.
        auto logit_vector = _get_logit_vector(sequence_group_logits, 0, logit_idx);
        logit_processor.apply(logit_vector);
        bonus_token = _greedy_sample(logit_vector, 0);
    }

    sequence->append_token(bonus_token->m_index, bonus_token->m_log_prob);
    logit_processor.register_new_generated_token(bonus_token->m_index);

    // Update EAGLE metadata with the accepted path for the next draft round.
    sequence->set_eagle_metadata({{}, {}, eagle_metadata.tree_position_ids, accepted_path});
    return accepted_steps;
}

bool Sampler::validate_candidate(
    Sequence::Ptr running_sequence,
    size_t& token_idx,
    Token& sampled_token,
    bool& is_extend_sequence,
    size_t& max_removed_tokens,
    bool do_sample,
    bool has_real_probabilities) {
    OPENVINO_ASSERT(token_idx > 0);
    const auto& generated_tokens = running_sequence->get_generated_ids();
    auto it_token_id = generated_tokens.rbegin();
    std::advance(it_token_id, token_idx - 1);

    bool is_candidate_accepted = false;
    // first tokens in case of speculative decoding should be generated by main model
    if (do_sample && has_real_probabilities &&
        running_sequence->get_generated_len() != running_sequence->get_sequence_group_ptr()->get_num_tokens_to_validate()) {
        const auto& generated_log_probs = running_sequence->get_generated_log_probs();
        auto it_log_prob = generated_log_probs.rbegin();
        std::advance(it_log_prob, token_idx - 1);

        float p_i = std::exp(*it_log_prob),
                q_i = std::exp(sampled_token.m_log_prob),
                probability_ratio = p_i / q_i;
        
        auto dist = std::uniform_int_distribution<>(0, 100); // equivalent to multinomial with number of trials == 1
        float r_i = dist(rng_engine);
        r_i /= 100;
        is_candidate_accepted = r_i <= probability_ratio;
    } else {
        is_candidate_accepted = *it_token_id == sampled_token.m_index;
    }

    // to validate candidates from assisting model and remove incorrect ones from generated sequence
    if (!is_candidate_accepted) {
        // we need to make resample in speculative sampling, if candidates have real values of logits
        if (do_sample && has_real_probabilities) {
            return false;
        }
        running_sequence->remove_last_tokens(token_idx);
        max_removed_tokens = std::max(max_removed_tokens, token_idx);
        is_extend_sequence = true;
        return false;
    } else {
        sampled_token.m_index = *it_token_id;
    }

    return true;
}

float get_p_prime(Sequence::Ptr& running_sequence,
                  const Token& sampled_token,
                  size_t token_offset) {
    auto generated_log_probs = running_sequence->get_generated_log_probs();
    auto it_log_prob = generated_log_probs.rbegin();
    std::advance(it_log_prob, token_offset - 1);

    running_sequence->remove_last_tokens(token_offset);

    float cumulative_prob = 0;
    for (auto& log_prob : running_sequence->get_generated_log_probs()) {
        cumulative_prob += std::exp(log_prob);
    }

    if (cumulative_prob == 0.f) {
        return 1.f;
    }
    
    float p_n = std::exp(sampled_token.m_log_prob),
          q_n = std::exp(*it_log_prob),
          p_prime = std::max(0.f, (p_n - q_n)) / std::log(cumulative_prob);

    return p_prime;
}

std::pair<size_t, std::set<std::string>>
process_stop_strings(const std::set<std::string>& stop_strings, Tokenizer& tokenizer) {
    std::pair<size_t, std::set<std::string>> result;
    for (const auto& stop_string : stop_strings) {
        auto encoded_stop_string = encode_and_process_string(stop_string, tokenizer);
        if (result.first < encoded_stop_string.size()) {
            result.first = encoded_stop_string.size();
        }
        result.second.insert(stop_string);
    }
    return result;
}

SequenceGroupSamplingInfo Sampler::sample_from_sequence_group(SequenceGroup::Ptr sequence_group, ov::Tensor sequence_group_logits, 
                                                              LogitProcessor& logit_processor, const std::pair<size_t, std::set<std::string>>& stop_strings, 
                                                              bool is_validation_mode_enabled) {
    SequenceGroupSamplingInfo sg_sampling_info;
    // Assistant pipeline info is relevant for speculative and prompt lookup decoding
    AssistingPipelineInfo& assisting_pipeline_info = sg_sampling_info.get_assisting_pipeline_info();
    const ov::genai::GenerationConfig& sampling_params = sequence_group->get_sampling_parameters();
    const size_t output_seq_len = sequence_group->get_output_seq_len();
    // get number of tokens to be validated
    size_t num_tokens_to_process = sequence_group->get_num_tokens_to_validate();
    size_t num_generated_tokens_to_validate = num_tokens_to_process;

    if (num_tokens_to_process > output_seq_len - 1) {
        auto delta = num_tokens_to_process - (output_seq_len - 1);
        assisting_pipeline_info.updated_validation_len = std::max(assisting_pipeline_info.updated_validation_len, delta);
        num_tokens_to_process -= delta;
    }

    if (sampling_params.is_greedy_decoding() || sampling_params.is_multinomial()) {
        std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
        size_t num_running_sequences = sequence_group->num_running_seqs();
        if (sampling_params.is_greedy_decoding()) {
            OPENVINO_ASSERT(num_running_sequences == 1);
        }
        for (size_t running_sequence_id = 0; running_sequence_id < num_running_sequences; ++running_sequence_id) {
            auto& running_sequence = running_sequences[running_sequence_id];
            bool is_validation_passed = true;
            // make `num_tokens_to_process` iteration to validate a candidate generated by `draft_model` + 1 iteration to generate one more token by `main_model`
            for (size_t i = 0; i <= num_tokens_to_process; ++i) {
                if (running_sequence->has_finished())
                    break;
                sg_sampling_info.sampler_output.num_generated_tokens++;
                // calculate token offset from the end of logit
                size_t logit_token_offset = num_tokens_to_process - i;
                size_t generated_seq_token_offset = num_generated_tokens_to_validate - i;
                // max counter of needed to be sampled tokens
                OPENVINO_ASSERT(running_sequence->get_generated_len() >= generated_seq_token_offset);
                size_t generated_and_verified_len = running_sequence->get_generated_len() - generated_seq_token_offset;
                OPENVINO_ASSERT(sequence_group->get_max_new_tokens() >= generated_and_verified_len);
                size_t max_num_sampled_token = sequence_group->get_max_new_tokens() - generated_and_verified_len;
                if (max_num_sampled_token == 0) {
                    stop_sample_tokens(running_sequence, generated_seq_token_offset, max_num_sampled_token, assisting_pipeline_info.max_removed_tokens_per_request);
                    break;
                }
                // do sampling only for token validation/generation.
                // continue in case of extending draft model sequences by main model generated tokens which
                // should be taken to KV cache without validation
                if (!is_validation_mode_enabled && generated_seq_token_offset > 0) {
                    continue;
                }

                auto logit_vector = _get_logit_vector(sequence_group_logits, running_sequence_id, logit_token_offset);
                logit_processor.apply(logit_vector);
                
                Token sampled_token;
                bool is_generate_n_tokens = false;
                if (sampling_params.is_greedy_decoding()) {
                    sampled_token = { _greedy_sample(logit_vector, sampling_params.logprobs) };
                } else {
                    // is_multinomial()
                    is_generate_n_tokens = sequence_group->num_total_seqs() == 1;
                    const size_t num_tokens_per_sequence = is_generate_n_tokens ? sampling_params.num_return_sequences : 1;
                    is_generate_n_tokens &= (num_tokens_per_sequence > 1);
                    auto sampled_token_ids = _multinomial_sample(logit_vector, num_tokens_per_sequence);
                    OPENVINO_ASSERT(sampled_token_ids.size(), num_tokens_per_sequence);
                    // to create n sequence just in case of `sequence_group->num_total_seqs() == 1` and `sampling_params.num_return_sequences > 1`
                    if (is_generate_n_tokens) {
                        const auto forked_seq_ids = create_n_forked_sequences(sequence_group, logit_processor, sampled_token_ids);
                        sg_sampling_info.sampler_output.m_forked_sequences.insert({running_sequences[0]->get_id(), forked_seq_ids});
                    }
                    sampled_token = sampled_token_ids.front();
                    // make `_speculative_sampling` in case of previous token was not accepted in speculative decoding
                    if (!is_validation_passed) {
                        float p_prime = get_p_prime(running_sequence, sampled_token, generated_seq_token_offset + 1);
                        assisting_pipeline_info.max_removed_tokens_per_request = std::max(assisting_pipeline_info.max_removed_tokens_per_request, generated_seq_token_offset);
                        // update prob only in case candidate prob > sampled token prob
                        if (p_prime > 0.f) {
                            auto prob = std::exp(sampled_token.m_log_prob);
                            prob /= p_prime;
                            sampled_token.m_log_prob = std::log(prob);
                        }
                    }
                }
                if (!is_validation_mode_enabled && m_d2t_mapping) { // compute token offset for draft model in speculative sampling
                    ov::Tensor d2t_tensor = m_d2t_mapping->get_tensor_view();
                    auto d2t = d2t_tensor.data<int64_t>();
                    sampled_token.m_index = sampled_token.m_index + (d2t ? d2t[sampled_token.m_index] : 0);
                }
                // flag to add sampled token to generated sequence or extend logit processors only
                bool is_extend_sequence = logit_token_offset == 0 || is_generate_n_tokens || !is_validation_passed;
                if (is_validation_mode_enabled && !is_extend_sequence) {
                    is_validation_passed = validate_candidate(running_sequences[running_sequence_id], generated_seq_token_offset,
                                                              sampled_token, is_extend_sequence, assisting_pipeline_info.max_removed_tokens_per_request,
                                                              sampling_params.do_sample, !sampling_params.is_prompt_lookup());

                    // doing resample in case of non accepted tokens in speculative sampling
                    if (!is_validation_passed && sampling_params.do_sample && !sampling_params.is_prompt_lookup()) {
                        continue;
                    }
                    // update log prob just while validation process
                    if (!is_extend_sequence) {
                        OPENVINO_ASSERT(generated_and_verified_len < running_sequences[running_sequence_id]->get_generated_len());
                        running_sequence->update_generated_log_prob(generated_and_verified_len, sampled_token.m_log_prob);
                    }
                }
                register_new_token(sampled_token, running_sequences[running_sequence_id], logit_processor, is_extend_sequence, is_validation_mode_enabled);
                               
                // to exit from sampling in case of failed token validation
                if (!is_validation_passed) {
                    break;
                } else {
                    const auto& sampling_params = sequence_group->get_sampling_parameters();
                    if (is_stop_token_id_hit(sampled_token.m_index, sampling_params.stop_token_ids) && !sampling_params.ignore_eos) {
                        running_sequence->set_status(SequenceStatus::FINISHED);
                        running_sequence->set_finish_reason(GenerationFinishReason::STOP);
                        sg_sampling_info.sampler_output.m_dropped_sequences.push_back(running_sequence->get_id());
                    }
                }
            }
            assisting_pipeline_info.min_generated_len = std::min(assisting_pipeline_info.min_generated_len, running_sequence->get_generated_len());
        }
        align_all_sequence_len(sequence_group, assisting_pipeline_info.min_generated_len, logit_processor);
        for (const auto& dropped_seq_id : _try_finish_generation(sequence_group)) {
            sg_sampling_info.sampler_output.m_dropped_sequences.push_back(dropped_seq_id);
        }
    } else if (sampling_params.is_beam_search()) {
        uint64_t request_id = sequence_group->get_request_id();

        // create beam search info if we are on the first generate
        GroupBeamSearcher* beam_searcher;
        {
            std::lock_guard<std::mutex> lock(m_beam_search_info_mutex);
            if (m_beam_search_info.find(request_id) == m_beam_search_info.end()) {
                m_beam_search_info.emplace(request_id, GroupBeamSearcher(sequence_group, m_tokenizer));
            }
            beam_searcher = &m_beam_search_info.at(request_id);
        }

        if (!sequence_group->has_finished()) {
            sg_sampling_info.sampler_output.num_generated_tokens++;
        }

        // current algorithm already adds new tokens to running sequences and
        beam_searcher->select_next_tokens(sequence_group_logits, sg_sampling_info.sampler_output, stop_strings);

        // check max length stop criteria
        std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
        if (!sequence_group->has_finished() &&
            running_sequences[0]->get_generated_len() == sequence_group->get_max_new_tokens()) {
            // stop sequence by max_new_tokens
            beam_searcher->finalize(sg_sampling_info.sampler_output);
        }
    } else if (sampling_params.is_tree_search()) {
        uint64_t request_id = sequence_group->get_request_id();

        if (is_validation_mode_enabled && num_generated_tokens_to_validate > 0) {
            // Target-model validation pass: validate the draft tree candidates.
            std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
            OPENVINO_ASSERT(running_sequences.size() == 1);
            size_t valid_tokens = verify_draft_tree(running_sequences[0],
                                                    sequence_group_logits,
                                                    logit_processor,
                                                    num_generated_tokens_to_validate);
            assisting_pipeline_info.max_removed_tokens_per_request =
                std::max(assisting_pipeline_info.max_removed_tokens_per_request,
                         num_generated_tokens_to_validate + 1 - valid_tokens);
            for (const auto& dropped_seq_id : _try_finish_generation(sequence_group)) {
                sg_sampling_info.sampler_output.m_dropped_sequences.push_back(dropped_seq_id);
            }
            sg_sampling_info.sampler_output.num_generated_tokens += valid_tokens;
            assisting_pipeline_info.min_generated_len =
                std::min(assisting_pipeline_info.min_generated_len, running_sequences[0]->get_generated_len());
        } else {
            // Create tree searcher on the draft step for this request
            std::shared_ptr<TreeSearcher> tree_searcher;
            {
                std::lock_guard<std::mutex> lock(m_tree_search_info_mutex);
                if (m_tree_search_info.find(request_id) == m_tree_search_info.end()) {
                    m_tree_search_info.emplace(request_id,
                                               std::make_shared<TreeSearcher>(
                                                   sequence_group,
                                                   m_d2t_mapping ? m_d2t_mapping->get_tensor_view() : ov::Tensor()));
                }
                tree_searcher = m_tree_search_info.at(request_id);
            }

            if (!sequence_group->has_finished()) {
                sg_sampling_info.sampler_output.num_generated_tokens++;
            }

            tree_searcher->advance_draft_step(sequence_group_logits, sg_sampling_info.sampler_output);
        }  // end else (draft tree search)
    } else {
        OPENVINO_THROW("Unsupported sampling method");
    }
    // Notify handle after sampling is done. 
    // For non-streaming this is effective only when the generation is finished.
    OPENVINO_ASSERT(num_generated_tokens_to_validate >= assisting_pipeline_info.max_removed_tokens_per_request);
    sequence_group->notify_handle();
    return sg_sampling_info;
}

SamplerOutput Sampler::sample(const std::vector<SequenceGroup::Ptr> & sequence_groups,
                              ov::Tensor logits,
                              bool is_validation_mode_enabled) {
    const float * logits_data = logits.data<float>();
    ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    size_t vocab_size = logits_shape[2];

    SamplerOutput sampler_output;
    std::unordered_map<uint64_t, std::future<SequenceGroupSamplingInfo>> sg_sampling_future_map;
    for (size_t sequence_group_id = 0, currently_processed_tokens = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
        if (!sequence_group->is_scheduled())
            continue;

        const size_t num_running_sequences = sequence_group->num_running_seqs();
        const size_t output_seq_len = sequence_group->get_output_seq_len();
        const ov::genai::GenerationConfig& sampling_params = sequence_group->get_sampling_parameters();

        const auto request_id = sequence_group->get_request_id();
        if (!m_logit_processors.count(request_id)) {
            std::shared_ptr<StructuredOutputController> structured_output_controller = nullptr;
            if (m_tokenizer.m_pimpl != nullptr) {
                structured_output_controller = m_tokenizer.m_pimpl->get_structured_output_controller(vocab_size);
            }
            m_logit_processors.insert({request_id, LogitProcessor(sampling_params, sequence_group->get_prompt_ids(), structured_output_controller)});
        }
        if (!m_stop_strings.count(request_id)) {
            if (!sampling_params.stop_strings.empty()) {
                OPENVINO_ASSERT(m_tokenizer.m_pimpl != nullptr, "Stop strings require a valid tokenizer");
                auto processed_stop_string = process_stop_strings(sampling_params.stop_strings, m_tokenizer);
                m_stop_strings.insert({static_cast<int64_t>(request_id), processed_stop_string});
                sequence_group->set_stream_window_size(processed_stop_string.first);
            } else {
                m_stop_strings.insert({static_cast<int64_t>(request_id), {size_t(0), {}}});
            }
        }
        const auto& stop_strings = m_stop_strings.at(request_id);
        auto& logit_processor = m_logit_processors.at(request_id);
        const void * sequence_group_logits_data = logits_data + vocab_size * currently_processed_tokens;
        ov::Tensor sequence_group_logits(ov::element::f32, ov::Shape{num_running_sequences, output_seq_len, vocab_size}, (void *)sequence_group_logits_data);
        if (sequence_group->requires_sampling()) {
            // Call sample_from_sequence_group asynchronously
            sg_sampling_future_map[request_id] = m_thread_pool.submit(&Sampler::sample_from_sequence_group, this, sequence_group, sequence_group_logits,
                                                                      logit_processor, stop_strings, is_validation_mode_enabled);
        } else {
            // we are in prompt processing phase when prompt is split into chunks and processed step by step
        }
        // accumulate a number of processed tokens
        currently_processed_tokens += output_seq_len * num_running_sequences;
    }

    // Update sequence groups internal states after sampling is done
    for (auto& sequence_group : sequence_groups) {
        if (!sequence_group->is_scheduled())
            continue;
        SequenceGroupSamplingInfo sg_sampling_info;
        const auto request_id = sequence_group->get_request_id();
        if (sg_sampling_future_map.find(request_id) != sg_sampling_future_map.end()) {
            // If there is a future assigned to a sequence group we read it's result (blocking if results not available yet)
            sg_sampling_info = sg_sampling_future_map[request_id].get();
            sampler_output.num_generated_tokens += sg_sampling_info.sampler_output.num_generated_tokens;

            // Merge sampler output from sequence group to the main one
            sampler_output.m_dropped_sequences.insert(
                sampler_output.m_dropped_sequences.end(),
                sg_sampling_info.sampler_output.m_dropped_sequences.begin(),
                sg_sampling_info.sampler_output.m_dropped_sequences.end()
            );

            for (const auto& forked_seq : sg_sampling_info.sampler_output.m_forked_sequences) {
                sampler_output.m_forked_sequences[forked_seq.first].insert(
                    sampler_output.m_forked_sequences[forked_seq.first].end(),
                    forked_seq.second.begin(),
                    forked_seq.second.end()
                );
            }
        }
        // NOTE: it should be before 'get_num_scheduled_tokens' is used
        // update internal state of sequence group to reset scheduler tokens and update currently processed ones
        const AssistingPipelineInfo& assisting_pipeline_info = std::as_const(sg_sampling_info.get_assisting_pipeline_info());
        sequence_group->finish_iteration();
        // decrease sequence_group context in case of candidates generated by draft_model were not accepted by main_model
        if (assisting_pipeline_info.max_removed_tokens_per_request) {
            auto min_processed_tokens = sequence_group->get_prompt_len() + assisting_pipeline_info.min_generated_len - 1;
            sequence_group->update_processed_tokens_num(min_processed_tokens);
            auto& logit_processor = get_logit_processor(sequence_group->get_request_id());
            logit_processor.update_generated_len(min_processed_tokens);
        }
        if (assisting_pipeline_info.updated_validation_len) {
            sequence_group->set_num_validated_tokens(assisting_pipeline_info.updated_validation_len);
        }
    }
    return sampler_output;
}

LogitProcessor& Sampler::get_logit_processor(uint64_t request_id) {
    OPENVINO_ASSERT(m_logit_processors.count(request_id));
    return m_logit_processors.at(request_id);
}


void Sampler::create_logit_processor(uint64_t request_id, const GenerationConfig& sampling_params, const TokenIds& prompt) {
    std::shared_ptr<StructuredOutputController> structured_output_controller = nullptr;
    if (m_tokenizer.m_pimpl != nullptr) {
        structured_output_controller = m_tokenizer.m_pimpl->get_structured_output_controller();
    }
    m_logit_processors.insert({request_id, LogitProcessor(sampling_params, prompt, structured_output_controller)});
}

void Sampler::clear_request_info(uint64_t request_id) {
    {
        std::lock_guard<std::mutex> lock(m_beam_search_info_mutex);
        m_beam_search_info.erase(request_id);
    }
    m_logit_processors.erase(request_id);
    m_stop_strings.erase(request_id);
    {
        std::lock_guard<std::mutex> lock(m_tree_search_info_mutex);
        m_tree_search_info.erase(request_id);
    }
}

int64_t Sampler::GroupBeamSearcher::Group::finish(Beam beam, const ov::genai::GenerationConfig& sampling_params) {
    int64_t preeempted_sequence_id = -1;
    float generated_len = beam.get_generated_len() + (is_stop_token_id_hit(beam.m_token_id, sampling_params.stop_token_ids) ? 1 : 0); // HF counts EOS token in generation length
    beam.m_score /= std::pow(generated_len, sampling_params.length_penalty);

    min_heap.push_back(beam);
    std::push_heap(min_heap.begin(), min_heap.end(), greater);
    assert(sampling_params.num_beams % sampling_params.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = sampling_params.num_beams / sampling_params.num_beam_groups;
    if (min_heap.size() > group_size) {
        std::pop_heap(min_heap.begin(), min_heap.end(), greater);
        preeempted_sequence_id = min_heap.back().m_sequence->get_id();
        min_heap.pop_back();
    }

    return preeempted_sequence_id;
}

void Sampler::GroupBeamSearcher::Group::is_done() {
    const auto sequence_group = ongoing.front().m_sequence->get_sequence_group_ptr();
    const ov::genai::GenerationConfig& sampling_params = sequence_group->get_sampling_parameters();

    assert(sampling_params.num_beams % sampling_params.num_beam_groups == 0 &&
        "number of beams should be divisible by number of groups");
    size_t group_size = sampling_params.num_beams / sampling_params.num_beam_groups;
    if (min_heap.size() < group_size)
        return;

    const Beam& best_running_sequence = ongoing.front(), & worst_finished_sequence = min_heap.front();
    size_t cur_len = best_running_sequence.m_sequence->get_generated_len();
    float best_sum_logprobs = best_running_sequence.m_score;
    float worst_score = worst_finished_sequence.m_score;
    switch (sampling_params.stop_criteria) {
    case ov::genai::StopCriteria::EARLY:
        done = true;
        return;
    case ov::genai::StopCriteria::HEURISTIC: {
        float highest_attainable_score = best_sum_logprobs / std::pow(float(cur_len), sampling_params.length_penalty);
        done = worst_score >= highest_attainable_score;
        return;
    }
    case ov::genai::StopCriteria::NEVER: {
        size_t length = sampling_params.length_penalty > 0.0 ? sequence_group->get_max_new_tokens() : cur_len;
        float highest_attainable_score = best_sum_logprobs / std::pow(float(length), sampling_params.length_penalty);
        done = worst_score >= highest_attainable_score;
        return;
    }
    default:
        OPENVINO_THROW("Beam search internal error: unknown mode");
    }
}
}
