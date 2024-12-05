// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <set>
#include <cstdlib>
#include <string_view>

#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/generation_config.hpp"
#include "generation_stream.hpp"

namespace ov::genai {
enum class SequenceStatus {
    RUNNING = 0,
    FINISHED = 1,
    OUT_OF_MEMORY = 2,
    WAITING = 3
};

using TokenIds = std::vector<int64_t>;
using LogProbs = std::vector<float>;
class SequenceGroup;

class Sequence {
    // This can be a problem if we launch two pipelines in the same application.
    static uint64_t _get_next_global_sequence_id() {
        const std::lock_guard<std::mutex> lock(m_counter_mutex);
        static uint64_t m_counter = 0;
        return m_counter++;
    }

    TokenIds m_generated_ids;
    LogProbs m_generated_log_probs;
    uint64_t m_grouped_id;
    uint64_t m_id = _get_next_global_sequence_id();
    SequenceStatus m_status = SequenceStatus::RUNNING;
    GenerationFinishReason m_finish_reason = GenerationFinishReason::NONE;
    float m_cumulative_log_prob = 0.0f;
    std::vector<int64_t> m_prefix_hashes;
    std::weak_ptr<SequenceGroup> m_sequence_group;
    static std::mutex m_counter_mutex;

    size_t _make_hash(size_t content_length);
public:
    using Ptr = std::shared_ptr<Sequence>;
    using CPtr = std::shared_ptr<const Sequence>;

    // don't use directly
    Sequence(const uint64_t id) : m_grouped_id(id) {};

    // don't use directly
    Sequence(const Sequence& seq, const uint64_t id) :
        m_generated_ids(seq.m_generated_ids),
        m_grouped_id(id),
        m_status(seq.m_status),
        m_cumulative_log_prob(seq.m_cumulative_log_prob){
        OPENVINO_ASSERT(seq.m_id != m_id);
    }

    static Sequence::Ptr create(const uint64_t id) {
        return std::make_shared<Sequence>(id);
    }

    static Sequence::Ptr fork(Sequence::CPtr sequence, const uint64_t id) {
        return std::make_shared<Sequence>(*sequence, id);
    }

    bool operator ==(const Sequence& other) const {
        return other.m_id == m_id;
    }

    uint64_t get_id() const {
        return m_id;
    }

    uint64_t get_grouped_id() const {
        return m_grouped_id;
    }

    bool has_finished() const {
        return m_status == SequenceStatus::FINISHED;
    }

    bool is_running() const {
        return m_status == SequenceStatus::RUNNING;
    }

    bool out_of_memory() const {
        return m_status == SequenceStatus::OUT_OF_MEMORY;
    }

    bool is_waiting() const {
        return m_status == SequenceStatus::WAITING;
    }

    void set_status(SequenceStatus status) {
        m_status = status;
    }

    GenerationFinishReason get_finish_reason() const {
        return m_finish_reason;
    }

    void set_finish_reason(GenerationFinishReason finish_reason) {
        m_finish_reason = finish_reason;
    }

    // appends new tokens to a generated part
    void append_token(int64_t token_id, float log_prob) {
        m_cumulative_log_prob += log_prob;
        m_generated_log_probs.push_back(log_prob);
        m_generated_ids.push_back(token_id);
    }

    // removes n last tokens and updates cumulative log prob
    // used to remove stop_string from the output
    void remove_last_tokens(int n) {
        OPENVINO_ASSERT(m_generated_ids.size() >= n, "Cannot remove more tokens than has been generated");
        for (int i = 0; i < n; i++) {
            m_cumulative_log_prob -= m_generated_log_probs.back();
            m_generated_log_probs.pop_back();
            m_generated_ids.pop_back();
        }
    }

    GenerationOutput get_last_generation_output(size_t token_cnt = 1) {
        GenerationOutput output;
        OPENVINO_ASSERT(m_generated_ids.size());
        output.score = get_cumulative_log_probs();

        auto generated_token_id = get_generated_ids();
        auto generated_log_probs = get_generated_log_probs();

        OPENVINO_ASSERT(get_generated_len() >= token_cnt);
        auto offset = get_generated_len() - token_cnt;

        std::vector<int64_t> token_id(generated_token_id.begin() + offset, generated_token_id.end());
        std::vector<float> log_probs(generated_log_probs.begin() + offset, generated_log_probs.end());

        output.generated_ids = token_id;
        output.generated_log_probs = log_probs;
        output.finish_reason = get_finish_reason();
        return output;
    }

    size_t get_generated_len() const {
        return m_generated_ids.size();
    }

    const TokenIds & get_generated_ids() const {
        return m_generated_ids;
    }

    const LogProbs & get_generated_log_probs() const {
        return m_generated_log_probs;
    }

    float get_cumulative_log_probs() const {
        return m_cumulative_log_prob;
    }

    void update_generated_log_prob(size_t idx, float log_prob) {
        OPENVINO_ASSERT(idx < m_generated_log_probs.size());
        m_generated_log_probs[idx] = log_prob;
    }

    float get_beam_search_score(const ov::genai::GenerationConfig& sampling_params) const {
        float cumulative_log_prob = get_cumulative_log_probs(), current_length = get_generated_len();
        float score = cumulative_log_prob / std::pow(current_length, sampling_params.length_penalty);
        return score;
    }



    // Each KV block can be uniquely identified by
    void set_sequence_group_ptr(std::shared_ptr<SequenceGroup> sequence_group) {
        m_sequence_group = sequence_group;
    }

    std::shared_ptr<SequenceGroup> get_sequence_group_ptr() const {
        OPENVINO_ASSERT(!m_sequence_group.expired());
        return m_sequence_group.lock();
    }

    // Each KV block can be uniquely identified by
    // the tokens within the block and the tokens in the prefix before the block.
    // hash(prefix tokens + block tokens) <--> KV Block
    size_t get_hash(size_t content_length = 0);
};

// contains a list of Sequences in generic case (beam search or parallel sampling)
// - each sequence shares the same prompt and KV-caches for promp
// - in case of beam search each sequence also shares specific part of generic phase
//   via reference counter mechanism on BlockManager level
class SequenceGroup {
    uint64_t m_request_id;
    std::vector<Sequence::Ptr> m_sequences;
    ov::genai::GenerationConfig m_sampling_params;
    std::size_t m_block_size;
    TokenIds m_prompt_ids;
    std::vector<float> m_prompt_log_probs;
    GenerationStream::Ptr m_generation_stream;
    bool m_enable_prefix_caching;
    size_t m_num_evicted_tokens = 0;
    bool m_has_echoed = false;

    uint64_t m_next_sequence_id = 0;
 
    // amount of processed tokens, e.g. prompt can be processed using multiple consequence inferences
    // so, we need to track which part of the prompt we have already processed
    size_t m_num_processed_tokens = 0;
    // a number of scheduled tokens by Scheduler::schedule logic
    size_t m_num_scheduled_tokens = 0;
    // context length of longest sequence within a group
    size_t m_max_content_len = 0;
    // max validation length within a group to check generated tokens
    size_t m_num_validation_tokens = 0;
    // flag to enable/disable token generation, e.g. in speculative decoding scenario
    bool m_is_gen_paused = false;


    SequenceGroup(uint64_t request_id, const ov::genai::GenerationConfig& sampling_params, std::size_t block_size, bool enable_prefix_caching)
        : m_request_id(request_id),
          m_sampling_params(sampling_params),
          m_block_size(block_size),
          m_enable_prefix_caching(enable_prefix_caching) {
            m_generation_stream = GenerationStream::create();    
           }

public:
    using Ptr = std::shared_ptr<SequenceGroup>;
    using CPtr = std::shared_ptr<const SequenceGroup>;

    SequenceGroup(uint64_t request_id, const TokenIds& input_ids, const ov::genai::GenerationConfig& sampling_params, std::size_t block_size, bool enable_prefix_caching)
        : SequenceGroup(request_id, ov::Tensor(ov::element::i64, ov::Shape{input_ids.size()}, (void *)input_ids.data()), sampling_params, block_size, enable_prefix_caching) {
    }

    SequenceGroup(uint64_t request_id, const ov::Tensor input_ids, const ov::genai::GenerationConfig& sampling_params, std::size_t block_size, bool enable_prefix_caching)
        : SequenceGroup(request_id, sampling_params, block_size, enable_prefix_caching) {
        add_sequence(Sequence::create(m_next_sequence_id++));

        m_prompt_ids.resize(input_ids.get_size());
        std::copy_n(input_ids.data<int64_t>(), input_ids.get_size(), m_prompt_ids.begin());
        m_prompt_log_probs.reserve(m_prompt_ids.size());
    }

    void add_sequence(const Sequence::Ptr & sequence) {
        m_sequences.emplace_back(sequence);
    }

    void remove_sequence(uint64_t sequence_id) {
        auto remove_it = std::remove_if(m_sequences.begin(), m_sequences.end(), [sequence_id] (Sequence::Ptr seq) {
            return seq->get_id() == sequence_id;
        });
        OPENVINO_ASSERT(remove_it != m_sequences.end(), "Failed to remove sequence with specified ID");
        m_sequences.erase(remove_it);
    }

    size_t get_prompt_len() const {
        return m_prompt_ids.size();
    }

    void pause_generation(bool status) {
        m_is_gen_paused = status;
    }

    // a sequence group can generate new tokens if it already processed m_max_content_len before
    bool can_generate_tokens() const {
        return m_max_content_len + m_num_validation_tokens >= get_prompt_len() && !m_is_gen_paused;
    }

    Sequence::Ptr operator[] (size_t index) {
        OPENVINO_ASSERT(m_sequences.size() > index);
        return m_sequences[index];
    }

    Sequence::CPtr operator[] (size_t index) const {
        OPENVINO_ASSERT(m_sequences.size() > index);
        return m_sequences[index];
    }

    size_t num_total_seqs() const {
        return m_sequences.size();
    }

    size_t num_finished_seqs() const {
        return std::count_if(m_sequences.begin(), m_sequences.end(), [] (Sequence::CPtr seq) {
            return seq->has_finished();
        });
    }

    size_t num_running_seqs() const {
        return num_total_seqs() - num_finished_seqs();
    }

    bool has_finished() const {
        return num_running_seqs() == 0;
    }

    bool is_running() const {
        return !has_finished();
    }

    const std::vector<Sequence::Ptr>& get_sequences() const {
        return m_sequences;
    }

    /**
     * @param seq_id Sequence identifier
     * @return Whether this group has the sequence with this ID.
     */
    bool has_sequence_with_id(size_t seq_id) const {
        auto it = std::find_if(m_sequences.begin(), m_sequences.end(), [seq_id](const Sequence::Ptr& val) {return val->get_id() == seq_id;});
        return it != m_sequences.end();
    }


    /**
     * @param seq_id Sequence identifier
     * @return Pointer to the sequence with this ID.
     * @throw ov::Exception if the sequence with ID seq_id is not in this SequenceGroup
     */
    Sequence::Ptr get_sequence_by_id(size_t seq_id) const {
        auto it = std::find_if(m_sequences.begin(), m_sequences.end(), [seq_id](const Sequence::Ptr& val) {return val->get_id() == seq_id;});
        OPENVINO_ASSERT(it != m_sequences.end(), "sequence with id ", seq_id, " not found in sequence group with request id ", m_request_id);
        return *it;
    }

    std::vector<Sequence::CPtr> get_finished_sequences() const {
        std::vector<Sequence::CPtr> finished_seqs;
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->has_finished() || m_sequences[seq_id]->out_of_memory()) {
                finished_seqs.push_back(m_sequences[seq_id]);
            }
        }

        // do we need to sort sequences here or sampler can handle it for us?
        std::sort(finished_seqs.begin(), finished_seqs.end(), [=] (Sequence::CPtr s1, Sequence::CPtr s2) {
            return s1->get_beam_search_score(m_sampling_params) > s2->get_beam_search_score(m_sampling_params);
        });

        return finished_seqs;
    }

    std::vector<Sequence::Ptr> get_running_sequences() {
        std::vector<Sequence::Ptr> running_seqs;
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->is_running()) {
                running_seqs.emplace_back(m_sequences[seq_id]);
            }
        }

        return running_seqs;
    }

    std::vector<Sequence::Ptr> get_not_finished_sequences() {
        std::vector<Sequence::Ptr> running_seqs;
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (!m_sequences[seq_id]->has_finished()) {
                running_seqs.emplace_back(m_sequences[seq_id]);
            }
        }

        return running_seqs;
    }

    std::vector<Sequence::CPtr> get_running_sequences() const {
        std::vector<Sequence::CPtr> running_seqs;
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->is_running()) {
                running_seqs.emplace_back(m_sequences[seq_id]);
            }
        }

        return running_seqs;
    }

    uint64_t get_request_id() const {
        return m_request_id;
    }

    size_t get_num_scheduled_tokens() const {
        return m_num_scheduled_tokens;
    }

    size_t get_num_processed_tokens() const {
        return m_num_processed_tokens;
    }

    /**
     * Registers within the sequence group that a given amount of tokens
     * has been evicted from the underlying KV cache.
     * NB: no per-layer or per-sequence indexing is required since current invariant is that
     * there is always the same amount of KV cache blocks for each layer (i.e. eviction algo
     * always evicts the same amount of blocks for each layer in each eviction step) and for each sequence in a group
     * @param num_evicted_tokens Number of tokens evicted for this sequence at this generation step.
     */
    void register_token_eviction(size_t num_evicted_tokens) {
        m_num_evicted_tokens += num_evicted_tokens;
    }


    /**
     * Resets the eviction tracking on this sequence to the state prior to any eviction taking place.
     */
    void reset_eviction_token_count() {
        m_num_evicted_tokens = 0;
    }

    /**
     * @return Number of tokens evicted for this sequence since the start of the processing for this sequence
     */
    size_t get_num_evicted_tokens() const {
        return m_num_evicted_tokens;
    }

    void preempt_tokens(size_t num_preempt_tokens) {
        OPENVINO_ASSERT(num_preempt_tokens <= m_num_processed_tokens);
        m_num_processed_tokens -= num_preempt_tokens;
    }

    // returns context length taking into account scheduled tokens
    size_t get_context_len() const {
        return get_num_processed_tokens() + get_num_scheduled_tokens();
    }


    bool requires_sampling() const {
        return get_context_len() >= get_prompt_len() && get_context_len() > m_max_content_len && m_sampling_params.max_new_tokens > 0;
    }

    void schedule_tokens(size_t num_tokens) {
        m_num_scheduled_tokens = num_tokens;
    }

    void clear_scheduled_tokens() {
        m_num_scheduled_tokens = 0;
        m_num_validation_tokens = 0;
    }

    bool is_scheduled() const {
        return m_num_scheduled_tokens > 0;
    }

    void set_num_validated_tokens(size_t k) {
        // in case of non-prompt we need to take prev tokens + token to validate
        // m_num_validation_tokens = get_num_processed_tokens() ? k + 1 : k;
        m_num_validation_tokens = k;
    }

    size_t get_num_tokens_to_validate() {
        return m_num_validation_tokens;
    }

    size_t get_num_available_tokens_for_batching() const {
        OPENVINO_ASSERT(!has_finished(), "Internal error: this function cannot be called on finished sequence group");
        OPENVINO_ASSERT(get_num_scheduled_tokens() == 0, "Internal error: this function cannot be called when we are already in scheduling phase");
        // if sequence group has not finished, it has at least one token to process
        size_t num_available_tokens = std::max(get_prompt_len(), m_max_content_len);
        return std::max<size_t>(num_available_tokens - m_num_processed_tokens, 1u) + m_num_validation_tokens;
    }

    // mark current schedule phase as finished and updates internal counters
    void finish_iteration() {
        m_num_processed_tokens += m_num_scheduled_tokens;
        // if some processed tokens were evicted, max content len is greater than number of processed tokens
        m_max_content_len = std::max(m_max_content_len, m_num_processed_tokens);
        clear_scheduled_tokens();
    }

    void update_processed_tokens_num(size_t processed_tokens) {
        m_num_processed_tokens = processed_tokens;
        m_max_content_len = processed_tokens;
    }

    void clear_waiting_sequences() {
        if (!is_waiting())
            return;

        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->is_waiting()) {
                m_sequences[seq_id]->set_status(SequenceStatus::RUNNING);
            }
        }
    }

    const TokenIds& get_prompt_ids() const {
        return m_prompt_ids;
    }

    void append_prompt_log_prob(float log_prob) {
        m_prompt_log_probs.push_back(log_prob);
    }

    /**
     * @return The number of logical KV cache blocks required to host all the tokens in this sequence group, taking into account previous token evictions.
     */
    size_t get_num_logical_blocks() const {
        return (get_context_len() - get_num_evicted_tokens() + m_block_size - 1) / m_block_size;
    }


    // requires number of physical blocks for next generation
    size_t get_num_blocks() const {
        return get_num_logical_blocks();
    }

    size_t get_block_size() const {
        return m_block_size;
    }

    Sequence::Ptr fork_sequence(Sequence::CPtr sequence) {
        auto ptr = sequence->get_sequence_group_ptr();
        m_sequences.emplace_back(Sequence::fork(std::move(sequence), m_next_sequence_id++));
        set_sequence_group_ptr(ptr);
        return m_sequences.back();
    }

    const ov::genai::GenerationConfig& get_sampling_parameters() const {
        return m_sampling_params;
    }

    void set_out_of_memory() {
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->is_running()) {
                m_sequences[seq_id]->set_status(SequenceStatus::OUT_OF_MEMORY);
            }
        }
    }

    void set_waiting() {
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->is_running()) {
                m_sequences[seq_id]->set_status(SequenceStatus::WAITING);
            }
        }
    }

    bool out_of_memory() const {
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->out_of_memory()) {
                return true;
            }
        }
        return false;
    }

    bool is_waiting() const {
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->is_waiting()) {
                return true;
            }
        }
        return m_is_gen_paused;
    }

    void set_sequence_group_ptr(std::shared_ptr<SequenceGroup> sequence_group) {
        for (auto sequence: m_sequences) {
            sequence->set_sequence_group_ptr(sequence_group);
        }
    }

    GenerationStream::Ptr get_generation_stream() {
        return m_generation_stream;
    }

    void set_generation_status(GenerationStatus status) {
        m_generation_stream->set_generation_status(status);
    }

    bool handle_dropped() {
        return m_generation_stream->get_status() == GenerationStatus::DROPPED_BY_HANDLE;
    }

    void push_empty_outputs() {
        m_generation_stream->push({});
    }

    void push_outputs() {
        GenerationOutputs outputs;
        for (auto& sequence: m_sequences) {
            GenerationOutput output;
            output.generated_ids = sequence->get_generated_ids();
            output.generated_log_probs = sequence->get_generated_log_probs();
            if (m_sampling_params.echo) {
                output.generated_ids.insert(output.generated_ids.begin(), m_prompt_ids.begin(), m_prompt_ids.end());
                output.generated_log_probs.insert(output.generated_log_probs.begin(), m_prompt_log_probs.begin(), m_prompt_log_probs.end());
            }
            output.score = m_sampling_params.is_beam_search() ? sequence->get_beam_search_score(m_sampling_params) : sequence->get_cumulative_log_probs();
            output.finish_reason = sequence->get_finish_reason();
            outputs.emplace(sequence->get_grouped_id(), output);
        }
        m_generation_stream->push(std::move(outputs));
    }

    void push_partial_outputs(size_t token_cnt = 1) {
        GenerationOutputs outputs;
        for (auto& sequence : m_sequences) {
            // todo: check seq.is_finished() to generate without several </s>
            // or is it ok to use padding?
            auto output = sequence->get_last_generation_output(token_cnt);
            if (m_sampling_params.echo && !m_has_echoed) {
                output.generated_ids.insert(output.generated_ids.begin(), m_prompt_ids.begin(), m_prompt_ids.end());
                output.generated_log_probs.insert(output.generated_log_probs.begin(), m_prompt_log_probs.begin(), m_prompt_log_probs.end());
            }
            outputs.emplace(sequence->get_grouped_id(), output);
        }
        m_has_echoed = true;
        m_generation_stream->push(std::move(outputs));
    }

    void notify_handle(size_t num_output_token_to_push = 0) {
        if (out_of_memory()) {
            set_generation_status(GenerationStatus::IGNORED);
        } else if (has_finished()) {
            set_generation_status(GenerationStatus::FINISHED);
        }
        // For beam search streaming is not available, so we notify only upon finishing
        if(m_sampling_params.is_beam_search()) {
            if (has_finished() || out_of_memory()) {
                push_outputs();
            }
        } else if (m_sampling_params.is_greedy_decoding() || m_sampling_params.is_multinomial()) {
            // We can stream only when one sequence is returned and we don't use stop strings that would be excluded from the output
            // (after stop string is detected its tokens are already sent)
            if (num_total_seqs() == 1 &&
                (m_sampling_params.stop_strings.empty() || m_sampling_params.include_stop_str_in_output)) {
                if (num_output_token_to_push)
                    push_partial_outputs(num_output_token_to_push);
            } else if (has_finished() || out_of_memory()) {
                push_outputs();
            }
        }
    }

    
    // Special notification path for max_new_tokens == 0 where we don't expect to return any new tokens, but only process prompt
    void notify_handle_echo_only() {
        // This method is called after scheduling and before sampling,
        // so m_num_processed_tokens does not include recently forwarded tokens hence this is our starting position
        // we return m_num_scheduled_tokens tokens as they were forwarded in the current step, meaning context length is our last position.
        size_t first_token_position = m_num_processed_tokens;
        size_t last_token_position = get_context_len();

        GenerationOutput output;
        output.generated_ids = std::vector<int64_t>(m_prompt_ids.begin() + first_token_position, m_prompt_ids.begin() + last_token_position);
        output.generated_log_probs = std::vector<float>(m_prompt_log_probs.begin() + first_token_position, m_prompt_log_probs.begin() + last_token_position);
        output.score = 0.0; // Should we accumulate prompt log probs here?
        output.finish_reason = GenerationFinishReason::NONE;

        if (last_token_position == get_prompt_len()) {
            output.finish_reason = GenerationFinishReason::LENGTH;
            set_generation_status(GenerationStatus::FINISHED);
            m_sequences[0]->set_status(SequenceStatus::FINISHED); // for cleanup
        }
        GenerationOutputs outputs;
        outputs.emplace(0, output);
        m_generation_stream->push(std::move(outputs));
    } 
};
}
