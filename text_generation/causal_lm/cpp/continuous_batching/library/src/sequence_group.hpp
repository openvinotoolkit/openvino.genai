// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <set>
#include <cstdlib>

#include "generation_handle.hpp"
#include "generation_config.hpp"
#include "generation_stream.hpp"

enum class SequenceStatus {
    RUNNING = 0,
    FINISHED = 1,
    OUT_OF_MEMORY = 2,
    WAITING = 3
};

using TokenIds = std::vector<int64_t>;

class Sequence {
    // This can be a problem if we launch two pipelines in the same application.
    static uint64_t _get_next_global_sequence_id() {
        static uint64_t m_counter = 0;
        return m_counter++;
    }

    TokenIds m_generated_ids;
    uint64_t m_grouped_id;
    uint64_t m_id = _get_next_global_sequence_id();
    SequenceStatus m_status = SequenceStatus::RUNNING;
    float m_cumulative_log_prob = 0.0f;

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
        m_cumulative_log_prob(seq.m_cumulative_log_prob) {
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

    // appends new tokens to a generated part
    void append_token(int64_t token_id, float log_prob) {
        m_cumulative_log_prob += log_prob;
        m_generated_ids.push_back(token_id);     
    }

    GenerationOutput get_last_generation_output() {
        GenerationOutput output;
        OPENVINO_ASSERT(m_generated_ids.size());
        output.score = get_cumulative_log_probs();
        output.generated_token_ids = std::vector<int64_t> {m_generated_ids.back()};
        return output;
    }

    size_t get_generated_len() const {
        return m_generated_ids.size();
    }

    const TokenIds & get_generated_ids() const {
        return m_generated_ids;
    }

    float get_cumulative_log_probs() const {
        return m_cumulative_log_prob;
    }

    float get_beam_search_score(const GenerationConfig& sampling_params) const {
        float cumulative_log_prob = get_cumulative_log_probs(), current_length = get_generated_len();
        float score = cumulative_log_prob / std::pow(current_length, sampling_params.length_penalty);
        return score;
    }
};

// contains a list of Sequences in generic case (beam search or parallel sampling)
// - each sequence shares the same prompt and KV-caches for promp
// - in case of beam search each sequence also shares specific part of generic phase
//   via reference counter machanism on BlockManager level
class SequenceGroup {
    uint64_t m_request_id;
    std::vector<Sequence::Ptr> m_sequences;
    GenerationConfig m_sampling_params;
    std::size_t m_block_size;
    TokenIds m_prompt_ids;
    GenerationStream::Ptr m_generation_stream;

    uint64_t m_next_sequence_id = 0;

    bool m_preempted = false;
 
    // amount of processed tokens, e.g. prompt can be processed using multiple consequence inferences
    // so, we need to track which part of the prompt we have already processed
    size_t m_num_processed_tokens = 0;
    // a number of scheduled tokens by Scheduler::schedule logic
    size_t m_num_scheduled_tokens = 0;
    // context length of longest sequence within a group
    size_t m_max_content_len = 0;

    SequenceGroup(uint64_t request_id, const GenerationConfig& sampling_params, std::size_t block_size)
        : m_request_id(request_id),
          m_sampling_params(sampling_params),
          m_block_size(block_size) {
            m_generation_stream = GenerationStream::create();    
           }
public:
    using Ptr = std::shared_ptr<SequenceGroup>;
    using CPtr = std::shared_ptr<const SequenceGroup>;

    SequenceGroup(uint64_t request_id, const TokenIds& input_ids, const GenerationConfig& sampling_params, std::size_t block_size)
        : SequenceGroup(request_id, ov::Tensor(ov::element::i64, ov::Shape{input_ids.size()}, (void *)input_ids.data()), sampling_params, block_size) {
    }

    SequenceGroup(uint64_t request_id, const ov::Tensor input_ids, const GenerationConfig& sampling_params, std::size_t block_size)
        : SequenceGroup(request_id, sampling_params, block_size) {
        add_sequence(Sequence::create(m_next_sequence_id++));

        m_prompt_ids.resize(input_ids.get_size());
        std::copy_n(input_ids.data<int64_t>(), input_ids.get_size(), m_prompt_ids.begin());
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

    std::vector<int64_t> try_finish_generation() {
        std::vector<int64_t> dropped_seq_ids;
        for (auto& running_sequence : get_running_sequences()) {
            const auto generated_len = running_sequence->get_generated_len();
            if (m_sampling_params.max_new_tokens == generated_len ||
                running_sequence->get_generated_ids().back() == m_sampling_params.eos_token_id && !m_sampling_params.ignore_eos) {
                // stop sequence by max_new_tokens or EOS token
                running_sequence->set_status(SequenceStatus::FINISHED);
                dropped_seq_ids.push_back(running_sequence->get_id());
            }
        }
        return dropped_seq_ids;
    }

    size_t get_prompt_len() const {
        return m_prompt_ids.size();
    }

    // a sequence group can generate new tokens if it already proccessed m_max_content_len before
    bool can_generate_tokens() const {
        return m_max_content_len >= get_prompt_len();
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

    void preempt_tokens(size_t num_preempt_tokens) {
        OPENVINO_ASSERT(num_preempt_tokens <= m_num_processed_tokens);
        m_num_processed_tokens -= num_preempt_tokens;
        m_preempted = true;
    }

    // returns context length taking into account scheduled tokens
    size_t get_context_len() const {
        OPENVINO_ASSERT(!has_finished());
        return get_num_processed_tokens() + get_num_scheduled_tokens();
    }

    bool requires_sampling() const {
        return get_context_len() >= get_prompt_len() && get_context_len() > m_max_content_len;
    }

    void schedule_tokens(size_t num_tokens) {
        m_num_scheduled_tokens = num_tokens;
    }

    void clear_scheduled_tokens() {
        m_num_scheduled_tokens = 0;
    }

    bool is_scheduled() const {
        return m_num_scheduled_tokens > 0;
    }

    size_t get_num_available_tokens_for_batching() const {
        OPENVINO_ASSERT(!has_finished(), "Internal error: this function cannot be called on finished sequence group");
        OPENVINO_ASSERT(get_num_scheduled_tokens() == 0, "Internal error: this function cannot be called when we are already in scheduling phase");
        // if sequence group has not finished, it has at least one token to process
        size_t num_available_tokens = std::max(get_prompt_len(), m_max_content_len);
        return std::max<size_t>(num_available_tokens - m_num_processed_tokens, 1u);
    }

    // mark current schedule phase as finished and updates internal counters
    void finish_iteration() {
        m_num_processed_tokens += m_num_scheduled_tokens;
        // if some processed tokens were evicted, max content len is greater than number of processed tokens
        m_max_content_len = std::max(m_max_content_len, m_num_processed_tokens);
        clear_scheduled_tokens();
    }

    void clear_waiting_sequences() {
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->is_waiting()) {
                m_sequences[seq_id]->set_status(SequenceStatus::RUNNING);
            }
        }
    }

    const TokenIds& get_prompt_ids() const {
        return m_prompt_ids;
    }

    size_t get_num_logical_blocks() const {
        return (get_context_len() + m_block_size - 1) / m_block_size;
    }

    // requires number of physical blocks for next generation
    size_t get_num_blocks() const {
        return get_num_logical_blocks();
    }

    size_t get_block_size() const {
        return m_block_size;
    }

    Sequence::Ptr fork_sequence(Sequence::CPtr sequence) {
        m_sequences.emplace_back(Sequence::fork(std::move(sequence), m_next_sequence_id++));
        return m_sequences.back();
    }

    const GenerationConfig& get_sampling_parameters() const {
        return m_sampling_params;
    }

    void reset() {
        m_sequences.clear();
        m_next_sequence_id = 0;
        add_sequence(Sequence::create(m_next_sequence_id++));
        clear_scheduled_tokens();
        m_num_processed_tokens = 0;
        m_max_content_len = 0;
    }

    bool is_empty() {
        if (m_sequences.size() > 1)
            return false;
        OPENVINO_ASSERT(m_sequences.size() == 1);
        if (m_sequences[0]->get_generated_len() > 0 || m_sequences[0]->get_cumulative_log_probs() != 0.0f)
            return false;
        return true; 
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
        return false;
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

    void notify_handle() {

        if (out_of_memory()) {
            set_generation_status(GenerationStatus::IGNORED);
        } else if (has_finished()) {
            set_generation_status(GenerationStatus::FINISHED);
        }

        GenerationOutputs outputs;

        // For beam search streaming is not available, so we notify only upon finishing
        if(m_sampling_params.is_beam_search()) {
            if (has_finished()) {
                std::vector<Sequence::CPtr> finished_sequences = get_finished_sequences();

                OPENVINO_ASSERT(finished_sequences.size() == num_total_seqs() && has_finished());
                for (auto& sequence: finished_sequences) {
                    GenerationOutput output;
                    output.generated_token_ids = sequence->get_generated_ids();
                    output.score = sequence->get_beam_search_score(m_sampling_params);
                    outputs.emplace(sequence->get_grouped_id(), output);
                }

                if (outputs.size()) {
                    m_generation_stream->push(std::move(outputs));
                }
            }
        // For greedy or multinomial sampling we decide whever to stream partial results depending on the user parameter
        } else if (m_sampling_params.is_greedy_sampling() || m_sampling_params.is_multinomial()) {
            // TO DO: Now we always stream for greedy search for the sake of benchmarking 
            if (num_total_seqs() == 1 /* m_sampling_params.stream */) {
                // TODO: support streamimg for n seqs
                for (auto& sequence : m_sequences) {
                    // todo: check seq.is_finished() to generate without several </s>
                    // or is it ok to use padding?
                    const auto last_gen_token = sequence->get_last_generation_output();
                    outputs.emplace(sequence->get_grouped_id(), last_gen_token);
                }
                m_generation_stream->push(std::move(outputs));
            } else if (has_finished()) {
                std::vector<Sequence::CPtr> finished_sequences = get_finished_sequences();

                OPENVINO_ASSERT(finished_sequences.size() == num_total_seqs() && has_finished());
                for (auto& sequence: finished_sequences) {
                    GenerationOutput output;
                    output.generated_token_ids = sequence->get_generated_ids();
                    output.score = sequence->get_cumulative_log_probs();
                    outputs.emplace(sequence->get_grouped_id(), output);
                }

                if (outputs.size()) {
                    m_generation_stream->push(std::move(outputs));
                }
            }
        }
    } 
};
