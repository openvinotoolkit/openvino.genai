// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdlib>

#include "sampling_parameters.hpp"

enum class SequenceStatus {
    WAITING = 0,
    FINISHED = 1
};

using TokenIds = std::vector<int64_t>;

class Sequence {
    size_t m_prompt_len;
    TokenIds m_generated_ids;
    uint64_t m_sequence_id = _get_next_sequence_id();
    SequenceStatus m_status = SequenceStatus::WAITING;
    size_t m_num_processed_tokens = 0;

    static uint64_t _get_next_sequence_id() {
        static uint64_t m_counter = 0;
        return m_counter++;
    }

public:
    explicit Sequence(size_t prompt_len)
        : m_prompt_len(prompt_len) {
    }

    bool operator ==(const Sequence& other) const {
        return other.m_sequence_id == m_sequence_id;
    }

    uint64_t get_id() const {
        return m_sequence_id;
    }

    size_t get_num_logical_blocks() const {
        return (m_num_processed_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    bool has_finished() const {
        return m_status == SequenceStatus::FINISHED;
    }

    void set_status(SequenceStatus status) {
        m_status = status;
    }

    // appends new tokens to a generated part
    void append_token(int64_t token_id) {
        m_generated_ids.push_back(token_id);
    }

    void update_processed_tokens(size_t num_processed_tokens) {
        m_num_processed_tokens += num_processed_tokens;
    }

    const TokenIds & get_generated_ids() const {
        return m_generated_ids;
    }
};

// contains a list of Sequences in generic case (beam search or parallel sampling)
// - each sequence shares the same prompt and KV-caches for promp
// - in case of beam search each sequence also shares specific part of generic phase
//   via reference counter machanism on BlockManager level
class SequenceGroup {
    uint64_t m_request_id;
    std::vector<Sequence> m_sequences;
    SamplingParameters m_sampling_params;
    TokenIds m_prompt_ids;

    // amount of processed tokens, e.g. prompt can be processed using multiple consequence inferences
    // so, we need to track which part of the prompt we have already processed
    size_t m_num_processed_tokens = 0;
    // a number of scheduled tokens by Scheduler::schedule logic
    size_t m_num_scheduled_tokens = 0;

    int64_t _get_position_id() const {
        return get_context_len() - 1;
    }

    SequenceGroup(uint64_t request_id, const SamplingParameters& sampling_params)
        : m_request_id(request_id),
          m_sampling_params(sampling_params) { }
public:
    SequenceGroup(uint64_t request_id, const TokenIds& input_ids, const SamplingParameters& sampling_params) :
        SequenceGroup(request_id, sampling_params) {
        add_sequence(Sequence(input_ids.size()));
    }

    SequenceGroup(uint64_t request_id, const ov::Tensor& input_ids, const SamplingParameters& sampling_params) :
        SequenceGroup(request_id, sampling_params) {
        add_sequence(Sequence(input_ids.get_size()));
    }

    void add_sequence(const Sequence & sequence) {
        m_sequences.push_back(sequence);
    }

    void remove_sequence(uint64_t sequence_id) {
        OPENVINO_ASSERT(std::remove_if(m_sequences.begin(), m_sequences.end(), [sequence_id] (const Sequence & seq) {
            return seq.get_id() == sequence_id;
        }) != m_sequences.end(), "Failed to remove sequence with specified ID");
    }

    bool is_prompt_phase() const {
        return m_num_processed_tokens < get_prompt_len();
    }

    size_t get_num_scheduled_tokens() const {
        return m_num_scheduled_tokens;
    }

    const Sequence& operator[] (size_t index) const {
        OPENVINO_ASSERT(m_sequences.size() > index);
        return m_sequences[index];
    }

    Sequence& operator[] (size_t index) {
        OPENVINO_ASSERT(m_sequences.size() > index);
        return m_sequences[index];
    }

    size_t num_total_seqs() const {
        return m_sequences.size();
    }

    size_t num_finished_seqs() const {
        return std::count_if(m_sequences.begin(), m_sequences.end(), [] (const Sequence& seq) {
            return seq.has_finished();
        });
    }

    size_t get_prompt_len() const {
        return m_prompt_ids.size();
    }

    size_t num_unfinished_seqs() const {
        return num_total_seqs() - num_finished_seqs();
    }

    bool has_finished() const {
        return num_unfinished_seqs() == 0;
    }

    std::vector<Sequence> get_unfinished_sequences() const {
        std::vector<Sequence> m_unfinished_seqs;
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (!m_sequences[seq_id].has_finished()) {
                m_unfinished_seqs.push_back(m_sequences[seq_id]);
            }
        }

        return m_unfinished_seqs;
    }

    size_t get_num_available_tokens_for_batching() const {
        size_t num_unfinished_sequences = num_unfinished_seqs();
        OPENVINO_ASSERT(num_unfinished_sequences > 0);
        return is_prompt_phase() ? get_num_available_tokens_for_batching() : num_unfinished_seqs();
    }

    uint64_t get_request_id() const {
        return m_request_id;
    }

    size_t get_context_len() const {
        OPENVINO_ASSERT(!has_finished());
        return get_num_processed_tokens() + get_num_scheduled_tokens();
    }

    bool requires_sampling() const {
        return get_context_len() >= get_prompt_len();
    }

    void schedule_tokens(size_t num_tokens) {
        m_num_scheduled_tokens = num_tokens;
    }

    bool is_scheduled() const {
        return m_num_scheduled_tokens > 0;
    }

    // mark current schedule phase as finished and updates internal counters
    void finish_iteration() {
        for (size_t i = 0; i < m_sequences.size(); ++i) {
            m_sequences[i].update_processed_tokens(m_num_scheduled_tokens);
        }

        m_num_processed_tokens += m_num_scheduled_tokens;
        m_num_scheduled_tokens = 0;
    }

    size_t get_num_processed_tokens() const {
        return m_num_processed_tokens;
    }

    size_t get_num_available_tokens_for_batching() const {
        OPENVINO_ASSERT(m_num_scheduled_tokens == 0);
        return get_prompt_len() - m_num_processed_tokens;
    }

    const TokenIds & get_prompt_ids() const {
        return m_prompt_ids;
    }

    size_t get_num_blocks() const {
        return (get_context_len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
};
