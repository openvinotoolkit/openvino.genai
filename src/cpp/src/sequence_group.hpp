// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cassert>
#include <set>
#include <cstdlib>
#include <string_view>
#include <memory>
#include <optional>

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

enum class SequenceGroupType {
    TOKENS,
    EMBEDDINGS
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
    SequenceGroup* m_sequence_group = nullptr;
    static std::mutex m_counter_mutex;
    std::vector<std::vector<float>> m_generated_ids_embeds;
    SequenceGroupType m_type;
    size_t m_hidden_size;

    // Embeddings hash calculation params
    static constexpr size_t m_embeddings_hash_max_num_values = 10; // max number of values used for embeddings hash calculation
    static constexpr size_t m_embeddings_hash_calculation_stride = 50; // the stride with which values are taken from embeddings vector

    size_t _make_hash(size_t content_length);

    static std::vector<int64_t> _reduce_embedding(const std::vector<float>& embedding);

    explicit Sequence(const uint64_t id, const SequenceGroupType type, const size_t hidden_size) : m_grouped_id(id), m_type(type), m_hidden_size(hidden_size) {}

    Sequence(const Sequence& seq, const uint64_t id) :
        m_generated_ids(seq.m_generated_ids),
        m_grouped_id(id),
        m_status(seq.m_status),
        m_cumulative_log_prob(seq.m_cumulative_log_prob),
        m_sequence_group(seq.m_sequence_group),
        m_type(seq.m_type),
        m_hidden_size(seq.m_hidden_size) {
        OPENVINO_ASSERT(seq.m_id != m_id);
    }

public:
    using Ptr = std::shared_ptr<Sequence>;
    using CPtr = std::shared_ptr<const Sequence>;

    static Sequence::Ptr create(const uint64_t id, const SequenceGroupType type = SequenceGroupType::TOKENS, const size_t hidden_size = 0) {
        return Sequence::Ptr(new Sequence(id, type, hidden_size));
    }

    static Sequence::Ptr fork(Sequence::CPtr sequence, const uint64_t id) {
        return Sequence::Ptr(new Sequence(*sequence, id));
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

    GenerationOutput get_last_generation_output(size_t token_cnt = 1, size_t num_token_to_ignore = 0) {
        GenerationOutput output;
        if (token_cnt > 0) {
            OPENVINO_ASSERT(m_generated_ids.size());
            output.score = get_cumulative_log_prob();

            auto generated_token_id = get_generated_ids();
            auto generated_log_probs = get_generated_log_probs();

            OPENVINO_ASSERT(get_generated_len() >= token_cnt);
            if (get_generated_len() > num_token_to_ignore) {
                auto offset = get_generated_len() - token_cnt - num_token_to_ignore;
                auto offset_back = get_generated_len() - num_token_to_ignore;

                std::vector<int64_t> token_id(generated_token_id.begin() + offset, generated_token_id.begin() + offset_back);
                std::vector<float> log_probs(generated_log_probs.begin() + offset, generated_log_probs.begin() + offset_back);

                output.generated_ids = std::move(token_id);
                output.generated_log_probs = std::move(log_probs);
                output.finish_reason = get_finish_reason();
            }
        }
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

    float get_cumulative_log_prob() const {
        return m_cumulative_log_prob;
    }

    void update_generated_log_prob(size_t idx, float log_prob) {
        OPENVINO_ASSERT(idx < m_generated_log_probs.size());
        m_generated_log_probs[idx] = log_prob;
    }

    float get_beam_search_score(const ov::genai::GenerationConfig& sampling_params) const {
        float cumulative_log_prob = get_cumulative_log_prob(), current_length = get_generated_len();
        float score = cumulative_log_prob / std::pow(current_length, sampling_params.length_penalty);
        return score;
    }

    // Each KV block can be uniquely identified by
    void set_sequence_group_ptr(SequenceGroup* sequence_group) {
        assert(sequence_group != nullptr);
        m_sequence_group = sequence_group;
    }

    const std::vector<std::vector<float>>& get_generated_ids_embeds() const {
        OPENVINO_ASSERT(m_type == ov::genai::SequenceGroupType::EMBEDDINGS);
        return m_generated_ids_embeds;
    }

    void append_generated_ids_embeds(ov::Tensor generated_ids_embeds) {
        OPENVINO_ASSERT(m_type == SequenceGroupType::EMBEDDINGS);
        auto embeds_count = generated_ids_embeds.get_shape()[1];
        OPENVINO_ASSERT(m_hidden_size == generated_ids_embeds.get_shape()[2]);

        auto current_embeds_size = m_generated_ids_embeds.size();
        for (size_t i = current_embeds_size, idx = 0; i < current_embeds_size + embeds_count; i++, idx++) {
            m_generated_ids_embeds.emplace_back(std::vector<float>());
            m_generated_ids_embeds[i].resize(m_hidden_size);
            std::copy_n(generated_ids_embeds.data<float>() + idx * m_hidden_size, m_hidden_size, m_generated_ids_embeds[i].begin());

        }
    }

    std::shared_ptr<SequenceGroup> get_sequence_group_ptr() const;

    // Each KV block can be uniquely identified by
    // the tokens within the block and the tokens in the prefix before the block.
    // hash(prefix tokens + block tokens) <--> KV Block
    size_t get_hash(size_t content_length = 0);
};

// contains a list of Sequences in generic case (beam search or parallel sampling)
// - each sequence shares the same prompt and KV-caches for prompt
// - in case of beam search each sequence also shares specific part of generic phase
//   via reference counter mechanism on BlockManager level
class SequenceGroup  : public std::enable_shared_from_this<SequenceGroup> {
    uint64_t m_request_id;
    std::vector<Sequence::Ptr> m_sequences;
    ov::genai::GenerationConfig m_sampling_params;
    std::size_t m_block_size;
    TokenIds m_prompt_ids;
    std::vector<std::vector<float>> m_input_embeds;
    std::optional<std::vector<int64_t>> m_token_type_ids;
    std::vector<float> m_prompt_log_probs;
    GenerationStream::Ptr m_generation_stream;
    size_t m_num_evicted_tokens = 0;
    bool m_has_echoed = false;
    SequenceGroupType m_sequence_group_type;

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
    // output seq len at current iteration
    size_t m_output_seq_len = 0;

    size_t m_num_streamed_tokens = 0, m_stream_window_size = 0;

    SequenceGroup(uint64_t request_id, const ov::genai::GenerationConfig& sampling_params, std::size_t block_size)
        : m_request_id(request_id),
          m_sampling_params(sampling_params),
          m_block_size(block_size),
          m_sequence_group_type(SequenceGroupType::TOKENS),
          m_generation_stream(GenerationStream::create()) { }

    bool out_of_memory() const {
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->out_of_memory()) {
                return true;
            }
        }
        return false;
    }

public:
    using Ptr = std::shared_ptr<SequenceGroup>;
    using CPtr = std::shared_ptr<const SequenceGroup>;

    SequenceGroup(uint64_t request_id, const TokenIds& input_ids, const ov::genai::GenerationConfig& sampling_params, std::size_t block_size)
        : SequenceGroup(request_id, ov::Tensor(ov::element::i64, ov::Shape{input_ids.size()}, (void *)input_ids.data()), sampling_params, block_size, std::nullopt) {
    }

    SequenceGroup(uint64_t request_id, const ov::Tensor input_ids, const ov::genai::GenerationConfig& sampling_params, std::size_t block_size, const std::optional<ov::Tensor>& token_type_ids = std::nullopt)
        : SequenceGroup(request_id, sampling_params, block_size) {
        
        size_t prompt_len;
        size_t hidden_size = 0;
        if (input_ids.get_shape().size() > 1) {
            prompt_len = input_ids.get_shape()[1];
        } else {
            prompt_len = input_ids.get_size();
        }
        OPENVINO_ASSERT(prompt_len > 0, "Prompt length cannot be 0");

        if (input_ids.get_element_type() == ov::element::i64) {
            m_prompt_ids.resize(prompt_len);
            OPENVINO_SUPPRESS_DEPRECATED_START
            std::copy_n(input_ids.data<int64_t>(), prompt_len, m_prompt_ids.begin());
            OPENVINO_SUPPRESS_DEPRECATED_END
            m_sequence_group_type = SequenceGroupType::TOKENS;
        } else if (input_ids.get_element_type() == ov::element::f32) {
            hidden_size = input_ids.get_shape()[2];
            m_input_embeds.resize(prompt_len);
            for (size_t i = 0; i < prompt_len; i++) {
                m_input_embeds[i].resize(hidden_size);
                OPENVINO_SUPPRESS_DEPRECATED_START
                std::copy_n(input_ids.data<float>() + i * hidden_size, hidden_size, m_input_embeds[i].begin());
                OPENVINO_SUPPRESS_DEPRECATED_END
            }
            if (token_type_ids.has_value()) {
                const ov::Tensor& tokens = token_type_ids.value();
                m_token_type_ids = std::vector<int64_t>(tokens.get_size());
                OPENVINO_SUPPRESS_DEPRECATED_START
                std::copy_n(tokens.data<int64_t>(), tokens.get_size(), m_token_type_ids->begin());
                OPENVINO_SUPPRESS_DEPRECATED_END
            }
            m_sequence_group_type = SequenceGroupType::EMBEDDINGS;
        }
        else {
            OPENVINO_THROW("Unknown tensor format.");
        }
        m_prompt_log_probs.reserve(prompt_len);

        // create a single sequence
        add_sequence(Sequence::create(m_next_sequence_id++, m_sequence_group_type, hidden_size));
    }

    void add_sequence(const Sequence::Ptr & sequence) {
        sequence->set_sequence_group_ptr(this);
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
        if (m_sequence_group_type == SequenceGroupType::EMBEDDINGS) {
            return m_input_embeds.size();
        }
        else if (m_sequence_group_type == SequenceGroupType::TOKENS) {
            return m_prompt_ids.size();
        }
        else {
            OPENVINO_THROW("Not implemented.");
        }
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

    size_t num_running_seqs() const {
        return std::count_if(m_sequences.begin(), m_sequences.end(), [] (Sequence::CPtr seq) {
            return seq->is_running();
        });
    }

    bool has_finished() const {
        return !is_running();
    }

    bool is_running() const {
        return num_running_seqs() > 0;
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

    // must be used only after sequence group generation loop has finished (either by lenght or OOM)
    // or stopped / cancelled via streamer / generation_stream->stop() / generation_stream->cancel()
    std::vector<Sequence::CPtr> get_finished_sequences() const {
        std::vector<Sequence::CPtr> finished_seqs;
        finished_seqs.reserve(num_total_seqs());

        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->has_finished() || m_sequences[seq_id]->out_of_memory() || handle_stopped() || handle_cancelled()) {
                finished_seqs.push_back(m_sequences[seq_id]);
            }
        }

        OPENVINO_ASSERT(finished_seqs.size() == num_total_seqs(), "Internal error: get_finished_sequences() must be called when all sequences are "
            "either finished / ignored by OOM or dropped via GenerationStream::stop() / GenerationStream::cancel()");

        std::sort(finished_seqs.begin(), finished_seqs.end(), [=] (Sequence::CPtr s1, Sequence::CPtr s2) -> bool {
            bool is_beam_search = m_sampling_params.is_beam_search();
            const float score_1 = is_beam_search ? s1->get_beam_search_score(m_sampling_params) : s1->get_cumulative_log_prob();
            const float score_2 = is_beam_search ? s2->get_beam_search_score(m_sampling_params) : s2->get_cumulative_log_prob();
            return score_1 > score_2;
        });

        return finished_seqs;
    }

    // returns running or waiting sequences
    std::vector<Sequence::Ptr> get_not_finished_sequences() {
        std::vector<Sequence::Ptr> running_seqs;
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (!m_sequences[seq_id]->has_finished()) {
                running_seqs.emplace_back(m_sequences[seq_id]);
            }
        }

        return running_seqs;
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

    size_t get_output_seq_len() const {
        return m_output_seq_len;
    }

    void set_output_seq_len(size_t len) {
        m_output_seq_len = len;
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
        return get_context_len() >= get_prompt_len() && get_context_len() > m_max_content_len && get_max_new_tokens() > 0;
    }

    void schedule_tokens(size_t num_tokens) {
        m_num_scheduled_tokens = num_tokens;
        // Unless otherwise specified, the sampler will process all scheduled tokens.
        m_output_seq_len = num_tokens;
    }

    void clear_scheduled_tokens() {
        m_num_scheduled_tokens = 0;
        m_num_validation_tokens = 0;
        m_output_seq_len = 0;
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
    
    void set_stream_window_size(size_t k) {
        m_stream_window_size = k;
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

    const std::vector<std::vector<float>>& get_input_embeds() const {
        OPENVINO_ASSERT(m_sequence_group_type == SequenceGroupType::EMBEDDINGS);
        return m_input_embeds;
    }

    std::optional<std::vector<int64_t>> get_token_type_ids() const {
        return m_token_type_ids;
    }

    size_t get_hidden_size() const {
        OPENVINO_ASSERT(m_sequence_group_type == SequenceGroupType::EMBEDDINGS);
        OPENVINO_ASSERT(m_input_embeds.size() > 0, "Embeddings should be set to get hidden size.");
        return m_input_embeds[0].size();
    }

    void append_prompt_log_prob(float log_prob) {
        m_prompt_log_probs.push_back(log_prob);
    }

    SequenceGroupType get_sequence_group_type() const {
        return m_sequence_group_type;
    }

    /**
     * @return The number of tokens for which KV cache has been filled already (i.e. not including scheduled tokens).
     */
    size_t get_num_cached_tokens() const {
        OPENVINO_ASSERT(get_num_processed_tokens() >= get_num_evicted_tokens());
        return (get_num_processed_tokens() - get_num_evicted_tokens());
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
        auto forked_sequence = Sequence::fork(sequence, m_next_sequence_id++);
        m_sequences.emplace_back(forked_sequence);
        return forked_sequence;
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

    bool is_waiting() const {
        for (size_t seq_id = 0; seq_id < m_sequences.size(); ++seq_id) {
            if (m_sequences[seq_id]->is_waiting()) {
                return true;
            }
        }
        return m_is_gen_paused;
    }

    GenerationStream::Ptr get_generation_stream() {
        return m_generation_stream;
    }

    void set_generation_status(GenerationStatus status) {
        m_generation_stream->set_generation_status(status);
    }

    bool handle_stopped() const {
        return m_generation_stream->get_status() == GenerationStatus::STOP;
    }

    bool handle_cancelled() const {
        return m_generation_stream->get_status() == GenerationStatus::CANCEL;
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
            output.score = m_sampling_params.is_beam_search() ? sequence->get_beam_search_score(m_sampling_params) : sequence->get_cumulative_log_prob();
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
            auto output = sequence->get_last_generation_output(token_cnt, m_stream_window_size);
            if (m_sampling_params.echo && !m_has_echoed) {
                output.generated_ids.insert(output.generated_ids.begin(), m_prompt_ids.begin(), m_prompt_ids.end());
                output.generated_log_probs.insert(output.generated_log_probs.begin(), m_prompt_log_probs.begin(), m_prompt_log_probs.end());
            }
            outputs.emplace(sequence->get_grouped_id(), output);
        }
        m_has_echoed = true;
        m_generation_stream->push(std::move(outputs));
    }

    void notify_handle() {
        if (out_of_memory()) {
            set_generation_status(GenerationStatus::IGNORED);
        } else if (has_finished()) {
            set_generation_status(GenerationStatus::FINISHED);
        }
        // For beam search streaming is not available, so we notify only upon finishing
        if (m_sampling_params.is_beam_search()) {
            if (has_finished()) {
                push_outputs();
            }
        } else if (m_sampling_params.is_greedy_decoding() || m_sampling_params.is_multinomial()) {
            // We can stream only when one sequence is returned and we don't use stop strings that would be excluded from the output
            // (after stop string is detected its tokens are already sent)
            if (num_total_seqs() == 1) {
                const auto generated_len = m_sequences.front()->get_generated_len();
                if (has_finished()) {
                    m_stream_window_size = 0;
                }
                // push empty output in case we won't stream generation res
                if (generated_len <= (m_num_streamed_tokens + m_stream_window_size)) {
                    if (has_finished()) {
                        push_empty_outputs();
                    }
                    return;
                }
                // speculative decoding draft handling
                if (generated_len < m_num_streamed_tokens) {
                    m_num_streamed_tokens = generated_len;
                }
                OPENVINO_ASSERT(generated_len >= (m_num_streamed_tokens + m_stream_window_size));
                size_t num_output_token_to_push = generated_len - m_num_streamed_tokens - m_stream_window_size;
                push_partial_outputs(num_output_token_to_push);
                m_num_streamed_tokens += (num_output_token_to_push);
            } else if (has_finished()) {
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

    size_t get_max_new_tokens() const {
        return m_sampling_params.get_max_new_tokens(get_prompt_len());
    }
};

inline std::shared_ptr<SequenceGroup> Sequence::get_sequence_group_ptr() const {
    assert(m_sequence_group != nullptr);
    return m_sequence_group->shared_from_this();
}

}
