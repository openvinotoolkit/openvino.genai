// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

namespace {

template <typename T>
void print_array(T * array, size_t size) {
    std::cout << " => [ ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << " ] " << std::endl;
}

void print_tensor(std::string name, ov::Tensor tensor) {
    std::cout << name;
    if (tensor.get_element_type() == ov::element::i32) {
        print_array(tensor.data<int>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::i64) {
        print_array(tensor.data<int64_t>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::f32) {
        print_array(tensor.data<float>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::boolean) {
        print_array(tensor.data<bool>(), tensor.get_size());
    }
}


constexpr size_t BATCH_SIZE = 1;

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string prompt) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

std::string detokenize(ov::InferRequest& detokenizer, ov::Tensor tokens) {
    detokenizer.set_input_tensor(tokens);
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
	    return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};

constexpr size_t BLOCK_SIZE = 16;

// TODO: extract from the model
constexpr int64_t SPECIAL_EOS_TOKEN = 2; // llm_model->get_rt_info()["eos_token_id"].as<int64_t>();

// TODO: compute based on the available memory
constexpr size_t NUM_BLOCKS = 3640;

struct KVCacheBlock {
    int m_ref_count;
    int m_index;
};

enum class StopCriteria {early, heuristic, never};

class Sequence;

struct SamplingParameters {
    // Generic
    size_t max_new_tokens = 20;
    bool ignore_eos = false;
    int64_t eos_token = 2; // There's no way to extract special token values from the tokenizer for now

    // Beam search specific
    size_t n_groups = 1;
    size_t group_size = 1; // beam_width
    float diversity_penalty = 1.0f;
    StopCriteria stop_criteria = StopCriteria::heuristic;
    float length_penalty = 1.0f;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    std::function<bool(const Sequence&)> early_finish = [](const Sequence&){return false;};

    // Multinomial
    float temperature = 0.0f; // by default we use greedy sampling
    int top_k = -1; // maybe to assign vocab_size ?
    float top_p = 1.0f; // by default convsider all tokens

    static SamplingParameters greedy() {
        SamplingParameters greedy_params;
        greedy_params.temperature = 0.0f;
        greedy_params.ignore_eos = true;
        return greedy_params;
    }

    static SamplingParameters beam_search() {
        SamplingParameters beam_search;
        beam_search.n_groups = 3;
        beam_search.group_size = 5;
        return beam_search;
    }

    static SamplingParameters multimomial() {
        SamplingParameters multimomial;
        multimomial.temperature = 0.8f;
        multimomial.top_p = 0.8;
        multimomial.top_k = 20;
        return multimomial;
    }
};

enum class SequenceStatus {
    WAITING = 0,
    FINISHED = 1
};

using TokenIds = std::vector<int64_t>;

class Sequence {
    TokenIds m_prompt_ids;
    TokenIds m_generated_ids;
    std::vector<KVCacheBlock> m_blocks;
    uint64_t m_sequence_id = _get_next_sequence_id();
    SequenceStatus m_status = SequenceStatus::WAITING;

    // amount of processed tokens, e.g. prompt can be processed using multiple consequence inferences
    // so, we need to track which part of the prompt we have already processed
    size_t m_num_processed_tokens = 0;
    // a number of scheduled tokens by Scheduler::schedule logic
    size_t m_num_scheduled_tokens = 0;
    // a max output length for generation to use as a stopping criteria
    size_t m_max_output_length = std::numeric_limits<size_t>::max();

    int64_t _get_position_id() const {
        return get_context_len() - 1;
    }

    int64_t _get_slot_id(int64_t token_idx) const {
        int block_id = token_idx / BLOCK_SIZE, block_offset = (token_idx % BLOCK_SIZE);
        return m_blocks[block_id].m_index * BLOCK_SIZE + block_offset;
    }

    static uint64_t _get_next_sequence_id() {
        static uint64_t m_counter = 0;
        return m_counter++;
    }

public:
    explicit Sequence(const TokenIds& prompt_ids)
        : m_prompt_ids(prompt_ids) {
    }

    explicit Sequence(const ov::Tensor& prompt_ids) {
        m_prompt_ids.reserve(prompt_ids.get_size());
        std::copy(prompt_ids.data<int64_t>(), prompt_ids.data<int64_t>() + prompt_ids.get_size(), std::back_inserter(m_prompt_ids));
    }

    bool operator ==(const Sequence& other) const {
        return other.m_sequence_id == m_sequence_id;
    }

    uint64_t get_id() const {
        return m_sequence_id;
    }

    // appends new tokens to a generated part
    void append_token(int64_t token_id) {
        m_generated_ids.push_back(token_id);
    }

    // total number of tokens in a sequence
    size_t get_context_len() const {
        return m_prompt_ids.size() + m_generated_ids.size();
    }

    // a number of blocks to hold tokens of current sequence in KV cache
    size_t get_num_blocks() const {
        return (get_context_len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    // are we still processing a prompt phase?
    bool is_prompt_phase() const {
        return m_generated_ids.empty();
    }

    bool requires_sampling() const {
        return get_num_processed_tokens() + get_num_scheduled_tokens() >= get_context_len(); 
    }

    bool requires_new_block() const {
        // if next token generation requires new block allocation
        return m_blocks.size() * BLOCK_SIZE <= get_context_len();
    }

    void append_new_block(KVCacheBlock block) {
        m_blocks.push_back(block);
    }

    void schedule_tokens(size_t num_tokens) {
        m_num_scheduled_tokens = num_tokens;
    }

    bool is_scheduled() const {
        return m_num_scheduled_tokens > 0;
    }

    size_t get_num_scheduled_tokens() const {
        return m_num_scheduled_tokens;
    }

    int64_t get_slot_id(size_t logical_token_id) const {
        size_t physical_block_id = logical_token_id / BLOCK_SIZE;
        size_t block_offset = logical_token_id % BLOCK_SIZE;
        return BLOCK_SIZE * m_blocks[physical_block_id].m_index + block_offset;
    }

    // mark current schedule phase as finished and updates internal counters
    void finish_iteration() {
        m_num_processed_tokens += m_num_scheduled_tokens;
        m_num_scheduled_tokens = 0;
    }

    size_t get_num_processed_tokens() const {
        return m_num_processed_tokens;
    }

    size_t get_num_available_tokens_for_batching() const {
        OPENVINO_ASSERT(m_num_scheduled_tokens == 0);
        return get_context_len() - m_num_processed_tokens;
    }

    bool has_finished() const {
        return m_max_output_length == m_generated_ids.size() ||
            (!m_generated_ids.empty() && m_generated_ids.back() == SPECIAL_EOS_TOKEN);
    }

    // get input_id by token_id within a sequence
    int64_t operator[] (size_t token_id) const {
        return token_id < m_prompt_ids.size() ?
            m_prompt_ids[token_id] : m_generated_ids[token_id - m_prompt_ids.size()];
    }

    const std::vector<KVCacheBlock> & get_kv_blocks() const {
        return m_blocks;
    }

    const TokenIds & get_prompt_ids() const {
        return m_prompt_ids;
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
public:
    SequenceGroup(uint64_t request_id, const Sequence& sequence, const SamplingParameters& sampling_params)
        : m_request_id(request_id),
          m_sampling_params(sampling_params) {
        add_sequence(sequence);
    }

    SequenceGroup(uint64_t request_id, const TokenIds& input_ids, const SamplingParameters& sampling_params) :
        SequenceGroup(request_id, Sequence(input_ids), sampling_params) { }

    SequenceGroup(uint64_t request_id, const ov::Tensor& input_ids, const SamplingParameters& sampling_params) :
        SequenceGroup(request_id, Sequence(input_ids), sampling_params) { }

    void add_sequence(const Sequence & sequence) {
        m_sequences.push_back(sequence);
    }

    void remove_sequence(uint64_t sequence_id) {
        OPENVINO_ASSERT(std::remove_if(m_sequences.begin(), m_sequences.end(), [sequence_id] (const Sequence & seq) {
            return seq.get_id() == sequence_id;
        }) != m_sequences.end(), "Failed to remove sequence with specified ID");
    }

    const Sequence& operator[] (size_t index) const {
        OPENVINO_ASSERT(m_sequences.size() > index);
        return m_sequences[index];
    }

    Sequence& operator[] (size_t index) {
        OPENVINO_ASSERT(m_sequences.size() > index);
        return m_sequences[index];
    }

    size_t num_seqs() const {
        return m_sequences.size();
    }

    size_t num_finished_seqs() const {
        return std::count_if(m_sequences.begin(), m_sequences.end(), [] (const Sequence& seq) {
            return seq.has_finished();
        });
    }

    size_t num_unfinished_seqs() const {
        return num_seqs() - num_finished_seqs();
    }

    bool has_finished() const {
        return num_finished_seqs() == 0;
    }

    bool requires_new_block() const {
        return std::any_of(m_sequences.begin(), m_sequences.end(), [] (const Sequence & seq) {
            return seq.requires_new_block();
        });
    }

    bool is_prompt_phase() const {
        return num_unfinished_seqs() == 1 && m_sequences[0].is_prompt_phase();
    }

    size_t get_num_available_tokens_for_batching() const {
        size_t _num_unfinished_seqs = num_unfinished_seqs();
        OPENVINO_ASSERT(_num_unfinished_seqs > 0);
        const Sequence & first_sequence = m_sequences[0];

        return first_sequence.is_prompt_phase() ? first_sequence.get_num_available_tokens_for_batching() :
                                                  _num_unfinished_seqs;
    }

    void schedule_tokens(size_t) {
        // TODO: move prompt handling to SequenceGroup
    }

    uint64_t get_request_id() const {
        return m_request_id;
    }
};

class BlockManager {
    std::vector<KVCacheBlock> m_blocks;
public:
    BlockManager(int num_blocks) {
        m_blocks.reserve(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            m_blocks.push_back({0, i});
        }
    }

    bool can_allocate() const {
        return !m_blocks.empty();
    }

    KVCacheBlock allocate() {
        OPENVINO_ASSERT(can_allocate());
        auto allocated_block = m_blocks.back();
        m_blocks.pop_back();
        return allocated_block;
    }
};

struct SchedulerConfig {
    // a maximum number of tokens to batch
    // (in constrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
    // TODO: benchmark this value and understand a required value to ensure inference is not memory bound
    const size_t max_tokens_to_batch = 16;

    // total number of KV blocks available to scheduler logic
    const size_t num_kv_blocks = NUM_BLOCKS;
};

class Scheduler {
    SchedulerConfig m_config;
    BlockManager m_block_manager;
public:
    Scheduler(const SchedulerConfig & config = {}) :
        m_config(config), m_block_manager(m_config.num_kv_blocks) { }

    void schedule(std::vector<SequenceGroup>& sequence_groups) {
        for (size_t i = 0, current_num_of_scheduled_tokens = 0;
            i < sequence_groups.size() && current_num_of_scheduled_tokens < m_config.max_tokens_to_batch; ++i) {
            SequenceGroup & sequence_group = sequence_groups[i];

            if (!sequence_group.has_finished()) {
                // check first whether new blocks are available for next tokens processing
                // TODO: implement more complex logic to handle:
                // 1. sequence can process tokens within its currently available slots
                // 2. sequence KV blocks can be evicted by sequence with higher priority
                // 3. num of available KV blocks can be less of num tokens available for generation
                while (sequence_group.requires_new_block()) {
                    for (size_t i = 0; i < sequence_group.num_seqs(); ++i) {
                        sequence_group[i].append_new_block(m_block_manager.allocate());
                    }
                }

                size_t num_batch_available_tokens = m_config.max_tokens_to_batch - current_num_of_scheduled_tokens;
                size_t num_seq_available_tokens = sequence_group.get_num_available_tokens_for_batching();

                // schedule all bare minimum of tokens from current sequence to fill up a batch!
                size_t num_scheduled_tokens = std::min(num_batch_available_tokens, num_seq_available_tokens);
                sequence_group.schedule_tokens(num_scheduled_tokens);

                current_num_of_scheduled_tokens += num_scheduled_tokens;
            }
        }
    }
};

class CacheManager {
    std::vector<ov::Tensor> m_key_cache;
    std::vector<ov::Tensor> m_value_cache;

public:
    CacheManager() {
        // TODO: make as a parameter
        constexpr auto kv_cache_precision = ov::element::f32;

        const size_t BLOCK_SIZE = 16, X = BLOCK_SIZE / kv_cache_precision.size();
        // TODO: take from model
        constexpr size_t NUM_KV_HEADS = 12, NUM_HEADS = 12, HIDDEN_DIMS = 768, HEAD_SIZE = HIDDEN_DIMS / NUM_HEADS;
        constexpr size_t NUM_DECODER_LAYERS = 12; // num KV cache pairs

        // Set PagedAttention specific parameters

        const ov::Shape k_cache_shape{NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X};
        const ov::Shape v_cache_shape{NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE};

        for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
            m_key_cache[decoder_layer_id] = ov::Tensor(kv_cache_precision, k_cache_shape);
            m_value_cache[decoder_layer_id] = ov::Tensor(kv_cache_precision, v_cache_shape);
        }
    }

    size_t get_num_layers() const {
        return m_key_cache.size();
    }

    ov::Tensor get_key_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size());
        return m_key_cache[decoder_layer_id];
    }

    ov::Tensor get_value_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size());
        return m_value_cache[decoder_layer_id];
    }

    // TODO: implement methods to copy / fork blocks by ID
};

class ModelRunner {
    CacheManager m_cache_manager;
    ov::InferRequest & m_request;
public:
    ModelRunner(ov::InferRequest & request) :
        m_request(request) {
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_cache_manager.get_num_layers(); ++decoder_layer_id) {
            m_request.set_input_tensor(2 + decoder_layer_id * 2, m_cache_manager.get_key_cache(decoder_layer_id));
            m_request.set_input_tensor(2 + decoder_layer_id * 2 + 1, m_cache_manager.get_value_cache(decoder_layer_id));
        }
    }

    ov::Tensor step(const std::vector<SequenceGroup> & sequence_groups) {
        size_t batch_size = 0, max_num_blocks = 0, max_context_len_value = 0;
        // since we merge sequence_len and batch to avoid ragged dimensions => batch dimension contains all tokens, while seq len is 1
        const size_t seq_len = 1;

        // compute aggregated values
        for (size_t i = 0; i < sequence_groups.size(); ++i) {
            const SequenceGroup & sequence_group = sequence_groups[i];
            // TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            const Sequence & sequence = sequence_group[0];

            if (sequence_group[0].is_scheduled()) {
                batch_size += sequence.get_num_scheduled_tokens();
                max_num_blocks = std::max(max_num_blocks, sequence.get_num_blocks());
                // TODO: compute in the cycle below
                max_context_len_value = std::max(max_context_len_value, sequence.get_num_processed_tokens() + 1);
            } else {
                OPENVINO_ASSERT(sequence.get_num_scheduled_tokens() == 0);
            }
        }

        ov::Tensor
            input_ids(ov::element::i64, {batch_size, seq_len}),
            position_ids(ov::element::i64, {batch_size, seq_len}),
            is_prompt(ov::element::boolean, {}),
            max_context_len(ov::element::i64, {}),
            slot_mapping(ov::element::i64, {batch_size, seq_len}),
            context_lens(ov::element::i64, {batch_size}),
            block_tables(ov::element::i32, {batch_size, max_num_blocks});

        max_context_len.data<int64_t>()[0] = max_context_len_value;
        // we don't differentiate prefill and generate phases
        is_prompt.data<bool>()[0] = false;

        // get raw pointers to copy to
        int64_t
            * input_ids_data = input_ids.data<int64_t>(),
            * position_ids_data = position_ids.data<int64_t>(),
            * slot_mapping_data = slot_mapping.data<int64_t>(),
            * context_lens_data = context_lens.data<int64_t>();
        int32_t
            * block_tables_data = block_tables.data<int32_t>();

        for (size_t i = 0; i < sequence_groups.size(); ++i) {
            // TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            const Sequence & sequence = sequence_groups[i][0];
            if (sequence.is_scheduled()) {
                size_t num_scheduled_tokens = sequence.get_num_scheduled_tokens();
                size_t position_id = sequence.get_num_processed_tokens(), context_len = position_id + 1;

                for (size_t token_id = 0; token_id < num_scheduled_tokens; ++token_id, ++position_id, ++context_len) {
                    input_ids_data[token_id] = sequence[position_id];
                    position_ids_data[token_id] = position_id;
                    slot_mapping_data[token_id] = sequence.get_slot_id(position_id);
                    context_lens_data[token_id] = context_len;

                    const std::vector<KVCacheBlock> & kv_blocks = sequence.get_kv_blocks();
                    for (size_t logical_block_id = 0; logical_block_id < sequence.get_num_blocks(); ++logical_block_id)
                        block_tables_data[logical_block_id] = kv_blocks[logical_block_id].m_index;
                    block_tables_data += max_num_blocks;
                }

                // apply strides to shift to next sequence
                input_ids_data += num_scheduled_tokens;
                position_ids_data += num_scheduled_tokens;
                slot_mapping_data += num_scheduled_tokens;
                context_lens_data += num_scheduled_tokens;
            }
        }

        // typical LLM parameters
        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("position_ids", position_ids);

        // PagedAttention specific parameetrs
        m_request.set_tensor("is_prompt", is_prompt);
        m_request.set_tensor("slot_mapping", slot_mapping);
        m_request.set_tensor("max_context_len", max_context_len);
        m_request.set_tensor("context_lens", context_lens);
        m_request.set_tensor("block_tables", block_tables);

        // print_tensor("input_ids", input_ids);
        // print_tensor("position_ids", position_ids);

        // print_tensor("is_prompt", is_prompt);
        // print_tensor("slot_mapping", slot_mapping);
        // print_tensor("max_context_len", max_context_len);
        // print_tensor("context_lens", context_lens);
        // print_tensor("block_tables", block_tables);

        m_request.infer();

        // return logits
        return m_request.get_output_tensor();
    }
};

bool has_unfinished_groups(const std::vector<SequenceGroup> & sequence_groups) {
    for (auto & sequence_group : sequence_groups) {
        if (!sequence_group.has_finished())
            return true;
    }

    return false;
}

class Sampler {
    SamplingParameters m_parameters;

    int64_t _greedy_sample(const float * logits_data, size_t vocab_size) const {
        // currently, greedy search is used
        // TODO: apply m_config
        int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
        return out_token;
    }

public:
    Sampler(const SamplingParameters & parameters = {}) :
        m_parameters(parameters) { }

    void decode(std::vector<SequenceGroup> & sequence_groups, ov::Tensor logits) const {
        const float * logits_data = logits.data<float>();
        ov::Shape logits_shape = logits.get_shape();
        OPENVINO_ASSERT(logits_shape.size() == 3);
        size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2], logits_stride = seq_len * vocab_size;
        OPENVINO_ASSERT(seq_len == 1);

        for (size_t i = 0, current_token_id = 0; i < sequence_groups.size(); ++i) {
            // TODO: process multuple sequences within a group
            Sequence & sequence = sequence_groups[i][0];

            if (sequence.is_scheduled()) {
                current_token_id += sequence.get_num_scheduled_tokens();

                if (sequence.requires_sampling()) {
                    int64_t sampled_token_id = _greedy_sample(logits_data + logits_stride * (current_token_id - 1), vocab_size);
                    sequence.append_token(sampled_token_id);
                } else {
                    // we are in prompt processing phase when prompt is split into chunks and processed step by step
                }

                // update internal state of sequence to reset scheduler tokens and update currently processed onces
                sequence.finish_iteration();
            }
        }
    }
};

struct GenerationResult {
    // request ID
    uint64_t m_request_id;
    // in a generic case we have multiple generation results per initial prompt
    // depending on sampling parameters (e.g. beam search or parallel sampling)
    std::vector<TokenIds> m_generation_ids;
    // score (cumulative logprob)
    float m_cumulative_logprob;

    static GenerationResult from_sequence_group(const SequenceGroup& sequence_group) {
        GenerationResult result;
        result.m_request_id = sequence_group.get_request_id();

        for (size_t sequence_id = 0; sequence_id < sequence_group.num_finished_seqs(); ++sequence_id) {
            result.m_generation_ids.push_back(sequence_group[sequence_id].get_generated_ids());
        }

        // TODO: track this information
        result.m_cumulative_logprob = 0.0f;

        return result;
    }
};

class LLMEngine {
    Scheduler m_scheduler;
    ModelRunner m_model_runner;
    Sampler m_sampler;

    // current requests to process
    std::vector<SequenceGroup> m_requests;
public:
    LLMEngine(ov::InferRequest& request,
              const SchedulerConfig& scheduler_config)
        : m_scheduler(scheduler_config),
          m_model_runner(request),
          m_sampler() {
    }

    void add_request(uint64_t request_id, const TokenIds input_ids, SamplingParameters sampling_params) {
        SequenceGroup sequence_group(request_id, input_ids, sampling_params);
        m_requests.push_back(sequence_group);
    }

    void add_request(uint64_t request_id, const ov::Tensor input_ids, SamplingParameters sampling_params) {
        SequenceGroup sequence_group(request_id, input_ids, sampling_params);
        m_requests.push_back(sequence_group);
    }

    // TODO: current function must return information about currentl generation step
    // e.g. what requests has generation phase, what are ignored, etc
    void step() {
        m_scheduler.schedule(m_requests);
        ov::Tensor logits = m_model_runner.step(m_requests);
        m_sampler.decode(m_requests, logits);
    }

    bool has_unifnished_requests() const {
        for (auto & sequence_group : m_requests) {
            if (!sequence_group.has_finished())
                return true;
        }

        return false;
    }

    void free_finished_groups() {
        std::remove_if(m_requests.begin(), m_requests.end(), [] (const SequenceGroup& seq_group) {
            return seq_group.has_finished();
        });
    }

    // more high level interface
    std::vector<GenerationResult> generate(const std::vector<ov::Tensor> prompts, std::vector<SamplingParameters> samplig_params) {
        OPENVINO_ASSERT(prompts.size() == samplig_params.size());

        for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
            add_request(request_id, prompts[request_id], samplig_params[request_id]);
        }

        while (has_unifnished_requests()) {
            // TODO: each step should return output to track generation process and fill
            step();
        }

        // create generation results
        std::vector<GenerationResult> results;
        results.reserve(m_requests.size());

        // TODO: fill values in the order to preserve request ID
        for (size_t i = 0; i < m_requests.size(); ++i) {
            results.push_back(GenerationResult::from_sequence_group(m_requests[i]));
        }

        // free KV blocks and other resourses
        free_finished_groups();

        return results;
    }
};

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }

    //
    // Compile models
    //

    ov::Core core;
    core.add_extension("libuser_ov_extensions.so");
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    // The model can be compiled for GPU as well
    std::shared_ptr<ov::Model> model = core.read_model(std::string{argv[1]} + "/vllm_optimum_openvino_model.xml");
    ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();

    //
    // Create requests for generation
    //

    const size_t dataset_size = 30;

    std::vector<std::string> prompt_examples = {
        "What is OpenVINO?",
        "How are you?",
        "What is the current time",
        "What is OpenVINO?",
    };

    std::vector<SamplingParameters> sampling_params_examples {
        SamplingParameters::greedy(),
        SamplingParameters::multimomial(),
        SamplingParameters::beam_search()
    };

    std::vector<ov::Tensor> input_ids;
    std::vector<SamplingParameters> sampling_params;

    input_ids.reserve(dataset_size);
    sampling_params.reserve(dataset_size);

    for (size_t request_id = 0; request_id < dataset_size; ++request_id) {
        auto [_input_ids, _attention_mask] = tokenize(tokenizer, prompt_examples[request_id % prompt_examples.size()]);
        input_ids.push_back(_input_ids);
        sampling_params.push_back(sampling_params_examples[request_id % sampling_params_examples.size()]);
    }

    //
    // Perform the first inference
    //

    SchedulerConfig scheduler_config {
        .max_tokens_to_batch = 16,
        .num_kv_blocks = NUM_BLOCKS
    };

    LLMEngine engine(request, scheduler_config);
    std::vector<GenerationResult> generation_results = engine.generate(input_ids, sampling_params);

    for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
        const GenerationResult & generation_result = generation_results[request_id];

        std::cout << "Question: " << detokenize(detokenizer, input_ids[request_id]) << std::endl;
        for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
            std::cout << "Answer " << output_id << ": " << detokenize(detokenizer, generation_result.m_generation_ids[output_id]) << std::endl;
        }
        std::cout << std::endl;
    }

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
