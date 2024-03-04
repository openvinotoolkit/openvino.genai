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

class Sequence {
    std::vector<int64_t> m_prompt_ids;
    std::vector<int64_t> m_generated_ids;
    std::vector<KVCacheBlock> m_blocks;

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

public:
    Sequence(ov::Tensor prompt_ids, size_t max_output_length = 32) :
        m_max_output_length(max_output_length) {
        m_prompt_ids.reserve(prompt_ids.get_size());
        std::copy_n(prompt_ids.data<int64_t>(), prompt_ids.get_size(), std::back_inserter(m_prompt_ids));
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

    const std::vector<int64_t> & get_prompt_ids() const {
        return m_prompt_ids;
    }

    const std::vector<int64_t> & get_generated_ids() const {
        return m_generated_ids;
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

struct SamplerConfig {
    // TODO: fill with parameters like num_beads, repetition_penalty, etc
};

class Scheduler {
    SchedulerConfig m_config;
    BlockManager m_block_manager;
public:
    Scheduler(const SchedulerConfig & config = {}) :
        m_config(config), m_block_manager(m_config.num_kv_blocks) { }

    void schedule(std::vector<Sequence>& sequences) {
        for (size_t i = 0, current_num_of_scheduled_tokens = 0;
            i < sequences.size() && current_num_of_scheduled_tokens < m_config.max_tokens_to_batch; ++i) {
            Sequence & sequence = sequences[i];

            if (!sequence.has_finished()) {
                // check first whether new blocks are available for next tokens processing
                // TODO: implement more complex logic to handle:
                // 1. sequence can process tokens within its currently available slots
                // 2. sequence KV blocks can be evicted by sequence with higher priority
                // 3. num of available KV blocks can be less of num tokens available for generation
                while (sequence.requires_new_block()) {
                    sequence.append_new_block(m_block_manager.allocate());
                }

                size_t num_batch_available_tokens = m_config.max_tokens_to_batch - current_num_of_scheduled_tokens;
                size_t num_seq_available_tokens = sequence.get_num_available_tokens_for_batching();

                // schedule all bare minimum of tokens from current sequence to fill up a batch!
                size_t num_scheduled_tokens = std::min(num_batch_available_tokens, num_seq_available_tokens);
                sequence.schedule_tokens(num_scheduled_tokens);

                current_num_of_scheduled_tokens += num_scheduled_tokens;
            }
        }
    }
};

class ModelRunner {
    ov::InferRequest & m_request;
public:
    ModelRunner(ov::InferRequest & request) :
        m_request(request) {
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
            m_request.set_input_tensor(2 + decoder_layer_id * 2, ov::Tensor(kv_cache_precision, k_cache_shape));
            m_request.set_input_tensor(2 + decoder_layer_id * 2 + 1, ov::Tensor(kv_cache_precision, v_cache_shape));
        }
    }

    ov::Tensor infer(const std::vector<Sequence> & sequences) {
        size_t batch_size = 0, max_num_blocks = 0, max_context_len_value = 0;
        // since we merge sequence_len and batch to avoid ragged dimensions => batch dimension contains all tokens, while seq len is 1
        const size_t seq_len = 1;

        // compute aggregated values
        for (size_t i = 0; i < sequences.size(); ++i) {
            const Sequence & sequence = sequences[i];

            if (sequence.is_scheduled()) {
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

        for (size_t i = 0; i < sequences.size(); ++i) {
            const Sequence & sequence = sequences[i];
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

bool has_unfinished_sequences(const std::vector<Sequence> & sequences) {
    for (auto & sequence : sequences) {
        if (!sequence.has_finished())
            return true;
    }

    return false;
}

class Sampler {
    SamplerConfig m_config;

    int64_t _get_next_token_id(const float * logits_data, size_t vocab_size) const {
        // currently, greedy search is used
        // TODO: apply m_config
        int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
        return out_token;
    }

public:
    Sampler(const SamplerConfig & config = {}) :
        m_config(config) { }

    void decode(std::vector<Sequence> & sequences, ov::Tensor logits) const {
        const float * logits_data = logits.data<float>();
        ov::Shape logits_shape = logits.get_shape();
        OPENVINO_ASSERT(logits_shape.size() == 3);
        size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2], logits_stride = seq_len * vocab_size;
        OPENVINO_ASSERT(seq_len == 1);

        for (size_t i = 0, current_token_id = 0; i < sequences.size(); ++i) {
            Sequence & sequence = sequences[i];

            if (sequence.is_scheduled()) {
                current_token_id += sequence.get_num_scheduled_tokens();

                if (sequence.requires_sampling()) {
                    int64_t sampled_token_id = _get_next_token_id(logits_data + logits_stride * (current_token_id - 1), vocab_size);
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
    // Create sequences
    //

    std::vector<std::string> prompts = {
        "What is OpenVINO?",
        "How are you?",
        "What is the current time",
        "What is OpenVINO?",
    };

    std::vector<Sequence> sequences;
    size_t dataset_size = 30;
    sequences.reserve(dataset_size);

    for (size_t i = 0; i < dataset_size; ++i) {
        auto [input_ids, attention_mask] = tokenize(tokenizer, prompts[i % prompts.size()]);
        sequences.push_back(Sequence(input_ids));
    }

    //
    // Perform the first inference
    //

    SchedulerConfig scheduler_config {
        .max_tokens_to_batch = 16,
        .num_kv_blocks = NUM_BLOCKS
    };

    Scheduler scheduler(scheduler_config);
    ModelRunner llm_model(request);
    Sampler sampler;

    while (has_unfinished_sequences(sequences)) {
        scheduler.schedule(sequences);
        ov::Tensor logits = llm_model.infer(sequences);
        sampler.decode(sequences, logits);
        std::cout << std::endl;
    }

    // print results
    for (size_t i = 0; i < sequences.size(); ++i) {
        const Sequence & sequence = sequences[i];
        std::cout << "Question: " << detokenize(detokenizer, sequence.get_prompt_ids()) << std::endl
                  << "Answer: " << detokenize(detokenizer, sequence.get_generated_ids()) << std::endl << std::endl;
    }

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
