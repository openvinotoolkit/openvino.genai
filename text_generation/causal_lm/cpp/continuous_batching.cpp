// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

namespace {

constexpr size_t BATCH_SIZE = 1;

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
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

int64_t get_next_token_id(ov::Tensor logits) {
    ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2];
    float* logits_data = logits.data<float>() + (seq_len - 1) * vocab_size;
    int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
    return out_token;
}

constexpr size_t BLOCK_SIZE = 16;

struct KVCacheBlock {
    int m_ref_count;
    int m_index;
};

class Sequence {
    std::vector<int64_t> m_prompt_ids;
    std::vector<int64_t> m_generated_ids;
    std::vector<KVCacheBlock> m_blocks;

    int64_t _get_position_id() const {
        return get_context_len() - 1;
    }

    int64_t _get_slot_id(int64_t token_idx) const {
        int block_id = token_idx / BLOCK_SIZE, block_offset = (token_idx % BLOCK_SIZE);
        return m_blocks[block_id].m_index * BLOCK_SIZE + block_offset;
    }

public:
    Sequence(ov::Tensor prompt_ids) {
        m_prompt_ids.reserve(prompt_ids.get_size());
        std::copy_n(prompt_ids.data<int64_t>(), prompt_ids.get_size(), std::back_inserter(m_prompt_ids));
    }

    void append_token(int64_t token_id) {
        m_generated_ids.push_back(token_id);
    }

    size_t get_context_len() const {
        return m_prompt_ids.size() + m_generated_ids.size();
    }

    size_t get_num_blocks() const {
        return (get_context_len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    bool is_prompt_phase() const {
        return m_generated_ids.empty();
    }

    ov::Tensor get_slot_mapping() const {
        bool is_prompt = is_prompt_phase();
        ov::Tensor slot_mapping_tensor(ov::element::i64, {BATCH_SIZE, is_prompt ? get_context_len() : 1});
        int64_t * slot_mapping_data = slot_mapping_tensor.data<int64_t>();
        if (is_prompt) {
            for (size_t i = 0; i < get_context_len(); ++i) {
                slot_mapping_data[i] = _get_slot_id(i);
            }
        } else {
            slot_mapping_data[0] = _get_slot_id(_get_position_id());
        }

        return slot_mapping_tensor;
    }

    ov::Tensor get_blocks_tensor() const {
        int num_blocks = get_num_blocks();
        ov::Tensor blocks_tensor(ov::element::i32, {BATCH_SIZE, get_num_blocks()});
        int * blocks_data = blocks_tensor.data<int>();
        for (size_t i = 0; i < num_blocks; ++i) {
            blocks_data[i] = m_blocks[i].m_index;
        }

        return blocks_tensor;
    }

    ov::Tensor get_context_lens_tensor() const {
        ov::Tensor context_lens_tensor(ov::element::i64, {BATCH_SIZE});
        context_lens_tensor.data<int64_t>()[0] = get_context_len();
        return context_lens_tensor;
    }

    // for batch = 1 it returns current context lenght
    ov::Tensor get_max_content_len_tensor() const {
        ov::Tensor max_context_len_tensor(ov::element::i64, {});
        max_context_len_tensor.data<int64_t>()[0] = get_context_len();
        return max_context_len_tensor;
    }

    bool requires_new_block() const {
        // if next token generation requires new block allocation
        return m_blocks.size() * BLOCK_SIZE <= get_context_len();
    }

    void append_new_block(KVCacheBlock block) {
        m_blocks.push_back(block);
    }

    ov::Tensor get_input_ids() const {
        bool is_prompt = is_prompt_phase();
        ov::Tensor input_ids(ov::element::i64, {1, is_prompt ? get_context_len() : 1});
        int64_t* input_ids_data = input_ids.data<int64_t>();

        if (is_prompt) {
            std::copy_n(m_prompt_ids.begin(), m_prompt_ids.size(), input_ids_data);
        } else {
            input_ids_data[0] = m_generated_ids.back();
        }

        return input_ids;
    }

    ov::Tensor get_is_prompt() const {
        ov::Tensor is_prompt_tensor(ov::element::boolean, {});
        is_prompt_tensor.data<bool>()[0] = is_prompt_phase();
        return is_prompt_tensor;
    }

    ov::Tensor get_position_ids() const {
        bool is_prompt = is_prompt_phase();
        ov::Tensor position_ids(ov::element::i64, {1, is_prompt ? get_context_len() : 1});
        int64_t * position_ids_data = position_ids.data<int64_t>();
        if (is_prompt) {
            std::iota(position_ids_data, position_ids_data + position_ids.get_size(), 0);
        } else {
            position_ids_data[0] = _get_position_id();
        }
        return position_ids;
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
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[2]);
    ov::InferRequest detokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    // The model can be compiled for GPU as well
    std::shared_ptr<ov::Model> llm_model = core.read_model(std::string{argv[1]} + "/vllm_optimum_openvino_model.xml");
    ov::InferRequest lm = core.compile_model(llm_model, "CPU").create_infer_request();

    TextStreamer text_streamer{std::move(detokenizer)};

    //
    // Constants
    //

    // TODO: extract from model
    constexpr auto model_precision = ov::element::f32;
    constexpr auto kv_cache_precision = ov::element::f32;

    const int64_t SPECIAL_EOS_TOKEN = 2; // llm_model->get_rt_info()["eos_token_id"].as<int64_t>();
    const size_t BLOCK_SIZE = 16, X = BLOCK_SIZE / model_precision.size();
    // TODO: take from model
    constexpr size_t NUM_KV_HEADS = 12, NUM_HEADS = 12, HIDDEN_DIMS = 768, HEAD_SIZE = HIDDEN_DIMS / NUM_HEADS;
    constexpr size_t NUM_DECODER_LAYERS = 12; // num KV cache pairs
    // TODO compute based on the available memory
    constexpr size_t NUM_BLOCKS = 3640;

    const ov::Shape k_cache_shape{NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X};
    const ov::Shape v_cache_shape{NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE};

    // Initialize KV cache

    std::vector<ov::Tensor> k_cache(NUM_DECODER_LAYERS), v_cache(NUM_DECODER_LAYERS);
    for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
        k_cache[decoder_layer_id] = ov::Tensor(kv_cache_precision, k_cache_shape);
        v_cache[decoder_layer_id] = ov::Tensor(kv_cache_precision, v_cache_shape);
    }

    // Set PagedAttention specific parameters

    for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
        lm.set_input_tensor(2 + decoder_layer_id * 2, k_cache[decoder_layer_id]);
        lm.set_input_tensor(2 + decoder_layer_id * 2 + 1, v_cache[decoder_layer_id]);
    }

    BlockManager block_manager(NUM_BLOCKS);

    // create current sequence
    Sequence sequence(input_ids);

    auto schedule_sequence = [&] () {
        // allocate blocks for current inference if required
        while (sequence.requires_new_block()) {
            sequence.append_new_block(block_manager.allocate());
        }

        lm.set_tensor("input_ids", sequence.get_input_ids());
        lm.set_tensor("position_ids", sequence.get_position_ids());

        lm.set_tensor("is_prompt", sequence.get_is_prompt());
        lm.set_tensor("slot_mapping", sequence.get_slot_mapping());
        lm.set_tensor("max_context_len", sequence.get_max_content_len_tensor());
        lm.set_tensor("context_lens", sequence.get_context_lens_tensor());
        lm.set_tensor("block_tables", sequence.get_blocks_tensor());
    };

    //
    // Perform the first inference
    //

    schedule_sequence();
    lm.infer();
    int64_t out_token = get_next_token_id(lm.get_tensor("logits"));

    while (out_token != SPECIAL_EOS_TOKEN) {
        // std::cout << "out_token = " << out_token << std::endl;
        sequence.append_token(out_token);
        schedule_sequence();

        lm.start_async();
        text_streamer.put(out_token);
        lm.wait();

        // perform decoding
        out_token = get_next_token_id(lm.get_tensor("logits"));

        static int iterations = 0;
        if (iterations++ > 50)
            break;
    }
    text_streamer.end();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
