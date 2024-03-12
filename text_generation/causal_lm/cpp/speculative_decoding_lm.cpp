// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cmath>
#include <random>

constexpr size_t BATCH_SIZE = 1;

// sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size], 
// threfore usually SEQ_LEN_AXIS = 2
constexpr size_t SEQ_LEN_AXIS = 2;

// There's no way to extract special token values from the detokenizer for now
constexpr int64_t SPECIAL_EOS_TOKEN = 2;

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
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
}

void forward_key_values(ov::InferRequest& request) {
    // Forwards present key/values from Results to Parameters with past key/values
    // The first Parameters are: input_ids, attention_masks, position_ids, other ones are past key/values.
    // The first Result is logits, others are present key.value.
    auto num_inputs = request.get_compiled_model().inputs().size();
    for (size_t idx = 3; idx < num_inputs; ++idx) {
        request.set_input_tensor(idx, request.get_output_tensor(idx - 2));
    }
}

void init_key_values(ov::InferRequest request) {
    auto num_inputs = request.get_compiled_model().inputs().size();
    for (size_t idx = 3; idx < num_inputs; ++idx) {
        auto kv_tensor = request.get_input_tensor(idx);
        ov::Shape tensor_shape = kv_tensor.get_shape();
        // tensor_shape[2] = 0;
        // kv_tensor.set_shape(tensor_shape);

        kv_tensor.set_shape({BATCH_SIZE, tensor_shape[1], 0, tensor_shape[3]});
    }
}

ov::Tensor trimm_tensor(ov::Tensor& tensor, uint64_t seq_len_axis, uint64_t new_seq_len) {
    // Copy elements from the old to a new tensor and return it.
    // It's assumed that key/values tensor has a shape [BATCH_SIZE, num_kv_heads, seq_len, head_size] or [seq_len, ...],
    // It that's not the case for your model please implement your own trim method.
    OPENVINO_ASSERT(seq_len_axis == 2 || seq_len_axis == 0, "Cannot trim key/values with sequence length axis = ", seq_len_axis);
    
    auto old_tensor_data = tensor.data<float>();
    auto shape = tensor.get_shape();
    size_t num_kv_heads = shape[1];
    size_t old_seq_len = shape[2];
    size_t head_size = shape[3];
    
    // if new_seq_len equal to old one no need to copy tensor, return as is
    if (old_seq_len == new_seq_len)
        return tensor;
    
    // if seq_len_axis is the very first dimension, this means data is contiguous, then trim by just setting the new shape
    if (seq_len_axis == 0) {
        shape[0] = new_seq_len;
        tensor.set_shape(shape);
        return tensor;
    }

    // if seq_len_axis == 2, then data is not contiguous, in order to trim need to repack tensor
    auto new_tensor = ov::Tensor{ov::element::f32, {BATCH_SIZE, num_kv_heads, new_seq_len, head_size}};
    auto new_tensor_data = new_tensor.data<float>();
    for (size_t batch = 0; batch < BATCH_SIZE; ++batch){
        for (size_t i = 0; i < num_kv_heads; ++i) {
            for (size_t j = 0; j < new_seq_len; ++j) {
                auto dst_ptr = new_tensor_data + num_kv_heads * new_seq_len * head_size * batch + new_seq_len * head_size * i +  head_size * j;
                auto src_ptr = old_tensor_data + num_kv_heads * new_seq_len * head_size * batch + old_seq_len * head_size * i +  head_size * j;
                std::memcpy(dst_ptr, src_ptr, head_size * sizeof(float));
            }
        }
    }
    return new_tensor;
}

void update_kv_cache(ov::InferRequest request, uint64_t seq_len_axis, uint64_t new_seq_len) {
    // trims kv_cache values up to the new_seq_len
    auto num_outputs = request.get_compiled_model().outputs().size();
    for (size_t idx = 1; idx < num_outputs; ++idx) {
        auto old_tensor = request.get_output_tensor(idx);
        auto trimmed_tensor = trimm_tensor(old_tensor, seq_len_axis, new_seq_len);
        request.set_output_tensor(idx, trimmed_tensor);
    }
}

int main(int argc, char* argv[]) try {
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <DRAFT MODEL_DIR> <MAIN MODEL_DIR> '<PROMPT>'");
    }

    // tokenizer model
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    auto [draft_input_ids, draft_attention_mask] = tokenize(tokenizer, argv[3]);
    ov::InferRequest detokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    TextStreamer text_streamer{std::move(detokenizer)};

    // draft model
    ov::InferRequest draft_model = core.compile_model(std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();

    draft_model.set_tensor("input_ids", draft_input_ids);
    draft_model.set_tensor("attention_mask", draft_attention_mask);
    
    ov::Tensor draft_position_ids = draft_model.get_tensor("position_ids");
    draft_position_ids.set_shape(draft_input_ids.get_shape());
    std::iota(draft_position_ids.data<int64_t>(), draft_position_ids.data<int64_t>() + draft_position_ids.get_size(), 0);
    uint64_t seq_len = draft_input_ids.get_shape()[1];

    // main model
    ov::InferRequest main_model = core.compile_model(std::string{argv[2]} + "/openvino_model.xml", "CPU").create_infer_request();

    // Input tensors for the main model should not be mixed with draft.
    // Do not feed the same draft_postion_ids to the main, but copy input_ids from the draft_input_ids
    auto input_ids = main_model.get_tensor("input_ids");
    input_ids.set_shape(draft_input_ids.get_shape());
    for (int i = 0; i < seq_len; ++i)
        input_ids.data<int64_t>()[i] = draft_input_ids.data<int64_t>()[i];

    auto attention_mask = main_model.get_tensor("attention_mask");
    attention_mask.set_shape(draft_input_ids.get_shape());
    std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

    auto position_ids = main_model.get_tensor("position_ids");
    position_ids.set_shape(draft_input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    
    // To coollect kv-cache for the <PROMPT> and to get the next token run the very first infer request
    init_key_values(draft_model);
    init_key_values(main_model);
    draft_model.infer();
    main_model.infer();

    size_t vocab_size = draft_model.get_tensor("logits").get_shape().back();
    OPENVINO_ASSERT(vocab_size == main_model.get_tensor("logits").get_shape().back(), "vocab size should be the same for the both models");
       
    // logits shape is [BATCH_SIZE, input_ids.size, vocab_size]
    auto logits = main_model.get_tensor("logits");
    auto data_logits = logits.data<float>() + (seq_len - 1) * vocab_size;
    int64_t next_token = std::max_element(data_logits, data_logits + vocab_size) - data_logits;
    
    // the first token which is fed to both draft and main netwoks on each iteration
    auto first_token = next_token;
    text_streamer.put(next_token);
    
    // run K infer requests on draft model and get next K prediction tokens on each iteration
    uint64_t K = 5;
    std::vector<int64_t> draft_tokens;

    // The draft model predicts tokens one by one in an auto-regressive manner, draft_input_ids length should be 1.
    draft_input_ids.set_shape({BATCH_SIZE, 1});
    draft_position_ids.set_shape({BATCH_SIZE, 1});

    /*
    Speculative decoding works the following way.
    The draft model predicts the next K tokens one by one. 
    These K tokens are then fed to the main model in a single inference request.
    We go through each predicted token, and if a difference is detected, we stop 
    and keep the last token predicted by the main network.
    Then the draft model gets the latest main prediction and again tries to predict 
    the next K tokens, repeating the cycle.
    
    This approach reduces the need for multiple infer requests 
    to the main model, enhancing performance. For instance, in more predictable 
    parts of text generation, the draft model can, in best-case scenarios, 
    generate the next K tokens that exactly match the target. In that case theу
    are validated in a single inference request of the main model (which is bigger, 
    more accurate but slower) instead of running K subsequent requests.
    */
    int max_sequence_length = 100;
    while (next_token != SPECIAL_EOS_TOKEN && seq_len < max_sequence_length) {
        // infer the K next tokens with draft model
        for (int i = 0; i < K; ++i) {
            draft_input_ids.data<int64_t>()[0] = next_token;
            draft_attention_mask.set_shape({BATCH_SIZE, seq_len + i + 1});
            std::fill_n(draft_attention_mask.data<int64_t>(), draft_attention_mask.get_size(), 1);
            draft_position_ids.data<int64_t>()[0] = int64_t(draft_attention_mask.get_size() - 1);

            forward_key_values(draft_model);  // forward KV cache from Result outputs to Parameter inputs
            draft_model.infer();

            auto draft_logits = draft_model.get_tensor("logits").data<float>();
            int64_t arg_max_token = std::max_element(draft_logits, draft_logits + vocab_size) - draft_logits;
            next_token = arg_max_token;
            draft_tokens.emplace_back(arg_max_token);
        }

        // For the main network, K tokens will be fed at once in a single infer request.
        input_ids.set_shape({BATCH_SIZE, K});
        // Set the first token for the main model to be the same as for the draft model.
        input_ids.data<int64_t>()[0] = first_token;
        for (int i = 0; i < K - 1; i++)
            input_ids.data<int64_t>()[i + 1] = draft_tokens[i];

        attention_mask.set_shape({BATCH_SIZE, seq_len + K});
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

        position_ids.set_shape({BATCH_SIZE, K});
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), seq_len);

        forward_key_values(main_model);
        main_model.infer();

        data_logits = logits.data<float>();  // [BATCH_SIZE, K, vocab_size]
        size_t disagree_idx = K - 1;
        // Iterate through the predicted tokens from the main model and compare them with draft predictions.
        // In the worst-case scenario (disagreement at the beginning), iter will increase by 1.
        // In the best-case scenario, all elements match, and K predicted tokens will be taken.
        for (size_t i = 0; i < K; i++) {
            auto start = data_logits + vocab_size * i;
            auto stop = data_logits + vocab_size * (i + 1);
            next_token = std::max_element(start, stop) - start;
            text_streamer.put(next_token);

            disagree_idx = i;                
            if (next_token != draft_tokens[i] || next_token == SPECIAL_EOS_TOKEN)
                break;
        }

        // After the inference request, key/values have shape [BATCH_SIZE, seq_len + K, vocab_size].
        // Increment the sequence length by the number of matched tokens, and
        // trim the KV cache to match the new sequence length.
        seq_len += disagree_idx + 1;
        update_kv_cache(draft_model, SEQ_LEN_AXIS, seq_len);
        update_kv_cache(main_model, SEQ_LEN_AXIS, seq_len);
        
        draft_tokens.clear();
        first_token = next_token;
    }
    text_streamer.end();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
