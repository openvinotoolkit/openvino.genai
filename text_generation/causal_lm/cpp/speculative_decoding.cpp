// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cmath>
#include <random>

constexpr size_t BATCH_SIZE = 1;

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    constexpr size_t BATCH_SIZE = 1;
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
}

void forward_key_values(ov::InferRequest& request) {
    // forwards present key/values from Results to Parameters with past key/values
    // the first inputs are: input_ids, attention_masks, position_ids, other ones are past key/values.
    // the first output is logits, others are present key.value
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
        kv_tensor.set_shape({BATCH_SIZE, tensor_shape[1], 0, tensor_shape[3]});
    }
}

ov::Tensor trimm_tensor(ov::Tensor tensor, uint64_t new_seq_len) {
    // copy elements from the old to the new tensor
    // tensor shape = [batch, size_1, seq_len, size_2]
    auto old_tensor_data = tensor.data<float>();
    auto shape = tensor.get_shape();
    size_t size_1 = shape[1];
    size_t old_seq_len = shape[2];
    size_t size_2 = shape[3];
    
    auto new_tensor = ov::Tensor{ov::element::f32, {BATCH_SIZE, size_1, new_seq_len, size_2}};
    auto new_tensor_data = new_tensor.data<float>();
    
    // todo: assume batch_size is always 1, if it's not 1 one more cycle is needed
    for (size_t i = 0; i < size_1; ++i) {
        for (size_t j = 0; j < new_seq_len; ++j) {
            auto dst_ptr = new_tensor_data + new_seq_len * size_2 * i +  size_2 * j;
            auto src_ptr = old_tensor_data + old_seq_len * size_2 * i +  size_2 * j;
            std::memcpy(dst_ptr, src_ptr, size_2 * sizeof(float));
        }
    }
    return new_tensor;
}

void update_kv_cache(ov::InferRequest request, uint64_t new_seq_len) {
    // trims kv_cache values up to the seq_len
    auto num_outputs = request.get_compiled_model().outputs().size();
    for (size_t idx = 1; idx < num_outputs; ++idx) {
        auto old_tensor = request.get_output_tensor(idx);
        auto trimmed_tensor = trimm_tensor(old_tensor, new_seq_len);
        request.set_output_tensor(idx, trimmed_tensor);
    }
}


auto tokens_to_string(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    return detokenize(detokenizer, tokens);
}

int main(int argc, char* argv[]) try {
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <DRAFT MODEL_DIR> <MAIN MODEL_DIR> '<PROMPT>'");
    }

    // Compile models
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[3]);
    ov::InferRequest detokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    // todo: for debug reason add one more detokenizer since the original one is moved to TextStreamer
    ov::InferRequest detokenizer_2 = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    
    // draft model
    ov::InferRequest draft_model = core.compile_model(std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();

    draft_model.set_tensor("input_ids", input_ids);
    draft_model.set_tensor("attention_mask", attention_mask);
    ov::Tensor position_ids = draft_model.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    init_key_values(draft_model);
    draft_model.infer();

    // main
    ov::InferRequest main_model = core.compile_model(
    std::string{argv[2]} + "/openvino_model.xml", "CPU").create_infer_request();

    auto main_input_ids = main_model.get_tensor("input_ids");
    main_input_ids.set_shape(input_ids.get_shape());
    for (int i = 0; i < input_ids.get_shape()[1]; i++) {
        main_input_ids.data<int64_t>()[i] = input_ids.data<int64_t>()[i];
    }

    auto main_attention_mask = main_model.get_tensor("attention_mask");
    main_attention_mask.set_shape(input_ids.get_shape());
    std::fill_n(main_attention_mask.data<int64_t>(), main_attention_mask.get_size(), 1);

    auto main_position_ids = main_model.get_tensor("position_ids");
    main_position_ids.set_shape(input_ids.get_shape());
    std::iota(main_position_ids.data<int64_t>(), main_position_ids.data<int64_t>() + main_position_ids.get_size(), 0);
    init_key_values(main_model);
    main_model.infer();

    size_t vocab_size = draft_model.get_tensor("logits").get_shape().back();
    
    // draft
    float* logits = draft_model.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    int64_t arg_max_token = std::max_element(logits, logits + vocab_size) - logits;
    int64_t out_token = arg_max_token;
    
    // main
    auto main_logits = main_model.get_tensor("logits");
    float* logits_main = main_logits.data<float>() + (input_ids.get_size() - 1) * vocab_size;
    int64_t main_arg_max_token = std::max_element(logits_main, logits_main + vocab_size) - logits_main;
    int64_t main_out_token = main_arg_max_token;

    std::vector<int64_t> all_tokens;
    std::vector<int64_t> accum_main_tokens;
    std::vector<int64_t> accum_draft_tokens;
    main_input_ids.set_shape(input_ids.get_shape());
    for (int i = 0; i < input_ids.get_shape()[1]; i++) {
        all_tokens.emplace_back(input_ids.data<int64_t>()[i]);
    }
    auto beginning_text = tokens_to_string(detokenizer_2, all_tokens);
    auto atten_shape = draft_model.get_tensor("attention_mask").get_shape();
    auto draft_shape = draft_model.get_tensor("present.0.key").get_shape();

    uint64_t iter = input_ids.get_shape()[1];
    main_model.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    main_model.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    draft_model.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    draft_model.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    
    constexpr int64_t SPECIAL_EOS_TOKEN = 2;
    int max_iter = 100;

    uint64_t K = 4;

    TextStreamer text_streamer{std::move(detokenizer)};
    text_streamer.put(main_out_token);
    out_token = main_out_token;
    auto first_token = main_out_token;
    all_tokens.emplace_back(main_out_token);
    std::string the_whole_text;  // todo: remove on final version, is used for debug purposes only
    while (out_token != SPECIAL_EOS_TOKEN && iter < max_iter) {
        
        for (int i = 0; i < K; i++) {
            // draft
            draft_model.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            draft_model.get_tensor("attention_mask").set_shape({BATCH_SIZE, draft_model.get_tensor("attention_mask").get_shape()[1] + 1});
            std::fill_n(draft_model.get_tensor("attention_mask").data<int64_t>(), draft_model.get_tensor("attention_mask").get_size(), 1);
            draft_model.get_tensor("position_ids").data<int64_t>()[0] = int64_t(draft_model.get_tensor("attention_mask").get_size() - 1);  // TODO: try with -2
            forward_key_values(draft_model);

            draft_model.infer();

            logits = draft_model.get_tensor("logits").data<float>();
            int64_t arg_max_token = std::max_element(logits, logits + vocab_size) - logits;
            out_token = arg_max_token;
            accum_draft_tokens.emplace_back(arg_max_token);
        }
        auto res_str_draft = tokens_to_string(detokenizer_2, accum_draft_tokens);

        // main
        main_input_ids.set_shape({BATCH_SIZE, K});
        // main network will give also K out tokens
        // feed the same first token to the main network and do not give the last token generated by draft
        main_input_ids.data<int64_t>()[0] = first_token;
        for (int i = 0; i < accum_draft_tokens.size() - 1; i++)
            main_input_ids.data<int64_t>()[i + 1] = accum_draft_tokens[i];

        main_attention_mask.set_shape({BATCH_SIZE, main_attention_mask.get_shape()[1] + K});
        std::fill_n(main_attention_mask.data<int64_t>(), main_attention_mask.get_size(), 1);

        main_position_ids.set_shape({BATCH_SIZE, K});
        std::iota(main_position_ids.data<int64_t>(), main_position_ids.data<int64_t>() + main_position_ids.get_size(), 
                  main_attention_mask.get_size() - K);

        forward_key_values(main_model);
        main_model.infer();

        logits_main = main_logits.data<float>();  // [batch, seq_len, vocab_size]
        accum_main_tokens.clear();
        int64_t main_arg_max_token;
        
        for (int i = 0; i < K; i++) {
            auto start = logits_main + vocab_size * i;
            auto stop = logits_main + vocab_size * (i + 1);
            main_arg_max_token = std::max_element(start, stop) - start;
            accum_main_tokens.emplace_back(main_arg_max_token);
        }
        // todo: these cycles are split for debug purpose, join with the next when ready
        auto res_str_main = tokens_to_string(detokenizer_2, accum_main_tokens);
        
        auto draft_shape = draft_model.get_tensor("present.0.key").get_shape();
        auto main_shape = main_model.get_tensor("present.0.key").get_shape();
        auto whole_text = tokens_to_string(detokenizer_2, all_tokens);
 
        bool newer_breaked = true;
        auto diagree_idx = K - 1;  // if all elements match will take the very last token from accum_tokens
        for (int i = 0; i < K; i++) { // check the very last element of accum_tokens
            text_streamer.put(accum_main_tokens[i]);
            all_tokens.emplace_back(accum_main_tokens[i]);
            the_whole_text = tokens_to_string(detokenizer_2, all_tokens);
            if (accum_main_tokens[i] != accum_draft_tokens[i]) {
                diagree_idx = i;
                break;
            }
        }
        
        first_token = accum_main_tokens[diagree_idx];
        out_token = accum_main_tokens[diagree_idx];

        iter += diagree_idx + 1;
        update_kv_cache(draft_model, iter);
        draft_model.get_tensor("attention_mask").set_shape({BATCH_SIZE, iter});
        std::fill_n(draft_model.get_tensor("attention_mask").data<int64_t>(), draft_model.get_tensor("attention_mask").get_size(), 1);
        
        update_kv_cache(main_model, iter);
        main_model.get_tensor("attention_mask").set_shape({BATCH_SIZE, iter});
        std::fill_n(main_model.get_tensor("attention_mask").data<int64_t>(), main_model.get_tensor("attention_mask").get_size(), 1);
        accum_draft_tokens.clear();
    }
    text_streamer.end();
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one sequence is processed,
    // it is called for education purposes:
    draft_model.reset_state();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
