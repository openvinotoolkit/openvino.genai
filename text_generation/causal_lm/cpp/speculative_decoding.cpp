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

void forward_key_values(ov::InferRequest& request, int size) {
    // forwards key values from Results to inputs
    // todo: rely on ports instead of names
    std::stringstream ss_1, ss_2;

    for (int i = 0; i < size; i++) {
        ss_1 << "present." << i << ".key";
        ss_2 << "past_key_values." << i << ".key";
        request.set_tensor(ss_2.str(), request.get_tensor(ss_1.str()));
        ss_1.str("");
        ss_2.str("");

        ss_1 << "present." << i << ".value";
        ss_2 << "past_key_values." << i << ".value";
        request.set_tensor(ss_2.str(), request.get_tensor(ss_1.str()));
        ss_1.str("");
        ss_2.str("");
    }
}

void init_key_values(ov::InferRequest request, int kv_length) {
    std::stringstream ss;
    ss.clear();
    
    for (int i = 0; i < kv_length; i++) {
        ss << "past_key_values." << i << ".key";
        auto tensor_shape = request.get_tensor(ss.str()).get_shape();

        request.set_tensor(ss.str(), ov::Tensor(ov::element::f32, {BATCH_SIZE, tensor_shape[1], 0, tensor_shape[3]}));
        ss.str("");

        ss << "past_key_values." << i << ".value";
        request.set_tensor(ss.str(), ov::Tensor(ov::element::f32, {BATCH_SIZE, tensor_shape[1], 0, tensor_shape[3]}));
        ss.str("");
    }
}

ov::Tensor trimm_tensor(ov::Tensor tensor, uint64_t axis, uint64_t new_seq_len) {
    // tensor shape = [batch, size_1, seq_len, size_2]
    auto old_tensor_data = tensor.data<float>();
    auto shape = tensor.get_shape();
    auto size_1 = shape[1];
    auto old_seq_len = shape[2];
    auto size_2 = shape[3];
    
    // todo: assume batch is always 1, if different one more cycle is needed
    // copy elements from the old to the new tensor
    auto new_tensor = ov::Tensor{ov::element::f32, {BATCH_SIZE, size_1, new_seq_len, size_2}};
    auto new_tensor_data = new_tensor.data<float>();
    for (int i = 0; i < size_1; i++) {
        for (int j = 0; j < new_seq_len; j++) {
            for (int k = 0; k < size_2; k++) {
                // todo: replace with memcpy
                new_tensor_data[new_seq_len * size_2 * i +  size_2 * j + k] = old_tensor_data[old_seq_len * size_2 * i +  size_2 * j + k];
            }
        }
    }
    return new_tensor;
}

void update_kv_cache(ov::InferRequest request, uint64_t new_seq_len, int kv_length) {
    // key/value shape = [batch, size_1, seq_len, size_2]
    uint64_t seq_len_axis = 2;  // check if can be removed hardcoded value
    
    std::stringstream ss;
    for (int i = 0; i < kv_length; i++) {
        
        for (auto name: {"key", "value"}) {
            ss << "present." << i;
            ss << (name == "key" ? ".key" : ".value");
            auto tensor_name = ss.str();
            if (request.get_tensor(tensor_name).get_shape()[seq_len_axis] < new_seq_len) {
                std::cout << "TERRIBLE" << std::endl;
            }
            auto trimmed_tensor = trimm_tensor(request.get_tensor(tensor_name), seq_len_axis, new_seq_len);
            request.set_tensor(tensor_name, trimmed_tensor);
            ss.str("");
        }
    }
}


using namespace std;
void print_accum_tokens(ov::InferRequest& detokenizer, std::vector<int64_t> tokens, string suffix) {
    std::string text = detokenize(detokenizer, tokens);
    stringstream ss;
    for (const auto& token: tokens)
        ss << token << " ";
    cerr << suffix << ": ";
    cerr << text << " | " << ss.str() << endl;
}

auto tokens_to_string(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    return detokenize(detokenizer, tokens);
}

int main(int argc, char* argv[]) try {
    int draft_kv_size = 22;  // todo: remove hardcoding and get this info from the network
    int main_kv_size = 32;

    // main_kv_size = 22;  // if TinyLlama is used as targed for testing purposes

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
    ov::InferRequest lm = core.compile_model(
        std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();

    lm.set_tensor("input_ids", input_ids);
    lm.set_tensor("attention_mask", attention_mask);
    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    init_key_values(lm, draft_kv_size);
    lm.infer();

    // main
    ov::InferRequest lm_main = core.compile_model(
    std::string{argv[2]} + "/openvino_model.xml", "CPU").create_infer_request();

    auto main_input_ids = lm_main.get_tensor("input_ids");
    main_input_ids.set_shape(input_ids.get_shape());
    for (int i = 0; i < input_ids.get_shape()[1]; i++) {
        main_input_ids.data<int64_t>()[i] = input_ids.data<int64_t>()[i];
    }

    auto main_attention_mask = lm_main.get_tensor("attention_mask");
    main_attention_mask.set_shape(input_ids.get_shape());
    std::fill_n(main_attention_mask.data<int64_t>(), main_attention_mask.get_size(), 1);

    auto main_position_ids = lm_main.get_tensor("position_ids");
    main_position_ids.set_shape(input_ids.get_shape());
    std::iota(main_position_ids.data<int64_t>(), main_position_ids.data<int64_t>() + main_position_ids.get_size(), 0);
    init_key_values(lm_main, main_kv_size);
    lm_main.infer();

    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    
    // draft
    float* logits = lm.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    int64_t arg_max_token = std::max_element(logits, logits + vocab_size) - logits;
    int64_t out_token = arg_max_token;
    
    // main
    auto main_logits = lm_main.get_tensor("logits");
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
    auto atten_shape = lm.get_tensor("attention_mask").get_shape();
    auto draft_shape = lm.get_tensor("present.0.key").get_shape();

    uint64_t iter = input_ids.get_shape()[1];
    lm.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    lm.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    lm_main.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    lm_main.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    
    constexpr int64_t SPECIAL_EOS_TOKEN = 2;
    int max_iter = 100;

    uint64_t K = 4;

    TextStreamer text_streamer{std::move(detokenizer)};
    text_streamer.put(main_out_token);
    out_token = main_out_token;
    auto first_token = main_out_token;
    all_tokens.emplace_back(main_out_token);
    string the_whole_text;  // todo: remove on final version, is used for debug purposes only
    while (out_token != SPECIAL_EOS_TOKEN && iter < max_iter) {
        
        for (int i = 0; i < K; i++) {
            // draft
            lm.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, lm.get_tensor("attention_mask").get_shape()[1] + 1});
            std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), lm.get_tensor("attention_mask").get_size(), 1);
            lm.get_tensor("position_ids").data<int64_t>()[0] = int64_t(lm.get_tensor("attention_mask").get_size() - 1);  // TODO: try with -2
            forward_key_values(lm, draft_kv_size);

            lm.infer();

            logits = lm.get_tensor("logits").data<float>();
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

        forward_key_values(lm_main, main_kv_size);
        lm_main.infer();

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
        
        auto draft_shape = lm.get_tensor("present.0.key").get_shape();
        auto main_shape = lm_main.get_tensor("present.0.key").get_shape();
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
        update_kv_cache(lm, iter, draft_kv_size);
        lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, iter});
        std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), lm.get_tensor("attention_mask").get_size(), 1);
        
        update_kv_cache(lm_main, iter, main_kv_size);
        lm_main.get_tensor("attention_mask").set_shape({BATCH_SIZE, iter});
        std::fill_n(lm_main.get_tensor("attention_mask").data<int64_t>(), lm_main.get_tensor("attention_mask").get_size(), 1);
        accum_draft_tokens.clear();
    }
    text_streamer.end();
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one sequence is processed,
    // it is called for education purposes:
    lm.reset_state();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
