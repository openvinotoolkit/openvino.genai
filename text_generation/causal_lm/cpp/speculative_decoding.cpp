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

std::vector<float> softmax(const ov::Tensor& logits, float temperature) {
    float* logits_data = logits.data<float>();
    int size = logits.get_size();
    
    double sum_exp = 0.0;
    for (int i = 0; i < size; i++) {
        sum_exp += std::exp(logits_data[i] / temperature);
    }
    
    std::vector<float> probabilities;
    for (int i = 0; i < size; i++) {
        double probability = exp(logits_data[i] / temperature) / sum_exp;
        probabilities.push_back(probability);
    }
    return probabilities;
}

int random_sample(const ov::Tensor& logits, float temperature) {
    auto probabilities = softmax(logits, temperature);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> distribution(probabilities.begin(), probabilities.end());
    int sampled_index = distribution(gen);
    
    return sampled_index;
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

std::vector<int64_t> convert_to_vector(ov::Tensor tensor) {
    std::vector<int64_t> res_vector;
    for (int i = 0; i << tensor.get_shape().back(); i++) {
        res_vector.emplace_back(tensor.data<int64_t>()[i]);
    }
    return res_vector;
}

ov::Tensor append_element(ov::Tensor tensor_val, int64_t element) {
    auto vec = convert_to_vector(tensor_val);
    vec.emplace_back(element);
    return ov::Tensor{ov::element::i64, {BATCH_SIZE, tensor_val.get_shape().back()}, vec.data()};
}

void set_key_values(ov::InferRequest& request, int size) {
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

void init_key_values(ov::InferRequest request, int kv_length, unsigned size_1, unsigned size_2) {
    std::stringstream ss_1, ss_2;
    ss_1.clear();
    ss_2.clear();

    for (int i = 0; i < kv_length; i++) {
        ss_2 << "past_key_values." << i << ".key";
        request.set_tensor(ss_2.str(), ov::Tensor(ov::element::f32, {BATCH_SIZE, size_1, 0, size_2}));
        ss_2.str("");

        ss_2 << "past_key_values." << i << ".value";
        request.set_tensor(ss_2.str(), ov::Tensor(ov::element::f32, {BATCH_SIZE, size_1, 0, size_2}));
        ss_2.str("");
    }
}

ov::Tensor trimm_tensor(ov::Tensor tensor, uint64_t axis, uint64_t new_seq_len) {
    // tensor shape = [batch, size_1, seq_len, size_2]
    auto old_tensor_data = tensor.data<float>();
    auto shape = tensor.get_shape();
    auto size_1 = shape[1];
    auto old_seq_len = shape[2];
    auto size_2 = shape[3];
    
    // assume batch is always 1, if different one more cycle is needed
    // copy elements from the old to the new tensor
    auto new_tensor = ov::Tensor{ov::element::f32, {BATCH_SIZE, size_1, new_seq_len, size_2}};
    auto new_tensor_data = new_tensor.data<float>();
    for (int i = 0; i < size_1; i++) {
        for (int j = 0; j < new_seq_len; j++) {
            for (int k = 0; k < size_2; k++) {
                new_tensor_data[new_seq_len * size_2 * i +  size_2 * j + k] = old_tensor_data[old_seq_len * size_2 * i +  size_2 * j + k];
            }
        }
    }
    return new_tensor;
}

void update_kv_cache(ov::InferRequest request, uint64_t new_seq_len, int kv_length) {
    // key/value shape = [batch, size_1, seq_len, size_2]
    std::stringstream ss;
    for (int i = 0; i < kv_length; i++) {
        
        for (auto name: {"key", "value"}) {
            ss << "present." << i;
            ss << (name == "key" ? ".key" : ".value");
            auto tensor_name = ss.str();
            
            auto trimmed_tensor = trimm_tensor(request.get_tensor(tensor_name), 2, new_seq_len);
            request.set_tensor(tensor_name, trimmed_tensor);
            ss.str("");
        }
    }
}

bool verbose = false;

using namespace std;
void print_accum_tokens(ov::InferRequest& detokenizer, std::vector<int64_t> tokens, string suffix) {
    std::string text = detokenize(detokenizer, tokens);
    stringstream ss;
    for (const auto& token: tokens)
        ss << token << " ";
    cerr << suffix << ": ";
    cerr << text << " | " << ss.str() << endl;
}

string tokens_to_string(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    return detokenize(detokenizer, tokens);
}

int main(int argc, char* argv[]) try {
    // int tiny_llama_kv_size = 22;
    // int tiny_llama_size_1 = 4;
    // int tiny_llama_size_2 = 64;

    int tiny_llama_kv_size = 22;
    int tiny_llama_size_1 = 4;
    int tiny_llama_size_2 = 64;

    int llama_kv_size = 32;
    int llama_size_1 = 32;
    int llama_size_2 = 128;

    // int llama_kv_size = 22;
    // int llama_size_1 = 4;
    // int llama_size_2 = 64;

    if (argc != 4 && argc != 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <DRAFT MODEL_DIR> <TARGET MODEL_DIR> '<PROMPT>'");
    }
    if (argc == 5) {
        if (!strcmp(argv[4], "verbose"))
            verbose = true;
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
    ov::InferRequest detokenizer_2 = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    
    // draft model
    ov::InferRequest lm = core.compile_model(
        std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();

    lm.set_tensor("input_ids", input_ids);
    // std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), 1);
    uint64_t iter = input_ids.get_shape()[1] + 1;

    lm.set_tensor("input_ids", input_ids);
    lm.set_tensor("attention_mask", attention_mask);
    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    init_key_values(lm, tiny_llama_kv_size, tiny_llama_size_1, tiny_llama_size_2);
    lm.infer();

    // target
    ov::InferRequest lm_target = core.compile_model(
    std::string{argv[2]} + "/openvino_model.xml", "CPU").create_infer_request();
        
    auto target_input_ids = lm_target.get_tensor("input_ids");
    target_input_ids.set_shape(input_ids.get_shape());
    for (int i = 0; i < input_ids.get_shape()[1]; i++) {
        target_input_ids.data<int64_t>()[i] = input_ids.data<int64_t>()[i];
    }

    lm_target.get_tensor("attention_mask").set_shape(input_ids.get_shape());
    std::fill_n(lm_target.get_tensor("attention_mask").data<int64_t>(), lm_target.get_tensor("attention_mask").get_size(), 1);

    ov::Tensor target_position_ids = lm_target.get_tensor("position_ids");
    target_position_ids.set_shape(input_ids.get_shape());
    std::iota(target_position_ids.data<int64_t>(), target_position_ids.data<int64_t>() + target_position_ids.get_size(), 0);
    init_key_values(lm_target, llama_kv_size, llama_size_1, llama_size_2);
    lm_target.infer();

    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    
    // draft
    float* logits = lm.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    int64_t arg_max_token = std::max_element(logits, logits + vocab_size) - logits;
    int64_t out_token = arg_max_token;
    
    // target
    float* logits_target = lm_target.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    int64_t target_arg_max_token = std::max_element(logits_target, logits_target + vocab_size) - logits_target;
    int64_t target_out_token = target_arg_max_token;

    lm.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    lm.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    lm_target.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    lm_target.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});

    TextStreamer text_streamer{std::move(detokenizer)};
    
    constexpr int64_t SPECIAL_EOS_TOKEN = 2;
    int max_iter = 35;

    int K = 3;
    std::vector<int64_t> accumulated_tokens;
    std::vector<int64_t> accumulated_draft_tokens;
    auto target_logits = lm_target.get_tensor("logits");
    auto target_attention_mask = lm_target.get_tensor("attention_mask");

    // text_streamer.put(target_out_token);
    out_token = target_out_token;
    accumulated_draft_tokens.emplace_back(out_token);
    
    while (out_token != SPECIAL_EOS_TOKEN && iter < max_iter) {

        for (int i = 0; i < K; i++) {
            // draft
            lm.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, lm.get_tensor("attention_mask").get_shape()[1] + 1});
            std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), lm.get_tensor("attention_mask").get_size(), 1);
            lm.get_tensor("position_ids").data<int64_t>()[0] = int64_t(lm.get_tensor("attention_mask").get_size() - 1);
            set_key_values(lm, tiny_llama_kv_size);

            lm.infer();

            logits = lm.get_tensor("logits").data<float>();
            int64_t arg_max_token = std::max_element(logits, logits + vocab_size) - logits;
            out_token = arg_max_token;
            accumulated_draft_tokens.emplace_back(arg_max_token);
        }
        print_accum_tokens(detokenizer_2, accumulated_draft_tokens, "\ndraft token");
        auto res = tokens_to_string(detokenizer_2, accumulated_draft_tokens);

        // sanity check
        #if 0
        out_token = target_out_token;
        accumulated_draft_tokens.clear();
        accumulated_draft_tokens.emplace_back(target_out_token);
        for (int i = 0; i < K; i++) {
            // draft
            lm_target.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            lm_target.get_tensor("attention_mask").set_shape({BATCH_SIZE, lm_target.get_tensor("attention_mask").get_shape()[1] + 1});
            std::fill_n(lm_target.get_tensor("attention_mask").data<int64_t>(), lm_target.get_tensor("attention_mask").get_size(), 1);
            lm_target.get_tensor("position_ids").data<int64_t>()[0] = int64_t(lm_target.get_tensor("attention_mask").get_size() - 2);
            set_key_values(lm_target, llama_kv_size);

            lm_target.infer();

            logits = lm_target.get_tensor("logits").data<float>();
            int64_t arg_max_token = std::max_element(logits, logits + vocab_size) - logits;
            out_token = arg_max_token;
            accumulated_draft_tokens.emplace_back(arg_max_token);
        }
        print_accum_tokens(detokenizer_2, accumulated_draft_tokens, "target token");
        res = tokens_to_string(detokenizer_2, accumulated_draft_tokens);
        target_out_token = out_token;
        accumulated_draft_tokens.clear();
        accumulated_draft_tokens.emplace_back(out_token);

        iter += K;
        #else

        // target
        target_input_ids.set_shape({BATCH_SIZE, accumulated_draft_tokens.size()});
        for (int i = 0; i < accumulated_draft_tokens.size(); i++) {
            target_input_ids.data<int64_t>()[i] = accumulated_draft_tokens[i];
        }

        target_attention_mask.set_shape({BATCH_SIZE, target_attention_mask.get_shape()[1] + accumulated_draft_tokens.size()});
        std::fill_n(target_attention_mask.data<int64_t>(), target_attention_mask.get_size(), 1);

        target_position_ids.set_shape({BATCH_SIZE, accumulated_draft_tokens.size()});
        // todo: check position ids
        std::iota(target_position_ids.data<int64_t>(), target_position_ids.data<int64_t>() + target_position_ids.get_size(), iter);

        set_key_values(lm_target, llama_kv_size);
        lm_target.infer();
        
        logits_target = target_logits.data<float>();  // [batch, seq_len, vocab_size]
        int64_t target_arg_max_token;
        
        std::vector<int64_t> accumulated_target_tokens;
        
        text_streamer.put(accumulated_draft_tokens[0]);
        accumulated_tokens.clear();
        accumulated_tokens.emplace_back(accumulated_draft_tokens[0]);


        bool unmatched = true;
        for (int i = 0; i < accumulated_draft_tokens.size(); i++) {
            auto start = logits_target + vocab_size * i;
            auto stop = logits_target + vocab_size * (i + 1);
            target_arg_max_token = std::max_element(start, stop) - start;
            accumulated_tokens.emplace_back(target_arg_max_token);
            
            #if 1
            if (i != accumulated_draft_tokens.size() - 1 && target_arg_max_token != accumulated_draft_tokens[i  + 1] && unmatched) {
                auto new_seq_len = iter + 1;

                update_kv_cache(lm_target, new_seq_len, llama_kv_size);
                target_attention_mask.set_shape({BATCH_SIZE, new_seq_len});
                std::fill_n(target_attention_mask.data<int64_t>(), target_attention_mask.get_size(), 1);

                // for draft model
                update_kv_cache(lm, new_seq_len, tiny_llama_kv_size);
                lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, new_seq_len});
                std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), lm.get_tensor("attention_mask").get_size(), 1);
                if (i == 0) {
                    // if it failed starting from the very first token then need to reinfer with target network
                    // target
                    target_input_ids.set_shape({BATCH_SIZE, 1});
                    target_input_ids.data<int64_t>()[0] = out_token;

                    target_attention_mask.set_shape({BATCH_SIZE, target_attention_mask.get_shape()[1] + 1});
                    std::fill_n(target_attention_mask.data<int64_t>(), target_attention_mask.get_size(), 1);

                    // todo: check position ids
                    target_position_ids.set_shape({BATCH_SIZE, 1});
                    std::iota(target_position_ids.data<int64_t>(), target_position_ids.data<int64_t>() + target_position_ids.get_size(), iter + 1);

                    set_key_values(lm_target, llama_kv_size);
                    lm_target.infer();

                    logits = lm.get_tensor("logits").data<float>();
                    int64_t arg_max_token = std::max_element(logits, logits + vocab_size) - logits;
                    iter += 1;
                    text_streamer.put(target_arg_max_token);
                    out_token = arg_max_token;
                }
                print_accum_tokens(detokenizer_2, accumulated_tokens, "target tokens");
                // break;
                unmatched = false;

            }
            if (unmatched) {
                iter += 1; // todo: if it's needed to update anywhere else, e.g. when all elements in K matched
                text_streamer.put(target_arg_max_token);
                out_token = target_arg_max_token;
            } else {
                ;
            }
            #endif
        }
        print_accum_tokens(detokenizer_2, accumulated_tokens, "target tokens");

        // TODO: check
        accumulated_draft_tokens.clear();
        #endif
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
