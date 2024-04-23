// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cmath>
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

constexpr size_t BATCH_SIZE = 1;

// sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size], 
// threfore usually SEQ_LEN_AXIS = 2
constexpr size_t SEQ_LEN_AXIS = 2;

int64_t SPECIAL_EOS_TOKEN;

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

template<typename T>
std::vector<int> findIndices(std::vector<T> const &v, int target) {
    std::vector<int> indices;

    for (auto it = v.begin(); it != v.end(); it++) {
        if (*it == target) {
            indices.push_back(std::distance(v.begin(), it));
        }
    }

    return indices;
}

std::vector<int64_t> find_candidate_pred_tokens(std::vector<int64_t> input_ids, int32_t max_ngram_size, int32_t num_pred_tokens){
	std::vector<int64_t> candidate_tokens_vec;
	int32_t input_length = input_ids.size();
	std::cout << "input_length: " << input_length << "\n";
	std::cout << "max_ngram_size: " << max_ngram_size << "\n";
	std::cout << "num_pred_tokens: " << num_pred_tokens << "\n";
	for (int ngram_size = max_ngram_size; ngram_size > 0; ngram_size--){
	    std::cout << "ngram_size: " << ngram_size << "\n";

	    // Extract the last n tokens as our search ngram
	    std::vector<int64_t> ngram(input_ids.end() - std::min<int64_t>(input_ids.size(), ngram_size), input_ids.end());
	    std::cout << "ngram: [";
	    for (auto& n: ngram) {
	       std::cout << n << ", ";
	    }
	    std::cout << "]\n";

	    // Create sliding windows of size ngram_size
	    std::vector<std::vector<int64_t>> sliding_windows;
	    for (int32_t i=0 ; i + ngram_size <= input_length; i++){
	        std::vector<int64_t> window(input_ids.begin() + i, input_ids.begin() + i + ngram_size);
		std::cout << "window: [";
		for (auto& n: window) {
			std::cout << n << ", ";
		}
		std::cout << "]\n";
		sliding_windows.push_back(window);
	    }
	    std::vector<int> matches(sliding_windows.size(), 0);
	    int i=0;
	    // Find where the windows match the ngram
	    for (auto& window: sliding_windows){
	        if (window == ngram){
		    matches[i] = 1;
		}
		i++;
	    }
	    std::cout << "matches: [";
            for (auto& n: matches) {
		    std::cout << n << ", ";
            }
            std::cout << "]\n";
	    // Get the indices of matches
	    std::vector<int> match_indices = findIndices(matches, 1);
	    std::cout << "match indices: [";
            for (auto& n: match_indices) {
                    std::cout << n << ", ";
            }
            std::cout << "]\n";

	    for (auto& n: match_indices) {
		int start_idx = n + ngram_size;
		int end_idx = start_idx + num_pred_tokens;
		// Ensure we don't go beyond the length of input_ids and avoid self-match
		if ((end_idx <= input_length) && (start_idx < input_length - ngram_size)){
		    std::copy(input_ids.begin() + start_idx, input_ids.begin() + end_idx, std::back_inserter(candidate_tokens_vec)); 
		    return candidate_tokens_vec;
		}
            }
	}

	return candidate_tokens_vec;
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
    
    OPENVINO_ASSERT(new_seq_len <= old_seq_len);
    
    // if new_seq_len equal to old one no need to copy tensor, return as is
    if (old_seq_len == new_seq_len)
        return tensor;

    if (seq_len_axis == 0) {
        shape[0] = new_seq_len;
        tensor.set_shape(shape);
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
    // trim kv_cache values up to the new_seq_len
    for (auto& state: request.query_state()) {
        ov::Tensor old_tensor = state.get_state();
        state.set_state(trimm_tensor(old_tensor, seq_len_axis, new_seq_len));
    }
}

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MAIN MODEL_DIR> '<PROMPT>'");
    }

    // tokenizer model
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    std::cout << "Load tokenizer... \n";
    auto tokenizer_model = core.read_model(std::string{argv[1]} + "/openvino_tokenizer.xml");
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(
        tokenizer_model, "CPU").create_infer_request();
    //auto [draft_input_ids, draft_attention_mask] = tokenize(tokenizer, argv[3]);
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[2]);

    std::cout << "Load detokenizer ...\n";
    ov::InferRequest detokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    TextStreamer text_streamer{std::move(detokenizer)};

    // draft model
    //ov::InferRequest draft_model = core.compile_model(std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();

    //draft_model.set_tensor("input_ids", draft_input_ids);
    //draft_model.set_tensor("attention_mask", draft_attention_mask);
    
    //ov::Tensor draft_position_ids = draft_model.get_tensor("position_ids");
    //draft_position_ids.set_shape(draft_input_ids.get_shape());
    //std::iota(draft_position_ids.data<int64_t>(), draft_position_ids.data<int64_t>() + draft_position_ids.get_size(), 0);
    //uint64_t seq_len = draft_input_ids.get_shape()[1];
    std::cout << "Load LLM start ...\n";
    // main model
    ov::InferRequest main_model = core.compile_model(std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();
    std::cout << "Load LLM finished\n";

    // Input tensors for the main model should not be mixed with draft.
    // Do not feed the same draft_postion_ids to the main, but copy input_ids from the draft_input_ids
    //auto input_ids = main_model.get_tensor("input_ids");
    main_model.set_tensor("input_ids", input_ids);
    main_model.set_tensor("attention_mask", attention_mask);
    uint64_t seq_len = input_ids.get_shape()[1];
    //input_ids.set_shape(draft_input_ids.get_shape());
    //draft_input_ids.copy_to(input_ids);

    //auto attention_mask = main_model.get_tensor("attention_mask");
    //attention_mask.set_shape(draft_input_ids.get_shape());
    //attention_mask.set_shape(input_ids.get_shape());
    //std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

    auto position_ids = main_model.get_tensor("position_ids");
    //position_ids.set_shape(draft_input_ids.get_shape());
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
    main_model.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    main_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    /* 
    // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
    draft_model.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    draft_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    main_model.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    main_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    */
    // To collect kv-cache for the <PROMPT> and to get the next token run the very first infer request
    //draft_model.infer();
    std::cout << "LLM first infer start...\n";
    main_model.infer();
    std::cout << "LLM first infer end...\n";

    //size_t vocab_size = draft_model.get_tensor("logits").get_shape().back();
    size_t vocab_size = main_model.get_tensor("logits").get_shape().back();
    //OPENVINO_ASSERT(vocab_size == main_model.get_tensor("logits").get_shape().back(), "vocab size should be the same for the both models");
       
    // logits shape is [BATCH_SIZE, seq_len, vocab_size]
    auto logits = main_model.get_tensor("logits");
    auto data_logits = logits.data<float>() + (seq_len - 1) * vocab_size;
    int64_t out_token = std::max_element(data_logits, data_logits + vocab_size) - data_logits;
    
    // the first token which is fed to both draft and main netwoks on each iteration
    auto first_token = out_token;
    text_streamer.put(out_token);
    
    // run K infer requests on draft model and get next K prediction tokens on each iteration
    uint64_t K = 5;
    //std::vector<int64_t> draft_tokens;

    // The draft model predicts tokens one by one in an auto-regressive manner, draft_input_ids length should be 1.
    //draft_input_ids.set_shape({BATCH_SIZE, 1});
    //draft_position_ids.set_shape({BATCH_SIZE, 1});
    std::cout << "Get runtime information from tokenizer start...\n";
    auto rt_info = tokenizer_model->get_rt_info(); //Get the runtime info for the model
    std::cout << "Get runtime information from tokenizer finished...\n";

    if (rt_info.count("eos_token_id") > 0) { //check if the runtime information has a valid EOS token ID
        SPECIAL_EOS_TOKEN = rt_info["eos_token_id"].as<int64_t>();
    } else {
        throw std::runtime_error("EOS token ID not found in model's runtime information.");
    }

   /* Speculative decoding works the following way. The draft model predicts the next K
   tokens one by one in an autoregressive manner, while the main model validates these
   predictions and corrects them if necessary. We go through each predicted token, and
   if a difference is detected between the draft and main model, we stop and keep the
   last token predicted by the main model. Then the draft model gets the latest main
   prediction and again tries to predict the next K tokens, repeating the cycle.

   This approach reduces the need for multiple infer requests to the main model,
   enhancing performance. For instance, in more predictable parts of text generation,
   the draft model can, in best-case scenarios, generate the next K tokens that exactly
   match the target. In tha caste the are validated in a single inference request to
   the main model (which is bigger, more accurate but slower) instead of running K
   subsequent requests. 
   */
    int max_sequence_length = 60;
    std::vector<int64_t> input_ids_vec = {};
    //std::cout << "Original main model input ids: [";
    for (size_t i=0; i < seq_len; i++){
        //std::cout << input_ids.data<int64_t>()[i] << ", ";
	input_ids_vec.push_back(input_ids.data<int64_t>()[i]);
    }
    //std::cout << "]\n";

    int32_t max_ngram_size = 3;
    int32_t num_pred_tokens = K; // K=5
    /*
    std::vector<int64_t> candidate_tokens_vec = find_candidate_pred_tokens(input_ids_vec, max_ngram_size, num_pred_tokens);
    std::cout << "Find candidate tokens: [";
        for (auto& ct: candidate_tokens_vec) {
            std::cout << ct << ", ";
        }
    std::cout << " ]\n";
    */
    while (out_token != SPECIAL_EOS_TOKEN && seq_len < max_sequence_length) {
	std::cout << "\nStart new 2nd token generation, current seq_len: " << seq_len << "\n";
	std::cout << "Current Input ids: [";
        for (auto& in: input_ids_vec) {
            std::cout << in << ", ";
        }
        std::cout << " ]\n";
        std::vector<int64_t> candidate_tokens_vec = find_candidate_pred_tokens(input_ids_vec, max_ngram_size, num_pred_tokens);
	if (candidate_tokens_vec.empty()){
		std::cout << "No candidate tokens found, set 100 as dummy candidate tokens\n";
		candidate_tokens_vec.push_back(100);
	}
	std::cout << "Current candidate tokens: [";
        for (auto& ct: candidate_tokens_vec) {
            std::cout << ct << ", ";
        }
	std::cout << " ]\n";

	// infer the K next tokens with draft model
	/*
        for (int i = 0; i < K; ++i) {
            draft_input_ids.data<int64_t>()[0] = out_token;
            draft_attention_mask.set_shape({BATCH_SIZE, seq_len + i + 1});
            std::fill_n(draft_attention_mask.data<int64_t>(), draft_attention_mask.get_size(), 1);
            draft_position_ids.data<int64_t>()[0] = int64_t(draft_attention_mask.get_size() - 1);

            draft_model.infer();

            auto draft_logits = draft_model.get_tensor("logits").data<float>();
            int64_t arg_max_token = std::max_element(draft_logits, draft_logits + vocab_size) - draft_logits;
            out_token = arg_max_token;
            draft_tokens.emplace_back(arg_max_token);
        }
	
	std::cout << "Draft tokens: [";
	for (auto& dt: draft_tokens) {
	    std::cout << dt << ", ";
	}
	std::cout << " ]\n";
	*/
        // For the main network, K tokens will be fed at once in a single infer request.
        input_ids.set_shape({BATCH_SIZE, K});
        // Set the first token for the main model to be the same as for the draft model.
        input_ids.data<int64_t>()[0] = first_token;
	/*
        for (int i = 0; i < K - 1; i++)
            input_ids.data<int64_t>()[i + 1] = draft_tokens[i];
	*/
	for (int i = 0; i < K - 1; i++)
            input_ids.data<int64_t>()[i + 1] = candidate_tokens_vec[i];
	std::cout << "Updated main model 2nd token input ids: [ ";
        for (size_t i=0; i < K; i++){
            std::cout <<  input_ids.data<int64_t>()[i]<< ", ";
        }
	std::cout << " ]\n";

        attention_mask.set_shape({BATCH_SIZE, seq_len + K});
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

        position_ids.set_shape({BATCH_SIZE, K});
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), seq_len);

        main_model.infer();

        data_logits = logits.data<float>();  // [BATCH_SIZE, K, vocab_size]
        size_t disagree_idx = K - 1;
        // Iterate through the predicted tokens from the main model and compare them with draft predictions.
        // In the worst-case scenario (disagreement at the beginning), iter will increase by 1.
        // In the best-case scenario, all elements match, and K predicted tokens will be taken.
	std::vector<int64_t> selected_tokens;
        for (size_t i = 0; i < K; i++) {
            auto start = data_logits + vocab_size * i;
            auto stop = data_logits + vocab_size * (i + 1);
            out_token = std::max_element(start, stop) - start;
	    std::cout << "out_token: " << out_token << "\n";
	    //text_streamer.put(out_token);

            disagree_idx = i;                
	    //std::cout << "start: " << *start << " , stop: " << *stop << " , out_token: " << out_token << "\n";
	    std::cout << "disagree_idx: " << disagree_idx << "\n";
	    //if (out_token != draft_tokens[i] || out_token == SPECIAL_EOS_TOKEN || seq_len + disagree_idx + 1 >= max_sequence_length)
	    selected_tokens.push_back(out_token);
	    if (out_token != candidate_tokens_vec[i] || out_token == SPECIAL_EOS_TOKEN || seq_len + disagree_idx + 1 >= max_sequence_length)
                break;
        }

        // After the inference request, key/values have shape [BATCH_SIZE, seq_len + K, vocab_size].
        // Increment the sequence length by the number of matched tokens, and
        // trim the KV cache to match the new sequence length.
        seq_len += disagree_idx + 1;
	
	//std::vector<int64_t> selected_tokens = candidate_tokens_vec
	//std::copy(candidate, std::back_inserter(input_ids_vec));
	std::cout << "Selected tokens: [";
        for (auto& st: selected_tokens) {
            std::cout << st << ", ";
        }
        std::cout << " ]\n";
        //update_kv_cache(draft_model, SEQ_LEN_AXIS, seq_len);		
        update_kv_cache(main_model, SEQ_LEN_AXIS, seq_len);
	std::copy(selected_tokens.begin(), selected_tokens.end(), std::back_inserter(input_ids_vec));
        //draft_tokens.clear();
	
        first_token = out_token;
	//break;
    }
    text_streamer.end();
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one sequence is processed,
    // it is called for education purposes:
    // draft_model.reset_state();
    main_model.reset_state();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
