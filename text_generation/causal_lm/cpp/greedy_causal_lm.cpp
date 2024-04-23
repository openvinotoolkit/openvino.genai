// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

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

void print_tensor(const ov::Tensor& tensor) {
    std::vector<int64_t> res;

    auto t_shape = tensor.get_shape();
    for (size_t i = 0; i < t_shape[1]; ++i) {
        if (tensor.get_element_type() == ov::element::i64) {
            res.emplace_back(tensor.data<int64_t>()[i]);
        }
    }
    std::cout << "";
}

template <class T>
void copy_partially(const ov::Tensor& src, const ov::Tensor& trg, int src_offset, int trg_offset, size_t size) {
    T* src_data = src.data<T>();
    T* dst_data = trg.data<T>();
    OPENVINO_ASSERT(src_offset + size <= src.get_shape()[1]);
    OPENVINO_ASSERT(trg_offset + size <= trg.get_shape()[1]);

    for (size_t i = 0; i < size; i++) {
        dst_data[trg_offset + i] = src_data[src_offset + i];
    }
}

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }
    // Compile models
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    //Read the tokenizer model information from the file to later get the runtime information
    auto tokenizer_model = core.read_model(std::string{argv[1]} + "/openvino_tokenizer.xml");
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(
        tokenizer_model, "CPU").create_infer_request();
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[2]);
    ov::InferRequest detokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    // The model can be compiled for GPU as well
    ov::InferRequest lm = core.compile_model(
        std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();
    auto seq_len = input_ids.get_size();

    ov::Tensor position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + seq_len, 0);
    

    constexpr size_t BATCH_SIZE = 1;
    // Input values are persistent between inference calls.
    // That allows to set values, which aren't going to change, only once
    lm.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    lm.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    // splitting the first inference
    auto shape = input_ids.get_shape();
    size_t split_pos = shape[1] * 0.95;
    int position = 4;
    if (position != -1)
        split_pos = position;
    
    ov::Shape split_1_shape = {1, split_pos};
    ov::Shape split_2_shape = {1, shape[1] - split_pos};

    ov::Tensor input_ids_1 = ov::Tensor{ov::element::i64, split_1_shape};
    ov::Tensor input_ids_2 = ov::Tensor{ov::element::i64, split_2_shape};
    
    ov::Tensor position_ids_1 = ov::Tensor{ov::element::i64, split_1_shape};
    ov::Tensor position_ids_2 = ov::Tensor{ov::element::i64, split_2_shape};

    ov::Tensor attention_mask_1 = ov::Tensor{ov::element::i64, split_1_shape};
    ov::Tensor attention_mask_2 = ov::Tensor{ov::element::i64, split_2_shape};

    copy_partially<int64_t>(input_ids, input_ids_1, 0, 0, split_pos);
    copy_partially<int64_t>(input_ids, input_ids_2, split_pos, 0, split_2_shape[1]);
    copy_partially<int64_t>(attention_mask, attention_mask_1, 0, 0, split_pos);
    copy_partially<int64_t>(attention_mask, attention_mask_2, split_pos, 0, split_2_shape[1]);
    copy_partially<int64_t>(position_ids, position_ids_1, 0, 0, split_pos);
    copy_partially<int64_t>(position_ids, position_ids_2, split_pos, 0, split_2_shape[1]);
    
    print_tensor(input_ids);
    print_tensor(input_ids_1);
    print_tensor(input_ids_2);

    print_tensor(position_ids);
    print_tensor(position_ids_1);
    print_tensor(position_ids_2);

    // 2 part inference
    lm.set_tensor("input_ids", input_ids_1);
    lm.set_tensor("attention_mask", attention_mask_1);
    lm.set_tensor("position_ids", position_ids_1);
    lm.infer();

    lm.set_tensor("input_ids", input_ids_2);
    lm.set_tensor("attention_mask", attention_mask);
    lm.set_tensor("position_ids", position_ids_2);
    lm.infer();
    auto shift = input_ids_2.get_shape()[1] - 1;

    // single inference
    // lm.set_tensor("input_ids", input_ids);
    // lm.set_tensor("attention_mask", attention_mask);
    // lm.set_tensor("position_ids", position_ids);
    // lm.infer();
    // auto shift = seq_len - 1;

    seq_len = lm.get_tensor("attention_mask").get_shape()[1];
    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    float* logits = lm.get_tensor("logits").data<float>() + shift * vocab_size;
    int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

    lm.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    lm.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    TextStreamer text_streamer{std::move(detokenizer)};

    // Get the runtime info from the tokenizer model that we read earlier
    auto rt_info = tokenizer_model->get_rt_info(); //Get the runtime info for the model
    int64_t SPECIAL_EOS_TOKEN;

    if (rt_info.count("eos_token_id") > 0) { //check if the runtime information has a valid EOS token ID
        SPECIAL_EOS_TOKEN = rt_info["eos_token_id"].as<int64_t>();
    } else {
        throw std::runtime_error("EOS token ID not found in model's runtime information.");
    }
    
    int max_sequence_length = 1315;
    while (out_token != SPECIAL_EOS_TOKEN && seq_len < max_sequence_length) {
        ++seq_len;
        lm.get_tensor("input_ids").data<int64_t>()[0] = out_token;
        lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, seq_len});
        std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), seq_len, 1);
        lm.get_tensor("position_ids").data<int64_t>()[0] = int64_t(seq_len - 1);

        lm.infer();
        text_streamer.put(out_token);
        // std::cout << out_token << " ";

        logits = lm.get_tensor("logits").data<float>();
        out_token = std::max_element(logits, logits + vocab_size) - logits;
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
