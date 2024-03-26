// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "generate_pipeline.hpp"

namespace {

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

}  // namespace

void print_generation_results(GenerationResult results, ov::InferRequest& detokenizer) {
    TextStreamer text_streamer{std::move(detokenizer)};
    for (const auto& result: results) {
        text_streamer.put(result);
    }
    text_streamer.end();
}

int main(int argc, char* argv[]) try {
    std::string model_path = "/home/epavel/devel/openvino.genai/text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/";
    ov::Core core;
    // core.add_extension("libuser_ov_extensions.so");
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(model_path + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(model_path + "/openvino_detokenizer.xml", "CPU").create_infer_request();

    // The model can be compiled for GPU as well
    std::shared_ptr<ov::Model> model = core.read_model(model_path + "/openvino_model.xml");
    ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();
    
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[1]);
    
    SamplingParameters sampling_params = SamplingParameters::greedy();

    LLMEngine engine(request);
    GenerationResult generation_results = engine.generate(input_ids, sampling_params);
    print_generation_results(generation_results, detokenizer);

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
