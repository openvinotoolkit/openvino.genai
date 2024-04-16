// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "generate_pipeline.hpp"


// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    LLMPipeline pipe;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = pipe.detokenize(token_cache);
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
        std::string text = pipe.detokenize(token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};

int main(int argc, char* argv[]) try {
    if (2 >= argc && argc <= 4)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" <DEVICE>");
    
    std::string prompt = "table is made of";
    std::string device = "CPU"; // can be replaced with GPU

    std::string model_path = argv[1];
    if (argc > 2)
        prompt = argv[2];
    if (argc > 3)
        device = argv[3];

    // Example 1: TextStreaming example with greedy search
    LLMPipeline pipe(model_path, device);
    // Will try to load config from generation_config.json.
    // but if not found default velues for gready search will be used
    GenerationConfig config = pipe.generation_config();

    auto text_streamer = TextStreamer{pipe};
    auto text_streamer_callback = [&text_streamer](std::vector<int64_t>&& tokens, LLMPipeline& pipe){
        text_streamer.put(tokens[0]);
    };

    cout << "greedy generate streaming mode:" << endl;
    config.max_new_tokens(20).set_callback(text_streamer_callback);
    pipe(prompt, config);
    text_streamer.end();
    
    // Example 2: Grouped Beam Search decoding example
    pipe = LLMPipeline(model_path, device);  
    config = pipe.generation_config();

    // will return vector with num_return_sequences strings
    auto num_return_sequences = 3;
    config.max_new_tokens(20).num_groups(3).group_size(5).num_return_sequences(num_return_sequences);
    
    cout << endl << "grouped beam search generated candidates:" << endl;
    auto generation_results = pipe({prompt}, config);
    for (int i = 0; i < num_return_sequences; ++i)
        cout << "candidate " << i << ": " << generation_results[i] << endl;

    // Example 3: Greedy Decoding with multiple batch
    pipe = LLMPipeline(model_path, device);
    config = pipe.generation_config();

    cout << endl << "greedy decoding with multiple batches:" << endl;
    std::vector<std::string> prompts = {"table is made of", "Alan Turing was a", "1 + 1 = ", "Why is the Sun yellow?"};
    auto results = pipe(prompts, config.max_new_tokens(20));
    for (int i = 0; i < prompts.size(); i++)
        cout << prompts[i] << ": " << results[i] << endl;

    // Example 4: Calling tokenizer/detokenizer manually and getting beam scores for all candidates
    pipe = LLMPipeline(model_path);
    auto [input_ids, attention_mask] = pipe.tokenize({prompt});
    config = GenerationConfig::beam_search();
    // config for grouped beam search
    config.max_new_tokens(30).num_groups(3).group_size(5).num_return_sequences(15);
    
    cout << endl << "beam search with printing of all candidates:" << endl;
    auto beams = pipe.generate(input_ids, attention_mask, config);
    for (const auto& beam : beams)
        std::cout << beam.first << ": " << pipe.detokenize(beam.second) << std::endl;

    {
        // Example 5: Speculative sampling
        std::string assitive_model_path = "text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16";
        pipe = LLMPipeline(model_path);
        auto [input_ids, attention_mask] = pipe.tokenize({prompt});
        config = GenerationConfig::assistive_decoding(assitive_model_path).num_assistant_tokens(5).max_new_tokens(20);
        
        auto results = pipe.generate(input_ids, attention_mask, config);
        for (const auto& beam : results)
            std::cout << pipe.detokenize(beam.second) << std::endl;
    }

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
