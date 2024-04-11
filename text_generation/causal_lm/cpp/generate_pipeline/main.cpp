// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "generate_pipeline.hpp"

namespace {

constexpr size_t BATCH_SIZE = 1;

}  // namespace

using namespace std;

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
    {
        // PIPELINE Ex.1
        std::string model_path = "text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/";
        LLMPipeline pipe(model_path, "CPU");
        GenerationConfig config = pipe.generation_config();

        auto text_streamer = TextStreamer{pipe};
        auto print_text_callback = [&text_streamer](std::vector<int64_t>&& tokens, LLMPipeline& pipe){
            text_streamer.put(tokens[0]);
        };

        pipe("table is made of", config.max_new_tokens(100).set_callback(print_text_callback));
        text_streamer.end();
        cout << endl <<  "------------- END OF GENERATE -------------" << endl;
    }

    {
        // PIPELINE Ex.2
        std::string model_path = "text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/";
        LLMPipeline pipe(model_path, "CPU");
        GenerationConfig config = pipe.generation_config();
        // batched inputs
        auto results = pipe({"table is made of", 
                            "Alan Turing was a",
                            "1 + 1 = ",
                            "Why is the Sun yellow?"
                            }, config.do_sample(false).max_new_tokens(100));
        
        for (const auto& res: results) {
            cout << res << endl;
            cout << "-------------------" << endl;
        }
    }

    // GENERATE
    std::string model_path = "text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/";
    LLMPipeline pipe(model_path);
    auto [input_ids, attention_mask] = pipe.tokenize("table is made of");
    auto res = pipe.generate(input_ids, attention_mask);
    std::cout << pipe.detokenize(res)[0];
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
