// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "generate_pipeline.hpp"

namespace {

constexpr size_t BATCH_SIZE = 1;

}  // namespace

using namespace std;

int main(int argc, char* argv[]) try {
    {
        // PIPELINE Ex.1
        std::string model_path = "text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/";
        LLMPipeline pipe(model_path);
        std::cout << pipe("table is made of");
    }
    cout << endl <<  "-------------END OF GENERATE ------" << endl;

    {
        // PIPELINE Ex.2
        std::string model_path = "text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/";
        LLMPipeline pipe(model_path);
        GenerationConfig config = pipe.generation_config();
        // batched inputs
        auto results = pipe({"table is made of", 
                            "Alan Turing was a",
                            "1 + 1 = ",
                            "Why is the Sun yellow?"
                            }, config.do_sample(false));
        
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
