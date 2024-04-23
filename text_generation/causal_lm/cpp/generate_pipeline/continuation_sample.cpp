// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    std::string prompt = "table is made of";
    std::string device = "CPU";

    std::string model_path = argv[1];
    if (argc > 2)
        prompt = argv[2];
    if (argc > 3)
        device = argv[3];
    vector<int64_t> all_results;

    LLMPipeline pipe(model_path, device);
    GenerationConfig config = pipe.generation_config();
    Tokenizer tokenizer = pipe.get_tokenizer();
    config.eos_token_id(2);

    ov::Tensor input_ids, attention_mask;
    std::tie(input_ids, attention_mask) = tokenizer.tokenize(prompt);
    // max_new_tokens should be 15 for reproducer case
    auto result = pipe.generate(input_ids, attention_mask, config.reset_state(false).max_new_tokens(55), true)[0].second;
    all_results.insert(all_results.end(), result.begin(), result.end());

    string text = tokenizer.detokenize(result);
    cout << text << endl;

    auto new_input_ids = ov::Tensor{ov::element::i64, {1, 1}};
    auto new_attention_mask = ov::Tensor{ov::element::i64, {1, 1}};
    auto data = new_attention_mask.data<int64_t>();
    data[0] = 1;
    data = new_input_ids.data<int64_t>();
    data[0] = result.back();
    auto new_result = pipe.generate(new_input_ids, new_attention_mask, config.reset_state(false).max_new_tokens(1000), true)[0].second;
    all_results.insert(all_results.end(), new_result.begin(), new_result.end());
    cout << tokenizer.detokenize(all_results);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
