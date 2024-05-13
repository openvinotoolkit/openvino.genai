// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
    if (2 > argc && argc > 4)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" <DEVICE>");
    std::string model_path = argv[1];
    
    std::string prompt = "table is made of ";
    std::string device = "CPU"; // can be replaced with GPU

    if (argc > 2)
        prompt = argv[2];
    if (argc > 3)
        device = argv[3];
    
    // Example 1: Simplest example with greedy search
    // Model, tokenizer and generation_config.json will be loaded from the model_path.
    // If generation_config.json is not found default velues for gready search will be used
    
    // ov::streamer_lambda([](std::string subword){std::cout << subword << std::flush;})
    ov::LLMPipeline pipe(model_path, device);
    // cout << prompt << pipe(prompt, ov::max_new_tokens(1000)) << endl;

    // todo: syntactic sugar to specify generation configs in place
    // cout << prompt << pipe(prompt, ov::max_new_tokens(100)) << endl;


    auto tokenizer = ov::Tokenizer(model_path);
    auto [input_ids, attention_mask] = tokenizer.encode("table is made of ");
    auto resuling_tokens = pipe.generate(input_ids, ov::max_new_tokens(1000));
    cout << tokenizer.decode(resuling_tokens.tokens[0]) << endl;

    // Example 2: Modifying generation_cofnig to use grouped beam search
    ov::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 100;
    config.num_beams = 15;
    config.num_beam_groups = 3;
    // cout << prompt << pipe(prompt, config) << endl;

    // cout << endl << "grouped beam search generated candidates:" << endl;
    // for (int i = 0; i < num_return_sequences; ++i)
    // will return vector with num_return_sequences strings
    // auto num_return_sequences = 3;
    
    // // Example 3: Greedy Decoding with multiple batch
    // pipe = ov::LLMPipeline(model_path, device);
    // config = pipe.generation_config();

    // cout << endl << "greedy decoding with multiple batches:" << endl;
    // std::vector<std::string> prompts = {"table is made of", "Alan Turing was a", "1 + 1 = ", "Why is the Sun yellow?"};
    // auto results = pipe(prompts, config.max_new_tokens(20));
    // for (const auto& res: results)
    //     std::cout << res.text << std::endl;

    // // Example 4: Calling tokenizer/detokenizer manually and getting beam scores for all candidates
    // pipe = ov::LLMPipeline(model_path);
    // auto [input_ids, attention_mask] = pipe.get_tokenizer().tokenize({prompt});
    // config = GenerationConfig::beam_search();
    // // config for grouped beam search
    // config.max_new_tokens(30).num_groups(3).group_size(5).num_return_sequences(15);
    
    // cout << endl << "beam search with printing of all candidates:" << endl;
    // auto beams = pipe.generate(input_ids, attention_mask, config);
    // for (size_t i = 0; i < beams.scores.size(); i++) {
    //     std::cout << beams.scores[i] << ": " << pipe.get_tokenizer().detokenize(beams.tokens[i]) << std::endl;
    // }

    // // for (const auto& beam : beams.second)
    // //     std::cout << beam.first << ": " << pipe.detokenize(beam.second) << std::endl;

    // {
    //     // Example 5: Speculative sampling
    //     std::string assitive_model_path = "text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16";
    //     pipe = ov::LLMPipeline(model_path);
    //     auto [input_ids, attention_mask] = pipe.get_tokenizer().tokenize({prompt});
    //     // config = GenerationConfig::assistive_decoding(assitive_model_path).num_assistant_tokens(5).max_new_tokens(20);
    //     pipe.generation_config().assistant_model(assitive_model_path);
        
    //     cout << endl << "Speculative sampling with TinyLlama assistance:" << endl;
    //     auto results = pipe.generate(input_ids, attention_mask, config);
    //     for (size_t i = 0; i < beams.scores.size(); i++) {
    //     for (const auto& result : results)
    //         std::cout << pipe.get_tokenizer().detokenize(result.tokens) << std::endl;
    //     }
    // }

    return 0;
}
