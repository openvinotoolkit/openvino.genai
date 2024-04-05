// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "generate_pipeline.hpp"

namespace {

constexpr size_t BATCH_SIZE = 1;

}  // namespace

int main(int argc, char* argv[]) try {
    // PIPELINE ex.1
    std::string model_path = "/home/epavel/devel/openvino.genai/text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/";
    // LLMPipeline pipe(model_path);
    // std::cout << pipe.call("Alan Turing was a");

    // {   
    //     // PIPELINE ex.2
    //     LLMPipeline pipe(model_path);
    //     GenerationConfig config = pipe.generation_config();
        
    //     std::cout << pipe("Alan Turing was a", config.temperature(0.2).top_k(4).do_sample(true).repetition_penalty(1.2));
        
    //     // batched inputs
    //     auto results = pipe({"table is made of ", 
    //                         "Alan Turing was a", 
    //                         "1 + 1 = ",
    //                         "Why is the Sun yellow?"
    //                         }, config.temperature(0.2).top_k(4).do_sample(true).repetition_penalty(1.2));
        
    //     for (const auto& res: results) {
    //         std::cout << res << std::endl;
    //     }
    // }
    {

GenerationConfig config = GenerationConfig().group_size(3).num_groups(3).diversity_penalty(1.2);
config.max_length(100);

    }

    {
GenerationConfig config;
config.m_bos_token_id = 0;
config.m_num_groups = 3;
config.m_group_size = 5;
config.m_max_new_tokens = 100;
    }

    // GENERATE
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    ov::InferRequest tokenizer = core.compile_model(model_path + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(model_path + "/openvino_detokenizer.xml", "CPU").create_infer_request();

    // todo: beam search does not work properly on GPU, when reshaped from batch=1 to batch=num_beams not broadcasted
    std::shared_ptr<ov::Model> model = core.read_model(model_path + "/openvino_model.xml");
    ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();
    
    auto [input_ids, attention_mask] = tokenize(tokenizer, "Alan Turing was a");
    GenerationConfig sampling_params = GenerationConfig::beam_search();
    LLMModel engine(request);
    GenerationResult generation_results = engine.generate(input_ids, sampling_params);
    std::cout << detokenize(detokenizer, generation_results[0]);
    
    
    // LLMModel engine(request);
    // GenerationConfig params.temperature(0.2).top_k(4).do_sample(true).repetition_penalty(1.2);
    // GenerationResult generation_results = engine.generate(input_ids, params);
    // std::cout << detokenize(detokenizer, generation_results[0]);

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
