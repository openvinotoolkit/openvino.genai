// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "generate_pipeline.hpp"

namespace {

constexpr size_t BATCH_SIZE = 1;

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
    
    LLMPipeline pipe(model_path);
    std::cout << pipe.call("Alan Turing was a");
    
    // ov::Core core;
    // // core.add_extension("libuser_ov_extensions.so");
    // core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    // // tokenizer and detokenizer work on CPU only
    // ov::InferRequest tokenizer = core.compile_model(model_path + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    // ov::InferRequest detokenizer = core.compile_model(model_path + "/openvino_detokenizer.xml", "CPU").create_infer_request();

    // // The model can be compiled for GPU as well
    // std::shared_ptr<ov::Model> model = core.read_model(model_path + "/openvino_model.xml");
    // ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();
    
    // auto [input_ids, attention_mask] = tokenize(tokenizer, argv[1]);
    
    // SamplingParameters sampling_params = SamplingParameters::greedy();

    // LLMEngine engine(request);
    // GenerationResult generation_results = engine.generate(input_ids, sampling_params);
    // print_generation_results(generation_results, detokenizer);

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
