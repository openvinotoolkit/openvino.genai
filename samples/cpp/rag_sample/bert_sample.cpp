// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <sstream>

// #include <openvino/openvino.hpp>
// #include "openvino/genai/llm_pipeline.hpp"
// #include "json.hpp"   


// ov::Core core;

// ov::InferRequest embedding_model;
// ov::InferRequest tokenizer;

// std::vector<ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string prompt) {
//     constexpr size_t BATCH_SIZE = 1;
//     auto input_tensor = ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt};
//     std::cout << "prompt: " << prompt;
//     //auto input_tensor = ov::Tensor{ov::element::string, {prompt.length() + 1}, &prompt};
//     auto shape = input_tensor.get_shape();
//     std::cout << "input tensor shape: [ ";
//     for(auto s: shape){
//         std::cout << s << " ";
//     }
//     std::cout << "]\n";

//     tokenizer.set_input_tensor(input_tensor);
//     tokenizer.infer();
    
//     std::cout << "Tokenizer infer works\n";
//     return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask"), tokenizer.get_tensor("token_type_ids")};
// }


// ov::Tensor convert_inttensor_to_floattensor(ov::Tensor itensor) {
//     ov::Shape shape = itensor.get_shape();
//     ov::Tensor ftensor = ov::Tensor{ov::element::f32, itensor.get_shape()};
//     std::copy_n(itensor.data<int64_t>(), itensor.get_size(), ftensor.data<float>());
//     return ftensor;
// }


// ov::Tensor padding_for_fixed_input_shape(ov::Tensor input, ov::Shape shape) {
//     ov::Tensor padded_input = ov::Tensor{ov::element::f32, shape};
//     std::fill_n(padded_input.data<float>(), padded_input.get_size(), 0.0);
//     std::copy_n(input.data<float>(), input.get_size(), padded_input.data<float>());
//     return padded_input;
// }

// void init_bert_embedding(std::string bert_path , std::string bert_tokenizer_path){
//     core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
//     //Read the tokenizer model information from the file to later get the runtime information
//     embedding_model = core.compile_model(bert_path, "CPU").create_infer_request();
//     std::cout << "Load embedding model successed\n";
//     auto tokenizer_model = core.read_model(bert_tokenizer_path);
//     tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
//     std::cout << "Load tokenizer model successed\n";
//     std::cout << "Init BERT successed\n";
// }


// void run_bert_embedding(std::string query, ov::InferRequest embedding_model, ov::InferRequest tokenizer){
//     //tokenize
//     auto tokenied_output = tokenize(tokenizer, query);

//     auto input_ids = convert_inttensor_to_floattensor(tokenied_output[0]); 
//     auto attention_mask = convert_inttensor_to_floattensor(tokenied_output[1]); 
//     auto token_type_ids = convert_inttensor_to_floattensor(tokenied_output[2]); 

//     auto input_ids_padding = padding_for_fixed_input_shape(input_ids, ov::Shape{1, 512});
//     auto attention_mask_padding = padding_for_fixed_input_shape(attention_mask, ov::Shape{1, 512});
//     auto token_type_ids_padding = padding_for_fixed_input_shape(token_type_ids, ov::Shape{1, 512});

//     std::cout << "tokenize encode successed\n";
//     auto seq_len = input_ids.get_size();
     
//     // Initialize inputs
//     embedding_model.set_tensor("input_ids", input_ids_padding);
//     embedding_model.set_tensor("attention_mask", attention_mask_padding);
//     embedding_model.set_tensor("token_type_ids", token_type_ids_padding);
//     // ov::Tensor token_type_ids = embedding_model.get_tensor("token_type_ids");
//     // token_type_ids.set_shape(input_ids.get_shape());
//     // std::iota(token_type_ids.data<int64_t>(), token_type_ids.data<int64_t>() + seq_len, 0);
//     constexpr size_t BATCH_SIZE = 1;
//     embedding_model.infer();
//     auto res = embedding_model.get_tensor("last_hidden_state");

//     // auto shape = res.get_shape();
//     // std::cout << "res shape: " << shape<< std::endl;
//     // float *output_buffer = res.data<float>();
//     // for (size_t i = 0; i < shape[0]; i++) {
//     //     for (size_t j = 0; j < shape[1]; j++) {
//     //         for (size_t k = 0; k < shape[2]; k++) {
//     //             std::cout << output_buffer[i, j, k] << " ";
//     //         }
//     //         std::cout << std::endl;
//     //     }
//     // }

//     std::cout << "embedding infer successed\n";
// }


// int main(int argc, char** argv) {
//     if (argc != 4) {
//         std::cerr << "Usage: ./single_sample <query> <bert_model> <tokenizer>\n";
//         return -1;
//     }

//     init_bert_embedding(argv[2], argv[3]);
//     run_bert_embedding(argv[1], embedding_model, tokenizer);
// }


#include "embeddings.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./single_sample <query> <bert_model_path> \n";
        return -1;
    }

    Embeddings embedding(argv[2], "CPU");
    embedding.run(argv[1]);
}
