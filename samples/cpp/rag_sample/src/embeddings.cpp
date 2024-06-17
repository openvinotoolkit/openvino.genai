#include "embeddings.hpp"   
#include <openvino/openvino.hpp>


Embeddings::Embeddings (std::string bert_path, std::string device){
    std::string bert_model_path = (std::filesystem::path(bert_path) / "openvino_model.xml").string();
    std::string bert_tokenizer_path = (std::filesystem::path(bert_path) / "openvino_tokenizer.xml").string();
    init(bert_model_path, bert_tokenizer_path, device);
    std::cout << "Init embedding models successed\n";
}


std::vector<ov::Tensor> Embeddings::tokenize(std::string prompt) {
    constexpr size_t BATCH_SIZE = 1;
    auto input_tensor = ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt};

    // std::cout << "prompt length: " << prompt.length() << std::endl;
    // std::cout << "prompt: " << prompt << std::endl;
    // auto shape = input_tensor.get_shape();
    // std::cout << "input tensor shape: [ ";
    // for(auto s: shape){
    //     std::cout << s << " ";
    // }
    // std::cout << "]\n";

    try
    {
        tokenizer.set_input_tensor(input_tensor);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    // tokenizer.set_input_tensor(input_tensor);
    std::cout << "Set input tensor works\n";
    tokenizer.infer();
    
    std::cout << "Tokenizer infer works\n";
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask"), tokenizer.get_tensor("token_type_ids")};
}


inline ov::Tensor Embeddings::convert_inttensor_to_floattensor(ov::Tensor itensor) {
    ov::Shape shape = itensor.get_shape();
    ov::Tensor ftensor = ov::Tensor{ov::element::f32, itensor.get_shape()};
    std::copy_n(itensor.data<int64_t>(), itensor.get_size(), ftensor.data<float>());
    return ftensor;
}


inline ov::Tensor Embeddings::padding_for_fixed_input_shape(ov::Tensor input, ov::Shape shape) {
    ov::Tensor padded_input = ov::Tensor{ov::element::f32, shape};
    std::fill_n(padded_input.data<float>(), padded_input.get_size(), 0.0);
    std::copy_n(input.data<float>(), input.get_size(), padded_input.data<float>());
    return padded_input;
}


void Embeddings::init(std::string bert_path , std::string bert_tokenizer_path, std::string device){
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    //Read the tokenizer model information from the file to later get the runtime information
    embedding_model = core.compile_model(bert_path, device).create_infer_request();
    std::cout << "Load embedding model successed\n";
    // auto tokenizer_model = core.read_model(bert_tokenizer_path);
    tokenizer = core.compile_model(bert_tokenizer_path, device).create_infer_request();
    std::cout << "Load tokenizer model successed\n";
}


void Embeddings::run(std::string query){
    // run_bert_embeddings(query, embedding_model, tokenizer);
}


std::vector<std::vector<std::vector<float>>> Embeddings::run(std::vector<std::string> queries){
    std::cout << "size of queries: " << queries.size() << std::endl;
    std::vector<std::vector<std::vector<float>>> embedding_results;
    for(auto query: queries){
        std::vector<std::vector<float>> embedding_result = run_bert_embeddings(query);
        embedding_results.push_back(embedding_result);
    }
    std::cout << "size of embedding_results: " << embedding_results.size() << std::endl;
    std::cout << "size of embedding_results0: " << embedding_results[0].size() << std::endl;
    std::cout << "size of embedding_results00: " << embedding_results[0][0].size() << std::endl;
    return embedding_results;
}


std::vector<std::vector<float>> Embeddings::run_bert_embeddings(std::string query){
    //tokenize
    auto tokenied_output = tokenize(query);

    auto input_ids = convert_inttensor_to_floattensor(tokenied_output[0]); 
    auto attention_mask = convert_inttensor_to_floattensor(tokenied_output[1]); 
    auto token_type_ids = convert_inttensor_to_floattensor(tokenied_output[2]); 

    auto input_ids_padding = padding_for_fixed_input_shape(input_ids, ov::Shape{1, 512});
    auto attention_mask_padding = padding_for_fixed_input_shape(attention_mask, ov::Shape{1, 512});
    auto token_type_ids_padding = padding_for_fixed_input_shape(token_type_ids, ov::Shape{1, 512});

    std::cout << "tokenize encode successed\n";
    auto seq_len = input_ids.get_size();
     
    // Initialize inputs
    embedding_model.set_tensor("input_ids", input_ids_padding);
    embedding_model.set_tensor("attention_mask", attention_mask_padding);
    embedding_model.set_tensor("token_type_ids", token_type_ids_padding);
    // ov::Tensor token_type_ids = embedding_model.get_tensor("token_type_ids");
    // token_type_ids.set_shape(input_ids.get_shape());
    // std::iota(token_type_ids.data<int64_t>(), token_type_ids.data<int64_t>() + seq_len, 0);
    constexpr size_t BATCH_SIZE = 1;
    embedding_model.infer();
    auto res = embedding_model.get_tensor("last_hidden_state");

    auto shape = res.get_shape();
    // std::cout << "res shape: " << shape<< std::endl;
    
    std::vector<std::vector<float>> embedding_result;
    float *output_buffer = res.data<float>();
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            std::vector<float> tmp(shape[2]);
            for (size_t k = 0; k < shape[2]; k++) {
                // std::cout << output_buffer[i, j, k] << " ";
                tmp[k] = output_buffer[i, j, k];
            }
            embedding_result.push_back(tmp);
            // std::cout << std::endl;
        }
    }
    std::cout << "embedding infer successed\n";
    return embedding_result;
}

