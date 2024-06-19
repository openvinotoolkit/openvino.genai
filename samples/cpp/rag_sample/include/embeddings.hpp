#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

#include <openvino/openvino.hpp>
#include "json.hpp"   


class Embeddings{
    public:
        Embeddings(std::string bert_path, std::string device);
        ~Embeddings() = default;

        ov::Core core;

        ov::InferRequest embedding_model;
        ov::InferRequest tokenizer;

        void init(std::string bert_path , std::string bert_tokenizer_path, std::string device);
        std::vector<std::vector<std::vector<float>>> encode_queries(std::vector<std::string> queries);

    private:
    
        size_t BATCH_SIZE = 1;
        std::vector<std::vector<float>> encode_query(std::string query);
        std::vector<ov::Tensor> tokenize(std::string prompt);

        inline ov::Tensor convert_inttensor_to_floattensor(ov::Tensor itensor);
        inline ov::Tensor padding_for_fixed_input_shape(ov::Tensor input, ov::Shape shape);
};




