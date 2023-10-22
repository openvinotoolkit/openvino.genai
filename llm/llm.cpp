// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

namespace {
void tokenize(ov::InferRequest& tokenizer, const std::string& prompt) {
    constexpr size_t BATCH_SIZE = 1;
    constexpr size_t INDEXES_SIZE = (2 + BATCH_SIZE) * sizeof(int32_t);
    ov::Tensor destination = tokenizer.get_input_tensor();
    destination.set_shape({INDEXES_SIZE + prompt.length()});
    // B - batch size, E - end idx (and start for the next prompt). Tensor layout in bytes:
    // Bbbb0000EeeeEeeePrompt1Prompt2
    int32_t* int_ptr = reinterpret_cast<int32_t*>(destination.data<uint8_t>());
    int_ptr[0] = BATCH_SIZE;
    int_ptr[1] = 0;
    int_ptr[2] = int32_t(prompt.length());
    std::copy(prompt.begin(), prompt.end(), reinterpret_cast<char*>(int_ptr + 3));
    tokenizer.infer();
}

void print_token(ov::InferRequest& detokenizer, int32_t out_token) {
    constexpr size_t BATCH_SIZE = 1;
    constexpr size_t INDEXES_SIZE = (2 + BATCH_SIZE) * sizeof(int32_t);
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, 1});
    inp.data<int32_t>()[0] = out_token;
    detokenizer.infer();
    const ov::Tensor& detokenized = detokenizer.get_output_tensor();
    size_t tensor_size = detokenized.get_size();
    if (tensor_size <= INDEXES_SIZE) {
        throw std::runtime_error("The detokenized tensor must contain batch size, first string offset and end indices");
    }
    const char* char_ptr = reinterpret_cast<const char*>(detokenized.data<const uint8_t>());
    if (reinterpret_cast<const int32_t*>(char_ptr)[0] != BATCH_SIZE) {
        throw std::runtime_error("Expected batch 1 in the detokenized tensor");
    }
    std::cout.write(char_ptr + INDEXES_SIZE, std::streamsize(tensor_size - INDEXES_SIZE)).flush();
}
}

int main(int argc, char* argv[]) try {
    if (argc != 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<prompt>'");
    }
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in CMakeLists.txt
    ov::InferRequest tokenizer = core.compile_model(argv[2], "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    tokenize(tokenizer, argv[4]);
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<std::string, ov::PartialShape> shapes = {
        {"input_ids", ov::PartialShape{
            BATCH_SIZE, {1, std::numeric_limits<ov::Dimension::value_type>::max()}
        }},
        {"attention_mask", ov::PartialShape{
            BATCH_SIZE, {1, std::numeric_limits<ov::Dimension::value_type>::max()}
        }}
    };
    for (const ov::Output<ov::Node>& input : model->inputs()) {
        for (const std::string& name : input.get_names()) {
            if (name.rfind("past_key_values", 0) == 0) {
                ov::PartialShape shape = input.get_partial_shape();
                shape[0] = BATCH_SIZE;
                shapes.emplace(name, shape);
                break;
            }
        }
    }
    model->reshape(shapes);
    ov::preprocess::PrePostProcessor p3(model);
    p3.input("input_ids").tensor().set_element_type(ov::element::i32);  // cast to the type of tokenyzer's output
    p3.input("attention_mask").tensor().set_element_type(ov::element::i32);
    model = p3.build();
    ov::InferRequest ireq = core.compile_model(model, "CPU", {ov::cache_dir("llm-cache")}).create_infer_request();
    for (const ov::Output<ov::Node>& input : model->inputs()) {
        for (const std::string& name : input.get_names()) {
            if (name.rfind("past_key_values", 0) == 0) {
                ireq.get_tensor(input).set_shape(input.get_partial_shape().get_min_shape());
                break;
            }
        }
    }
    ireq.get_tensor("input_ids").set_shape(tokenizer.get_tensor("input_ids").get_shape());  // TODO: replace with ireq.set_tensor("input_ids", tokenizer.get_tensor("input_ids")); after it's fixed
    ireq.get_tensor("attention_mask").set_shape(tokenizer.get_tensor("input_ids").get_shape());
    std::copy_n(tokenizer.get_tensor("input_ids").data<int32_t>(), tokenizer.get_tensor("input_ids").get_size(), ireq.get_tensor("input_ids").data<int32_t>());
    std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), tokenizer.get_tensor("input_ids").get_size(), 1);
    ireq.infer();
    size_t n_vocab = ireq.get_tensor("logits").get_shape().back();
    float* logits = ireq.get_tensor("logits").data<float>() + (tokenizer.get_tensor("input_ids").get_size() - 1) * n_vocab;
    int32_t out_token = int32_t(std::max_element(logits, logits + n_vocab) - logits);

    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("attention_mask").data<int32_t>()[0] = 1;
    constexpr int32_t SPECIAL_EOS_TOKEN = 2;
    while (out_token != SPECIAL_EOS_TOKEN) {
        for (const ov::Output<ov::Node>& input : model->inputs()) {
            for (const std::string& name : input.get_names()) {
                if (name.rfind("past_key_values", 0) == 0) {
                    ireq.set_tensor(input, ireq.get_tensor("present" + name.substr(15)));
                    break;
                }
            }
        }
        ireq.get_tensor("input_ids").data<int32_t>()[0] = out_token;
        ireq.start_async();
        print_token(detokenizer, out_token);
        ireq.wait();
        logits = ireq.get_tensor("logits").data<float>();
        out_token = int32_t(std::max_element(logits, logits + n_vocab) - logits);
    }
    std::cout << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
