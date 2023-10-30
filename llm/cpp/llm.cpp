// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <utils.hpp>

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest&& tokenizer, std::string_view prompt) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor destination = tokenizer.get_input_tensor();
    pack_strings(std::array<std::string_view, BATCH_SIZE>{prompt}, destination);
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void print_token(ov::InferRequest& detokenizer, int32_t out_token) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, 1});
    inp.data<int32_t>()[0] = out_token;
    detokenizer.infer();
    std::cout << unpack_strings(detokenizer.get_output_tensor()).front() << std::flush;
}
}

int main(int argc, char* argv[]) try {
    if (argc != 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<prompt>'");
    }
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    auto [input_ids, attention_mask] = tokenize(core.compile_model(argv[2], "CPU").create_infer_request(), argv[4]);
    ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            BATCH_SIZE, {1, std::numeric_limits<ov::Dimension::value_type>::max()}
        }},
        {1, ov::PartialShape{
            BATCH_SIZE, {1, std::numeric_limits<ov::Dimension::value_type>::max()}
        }}
    };
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (size_t idx = 2; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }
    model->reshape(shapes);
    ov::preprocess::PrePostProcessor p3(model);
    p3.input("input_ids").tensor().set_element_type(ov::element::i32);  // cast to the type of tokenyzer's output
    p3.input("attention_mask").tensor().set_element_type(ov::element::i32);
    model = p3.build();
    ov::InferRequest ireq = core.compile_model(model, "CPU", {ov::cache_dir("llm-cache")}).create_infer_request();
    for (size_t idx = 2; idx < inputs.size(); ++idx) {
        ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
    }
    ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
    ireq.get_tensor("attention_mask").set_shape(input_ids.get_shape());
    std::copy_n(input_ids.data<const int32_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int32_t>());
    std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), input_ids.get_size(), 1);
    ireq.infer();
    size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
    float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    int32_t out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);

    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("attention_mask").data<int32_t>()[0] = 1;
    constexpr int32_t SPECIAL_EOS_TOKEN = 2;  // There's no way to extract the value from the tokenizer for now
    while (out_token != SPECIAL_EOS_TOKEN) {
        for (size_t idx = 2; idx < inputs.size(); ++idx) {
             ireq.set_input_tensor(idx, ireq.get_output_tensor(idx - 1));
        }
        ireq.get_tensor("input_ids").data<int32_t>()[0] = out_token;
        ireq.start_async();
        print_token(detokenizer, out_token);
        ireq.wait();
        logits = ireq.get_tensor("logits").data<float>();
        out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
    }
    std::cout << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
