// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino_extensions/strings.hpp>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest & tokenizer, std::string_view prompt) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor destination = tokenizer.get_input_tensor();
    openvino_extensions::pack_strings(std::array<std::string_view, BATCH_SIZE>{prompt}, destination);
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void print_token(ov::InferRequest& detokenizer, int64_t out_token) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, 1});
    inp.data<int64_t>()[0] = out_token;
    detokenizer.infer();
    std::cout << openvino_extensions::unpack_strings(detokenizer.get_output_tensor()).front() << std::flush;
}

static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

}

int main(int argc, char* argv[]) try {
    if (argc != 6) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<device>' '<prompt>'");
    }
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    //auto [input_ids, attention_mask] = tokenize(core.compile_model(argv[2], "CPU").create_infer_request(), argv[4]);
    ov::InferRequest tokenizer = core.compile_model(argv[2], "CPU").create_infer_request();
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[5]);
    ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            BATCH_SIZE, -1
        }},
        {1, ov::PartialShape{
            BATCH_SIZE, -1
        }},
        {2, ov::PartialShape{
	    BATCH_SIZE, -1
        }}
    };

    
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[1] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }

    model->reshape(shapes);
    //ov::InferRequest ireq = core.compile_model(model, "CPU", ov::cache_dir("llm-cache")).create_infer_request();
    double total_time = 0;
    int count = 0;
    auto startTime = Time::now();
    ov::InferRequest ireq = core.compile_model(model, argv[4]).create_infer_request();
    auto duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Compile LLM model took " << duration_ms << " ms" << std::endl;
   
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
    }

    ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
    ireq.get_tensor("attention_mask").set_shape(attention_mask.get_shape());
    std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int64_t>());
    std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), attention_mask.get_size(), 1);
    ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
    std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);

    startTime = Time::now();
    ireq.infer();
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "First token took " << duration_ms << " ms" << std::endl;

    size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
    float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});

    constexpr int64_t SPECIAL_EOS_TOKEN = 2;  // There's no way to extract the value from the detokenizer for now
    while (out_token != SPECIAL_EOS_TOKEN) {
        startTime = Time::now();
        ireq.get_tensor("input_ids").data<int64_t>()[0] = out_token;
        ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1});
        std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
        ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;

        for (size_t idx = 3; idx < inputs.size(); ++idx) {
            ireq.set_input_tensor(idx, ireq.get_output_tensor(idx - 2));
        }
        ireq.start_async();
        ireq.wait();
        duration_ms = get_duration_ms_until_now(startTime);
        count += 1;
        total_time += duration_ms;

        print_token(detokenizer, out_token);
        logits = ireq.get_tensor("logits").data<float>();
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }
    std::cout << '\n';

    if (count > 0) {
        std::cout << "Other Avg inference took total " << total_time << " ms token num " << count << " avg " << total_time / (count) << " ms" << std::endl;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
