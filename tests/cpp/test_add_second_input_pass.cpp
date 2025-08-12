// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/add_second_input_pass.hpp"
#include "tokenizer/tokenizers_path.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"
#include <gtest/gtest.h>
#include <openvino/pass/manager.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/select.hpp>
#include <openvino/op/equal.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/multiply.hpp>
#include <filesystem>
#include <memory>


using namespace ov::genai;
using namespace ov;
using namespace ov::op;

static FactoryCreateType create_func = nullptr;
static std::shared_ptr<void> shared_object_ov_tokenizers = nullptr;

static void initlize_shared_ov_tokenziers() {
    // Get the parent path two levels up and append "openvino_genai"
    auto genai_root = std::filesystem::path(tokenizers_relative_to_genai());
    auto openvino_genai_path = genai_root.parent_path().parent_path() / "openvino_genai" / genai_root.filename();
    ScopedVar env_manager(openvino_genai_path.string());
    std::filesystem::path ov_tokenizer_filesystem_path;

    #ifdef _WIN32
            const wchar_t* ov_tokenizer_path = _wgetenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME_W);
    #else
            const char* ov_tokenizer_path = getenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME);
    #endif

    if (std::filesystem::exists(ov_tokenizer_path)) {
        shared_object_ov_tokenizers = load_shared_object(ov_tokenizer_path);
    }
}

static FactoryCreateType get_factory_create_func() {
    if (create_func != nullptr) {
        return create_func;
    }

    if (shared_object_ov_tokenizers == nullptr) {
        initlize_shared_ov_tokenziers();
    }

    if (shared_object_ov_tokenizers != nullptr) {
        create_func = reinterpret_cast<FactoryCreateType>(get_symbol(shared_object_ov_tokenizers, "create_tokenizer_node"));
        return create_func;
    }

    // If we made it here then we failed to load the shared object
    OPENVINO_THROW("Failed to load the shared object for tokenizer factory creation function.");
}


TEST(AddSecondInputTest, add_second_input_test_1) {
    std::shared_ptr<Model> model;
    auto parameter_1 = std::make_shared<v0::Parameter>(element::string, Shape{2});
    OutputVector outputs = get_factory_create_func()("StringTensorUnpack", {parameter_1}, {});

    // Prepare all necessary BPETokenizer inputs according to evaluate() logic

    auto ragged_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto ragged_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{3});
    
    auto vocab_begins = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, 1});
    auto vocab_ends = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{1, 2});
    auto vocab_chars = std::make_shared<v0::Constant>(element::u8, Shape{2}, std::vector<uint8_t>{'a', 'b'});
    
    auto merges_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto merges_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{3});
    auto merges_chars = std::make_shared<v0::Constant>(element::u8, Shape{3}, std::vector<uint8_t>{'a', 'b', 'c'});

    // Compose the input vector for BPETokenizer (11 inputs for minimal case)
    std::vector<ov::Output<ov::Node>> bpe_inputs = {
        ragged_begins, ragged_ends, outputs[0], outputs[1], outputs[2],
        vocab_begins, vocab_ends, vocab_chars,
        merges_begins, merges_ends, merges_chars
    };

    auto BPETokenizer = get_factory_create_func()("BPETokenizer", bpe_inputs, {});

    auto max_length = std::make_shared<v0::Constant>(element::i32, Shape{}, std::vector<int32_t>({10}));
    // Create trunc_side constant: "right" as u8 chars
    auto trunc_side = std::make_shared<v0::Constant>(
        element::u8, Shape{5}, std::vector<uint8_t>{'r', 'i', 'g', 'h', 't'}
    );
    auto truncate = get_factory_create_func()("Truncate", {BPETokenizer[0], BPETokenizer[1], BPETokenizer[2], max_length, trunc_side}, {});

    int32_t eos_token_id = 42;
    auto eos_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto eos_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{1});
    auto eos_token = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{eos_token_id});

    // IDs tensor (required as the last input for CombineSegment)
    auto ids = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, 1});

    // Prepare CombineSegment inputs: [begins1, ends1, data1, begins2, ends2, data2, ..., ids]
    ov::OutputVector combine_inputs = {
        // Main BPETokenizer outputs which went through Truncate
        truncate[0], truncate[1], truncate[2],
        // EOS token tensors
        eos_begins, eos_ends, eos_token,
        // IDs tensor
        ids
    };

    auto CombineSegments = get_factory_create_func()("CombineSegments", {combine_inputs}, {});
    model = std::make_shared<ov::Model>(OutputVector{CombineSegments}, ParameterVector{parameter_1});
    
    // Add a valid post_processor to model's rt_info
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    // Signature for single input [first_seq, eos]
    std::vector<int> input_signature = {-1, eos_token_id};
    // Signature for pair [first_seq, eos_seq, second_seq, eos]
    std::vector<int> pair_signature = {-1, eos_token_id, -1, eos_token_id};
    
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };

    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();

    std::ostringstream pass_errors;
    
    ov::pass::Manager manager;
    manager.register_pass<ov::genai::AddSecondInputPass>(shared_object_ov_tokenizers, pass_errors);
    manager.run_passes(model);

    // Check that resulting graph contains required fields.
    ASSERT_EQ(model->get_parameters().size(), 2);
    // Find combine segment op and assert that it has 2 extra inputs
    auto results = model->get_results();
    // ASSERT_EQ(results.size(), 1);
    auto combine_segments_node = results[0]->get_input_node_shared_ptr(0);
    ASSERT_NE(combine_segments_node, nullptr);
    ASSERT_EQ(combine_segments_node->get_input_size(), combine_inputs.size() + 6); // 6 extra inputs for the second input and eos token
}

// Additional negative tests for AddSecondInputPass error cases

// Helper to run the pass and return result (false if pass failed)
static bool run_add_second_input_pass(std::shared_ptr<ov::Model> model, std::ostringstream& pass_errors) {
    ov::pass::Manager manager;
    manager.register_pass<ov::genai::AddSecondInputPass>(shared_object_ov_tokenizers, pass_errors);
    manager.run_passes(model);
    
    return pass_errors.str().empty();
}

// Helper to create a minimal valid model for negative tests
static std::shared_ptr<ov::Model> make_minimal_model(
    bool with_combine_segments = true,
    bool with_string_unpack = true,
    std::vector<ov::Output<ov::Node>>* combine_inputs_out = nullptr)
{
    using namespace ov::op;
    auto parameter_1 = std::make_shared<v0::Parameter>(element::string, Shape{2});
    OutputVector outputs;
    if (with_string_unpack)
        outputs = get_factory_create_func()("StringTensorUnpack", {parameter_1}, {});
    else
        outputs = {parameter_1, parameter_1, parameter_1};

    auto ragged_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto ragged_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{3});
    auto vocab_begins = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, 1});
    auto vocab_ends = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{1, 2});
    auto vocab_chars = std::make_shared<v0::Constant>(element::u8, Shape{2}, std::vector<uint8_t>{'a', 'b'});
    auto merges_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto merges_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{3});
    auto merges_chars = std::make_shared<v0::Constant>(element::u8, Shape{3}, std::vector<uint8_t>{'a', 'b', 'c'});

    std::vector<ov::Output<ov::Node>> bpe_inputs = {
        ragged_begins, ragged_ends, outputs[0], outputs[1], outputs[2],
        vocab_begins, vocab_ends, vocab_chars,
        merges_begins, merges_ends, merges_chars
    };
    auto BPETokenizer = get_factory_create_func()("BPETokenizer", bpe_inputs, {});
    auto max_length = std::make_shared<v0::Constant>(element::i32, Shape{}, std::vector<int32_t>({10}));
    auto trunc_side = std::make_shared<v0::Constant>(element::u8, Shape{5}, std::vector<uint8_t>{'r', 'i', 'g', 'h', 't'});
    auto truncate = get_factory_create_func()("Truncate", {BPETokenizer[0], BPETokenizer[1], BPETokenizer[2], max_length, trunc_side}, {});
    int32_t eos_token_id = 42;
    auto eos_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto eos_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{1});
    auto eos_token = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{eos_token_id});
    auto ids = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, 1});
    ov::OutputVector combine_inputs = {
        truncate[0], truncate[1], truncate[2],
        eos_begins, eos_ends, eos_token,
        ids
    };
    if (combine_inputs_out) *combine_inputs_out = combine_inputs;
    std::shared_ptr<ov::Node> combine_node;
    if (with_combine_segments)
        combine_node = get_factory_create_func()("CombineSegments", {combine_inputs}, {})[0].get_node_shared_ptr();
    else
        combine_node = truncate[0].get_node_shared_ptr();
    auto model = std::make_shared<ov::Model>(OutputVector{combine_node}, ParameterVector{parameter_1});
    return model;
}

// 1. Model with more than one input
TEST(AddSecondInputTest, error_multiple_inputs) {
    auto p1 = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{2});
    auto p2 = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{2});
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{c}, ov::ParameterVector{p1, p2});
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
}

// 2. Model with no CombineSegments node
TEST(AddSecondInputTest, error_no_combine_segments) {
    auto model = make_minimal_model(/*with_combine_segments=*/false);
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
}

// 3. parse_inputs: begin is Constant but data is not Constant
TEST(AddSecondInputTest, error_parse_inputs_data_not_constant) {
    // Replace data input with a Parameter (not Constant)
    std::vector<ov::Output<ov::Node>> combine_inputs;
    auto model = make_minimal_model(true, true, &combine_inputs);
    // fake param is not a Constant, so it will fail the check
    auto fake_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    combine_inputs[2] = fake_param;
    auto combine_node = get_factory_create_func()("CombineSegments", {combine_inputs}, {})[0];
    auto parameter_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{2});

    auto new_inputs = model->get_parameters();
    new_inputs.emplace_back(fake_param);
    auto model2 = std::make_shared<ov::Model>(ov::OutputVector{combine_node}, new_inputs);
    // Add valid post_processor
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {-1, 42};
    std::vector<int> pair_signature = {-1, 42, -1, 42};
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };
    model2->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model2, pass_errors);
    ASSERT_FALSE(ok);
}

// 4. parse_inputs: begin is not Truncate node
TEST(AddSecondInputTest, error_parse_inputs_begin_not_truncate) {
    // Replace begin input with a Parameter (not Truncate)
    std::vector<ov::Output<ov::Node>> combine_inputs;
    auto model = make_minimal_model(true, true, &combine_inputs);
    auto fake_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    combine_inputs[0] = fake_param;
    auto combine_node = get_factory_create_func()("CombineSegments", {combine_inputs}, {})[0];
    auto parameter_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{2});
    auto new_inputs = model->get_parameters();
    new_inputs.emplace_back(fake_param);
    auto model2 = std::make_shared<ov::Model>(ov::OutputVector{combine_node}, new_inputs);
    // Add valid post_processor
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {-1, 42};
    std::vector<int> pair_signature = {-1, 42, -1, 42};
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };
    model2->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model2, pass_errors);
    ASSERT_FALSE(ok);
}

// 5. parse_and_assert_postprocessor: no post_processor in rt_info
TEST(AddSecondInputTest, error_no_post_processor) {
    auto model = make_minimal_model();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
    ASSERT_NE(pass_errors.str().find("Post processor is not present"), std::string::npos);
}

// 6. parse_and_assert_postprocessor: post_processor missing "pair"
TEST(AddSecondInputTest, error_post_processor_no_pair) {
    auto model = make_minimal_model();
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {-1, 42};
    nlohmann::json post_processor = {{"single", {{"ids", input_signature}}}};
    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
    ASSERT_NE(pass_errors.str().find("Could not add second input. post_processor does not contain input signature for paired input"), std::string::npos);
}

// 7. parse_and_assert_postprocessor: input_signature mismatch
TEST(AddSecondInputTest, error_post_processor_input_signature_mismatch) {
    auto model = make_minimal_model();
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {-1, 42};
    std::vector<int> wrong_input_signature = {-1, 99};
    std::vector<int> pair_signature = {-1, 42, -1, 42};
    nlohmann::json post_processor = {
        {"single", {{"ids", wrong_input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };
    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
}

// 8. parse_and_assert_postprocessor: pair signature not widening single
TEST(AddSecondInputTest, error_post_processor_pair_not_widening) {
    auto model = make_minimal_model();
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {-1, 42};
    std::vector<int> pair_signature = {99, 42, -1, 42}; // first two do not match input_signature
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };
    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
    ASSERT_NE(pass_errors.str().find("Paired inputs are allowed only when it's widening the single input"), std::string::npos);
}

// 9. parse_and_assert_postprocessor: not exactly 2 sequence inputs in pair
TEST(AddSecondInputTest, error_post_processor_pair_not_two_sequences) {
    auto model = make_minimal_model();
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {-1, 42};
    std::vector<int> pair_signature = {-1, 42, 42, 42}; // only one -1
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };
    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
    ASSERT_NE(pass_errors.str().find("Only 2 inputs are allowed for the paired input"), std::string::npos);
}

// 10. parse_and_assert_postprocessor: not exactly one sequence input in single
TEST(AddSecondInputTest, error_post_processor_single_not_one_sequence) {
    auto model = make_minimal_model();
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {42, 42}; // no -1
    std::vector<int> pair_signature = {42, 42, -1, 42};
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };
    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
    ASSERT_NE(pass_errors.str().find("Could not add second input. Input signature from rt_info does not "), std::string::npos);
}

// 11. get_new_inputs: post_processor["pair"] missing "type_ids"
TEST(AddSecondInputTest, error_post_processor_pair_missing_type_ids) {
    auto model = make_minimal_model();
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {-1, 42};
    std::vector<int> pair_signature = {-1, 42, -1, 42};
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}}} // no type_ids
    };
    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
    ASSERT_NE(pass_errors.str().find("does not contain 'type_ids' for paired input"), std::string::npos);
}

// 12. run_on_model: no target inputs for model parameter
TEST(AddSecondInputTest, error_no_target_inputs_for_parameter) {
    // Model with a CombineSegments node, but parameter not connected to anything
    auto parameter_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{2});
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
    auto combine_node = get_factory_create_func()("CombineSegments", {{c, c, c, c, c, c, c}}, {})[0];
    auto model = std::make_shared<ov::Model>(ov::OutputVector{combine_node}, ov::ParameterVector{parameter_1});
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    std::vector<int> input_signature = {-1, 42};
    std::vector<int> pair_signature = {-1, 42, -1, 42};
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };
    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();
    std::ostringstream pass_errors;
    bool ok = run_add_second_input_pass(model, pass_errors);
    ASSERT_FALSE(ok);
    ASSERT_NE(pass_errors.str().find("Could not add second input. Input signature from rt_info does not match to the CombineSegments node inputs"), std::string::npos);
}
