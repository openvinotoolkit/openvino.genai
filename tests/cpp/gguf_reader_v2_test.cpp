#include <gtest/gtest.h>
#include "gguf_utils/gguf_reader_v2.hpp"

#include "llama.h"
#include <vector>
#include <cstring>
#include <cmath>
#include "ggml-openvino-extra.h"

TEST(GGUFReaderV2Test, DefaultConstructionSucceeds) {
    EXPECT_NO_THROW({
        ov::genai::GGUFReaderV2 reader;
    });
}

TEST(GGUFReaderV2Test, MultipleInstancesDoNotCrash) {
    EXPECT_NO_THROW({
        ov::genai::GGUFReaderV2 reader1;
        ov::genai::GGUFReaderV2 reader2;
        ov::genai::GGUFReaderV2 reader3;
    });
}

TEST(GGUFReaderV2Test, InvalidPathThrows) {
    ov::genai::GGUFReaderV2 reader;
    EXPECT_THROW(
        reader.read("non_existent_fake_path.gguf"),
        std::runtime_error
    );
}

TEST(GGUFReaderV2Test, DestructorCleanupIsCorrect) {
    EXPECT_NO_THROW({
        {
            ov::genai::GGUFReaderV2 reader;
        }
    });
}

TEST(GGUFReaderV2Test, MathEquivalenceValidation) {
    const char* model_path = std::getenv("GGUF_TEST_MODEL");
    if (!model_path) {
        GTEST_SKIP() << "GGUF_TEST_MODEL not set, skipping real model test";
    }
    
    // Pass 1: Generate "Gold Standard" reference logits using native llama.cpp
    ggml_backend_ov_set_bypass(true);

    llama_model_params base_params = llama_model_default_params();
    base_params.n_gpu_layers = 0; // Force pure CPU execution for a stable gold standard
    llama_model* gold_model = llama_load_model_from_file(model_path, base_params);
    ASSERT_NE(gold_model, nullptr) << "Failed to load gold model";

    llama_context_params gold_ctx_params = llama_context_default_params();
    gold_ctx_params.n_ctx = 1;
    llama_context* gold_ctx = llama_init_from_model(gold_model, gold_ctx_params);
    
    // Create a batch with just the initial BOS token
    const llama_vocab* vocab = llama_model_get_vocab(gold_model);
    llama_token bos = llama_token_bos(vocab);
    if (bos == -1) bos = 0;
    llama_batch batch = llama_batch_get_one(&bos, 1);

    ASSERT_EQ(llama_decode(gold_ctx, batch), 0) << "Gold standard decode failed!";
    
    // Note: Assuming TinyLlama vocab size for testing. Dynamic vocab size retrieval can be added later.
    int vocab_size = 32000; 
    std::vector<float> reference_logits(vocab_size);
    float* raw_llama_logits = llama_get_logits(gold_ctx);
    std::memcpy(reference_logits.data(), raw_llama_logits, vocab_size * sizeof(float));

    llama_free(gold_ctx);
    llama_free_model(gold_model);
    ggml_backend_ov_set_bypass(false);

    // Pass 2: Capture Graph and Run OpenVINO Translation
    ov::genai::GGUFReaderV2 reader;
    std::shared_ptr<ov::Model> ov_model = reader.read(model_path);
    ASSERT_NE(ov_model, nullptr) << "FATAL: Reader did not produce an OpenVINO model!";

    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(ov_model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Map inputs for the initial BOS token evaluation
    for (const auto& input : ov_model->inputs()) {
        ov::Shape static_shape;
        for (const auto& dim : input.get_partial_shape()) {
            static_shape.push_back(dim.is_dynamic() ? 1 : dim.get_length());
        }
        
        ov::Tensor dummy_tensor(input.get_element_type(), static_shape);
        std::string name = input.get_names().empty() ? "UNKNOWN" : input.get_any_name();

        // Handle anonymous scalar inputs (n_past) used by llama.cpp for cache ScatterUpdates
        if (name == "leaf_8" || name == "leaf_10") {
            ov::element::Type expected_type = input.get_element_type();
            ov::Tensor scalar_tensor(expected_type, ov::Shape{1, 1, 1, 1});
            if (expected_type == ov::element::i64) {
                // Must be 0 for the first token since we are writing to the 0th index of the KV cache
                scalar_tensor.data<int64_t>()[0] = 0; 
            }
            infer_request.set_tensor(name, scalar_tensor);
            continue; 
        }

        if (name == "inp_tokens") {
            std::fill_n(dummy_tensor.data<int32_t>(), dummy_tensor.get_size(), static_cast<int32_t>(bos));
        } else if (name == "inp_pos" || name == "inp_out_ids") {
            std::fill_n(dummy_tensor.data<int32_t>(), dummy_tensor.get_size(), 0);
        } else if (name == "self_kq_mask") {
            // Apply causal mask: Unmask only the current token, block empty cache slots with -INF
            ov::Tensor mask_tensor(ov::element::f32, static_shape);
            float* mask_data = mask_tensor.data<float>();
            std::fill_n(mask_data, mask_tensor.get_size(), -1e9f); 
            mask_data[0] = 0.0f; 
            infer_request.set_tensor(name, mask_tensor);
            continue;
        } else {
            // Safely zero out intermediate KV caches and undefined variables
            std::memset(dummy_tensor.data(), 0, dummy_tensor.get_byte_size());
        }
        
        infer_request.set_tensor(input, dummy_tensor);
    }

    ASSERT_NO_THROW(infer_request.infer()) << "OpenVINO inference crashed!";

    // Pass 3: The Math Comparison
    ov::Tensor ov_logits_tensor;
    bool found_logits = false;
    for (const auto& output : compiled_model.outputs()) {
        auto shape = output.get_partial_shape();
        if (output.get_element_type() == ov::element::f32 && shape.rank().is_static() && shape.rbegin()->get_length() == vocab_size) {
            ov_logits_tensor = infer_request.get_tensor(output);
            found_logits = true;
            break;
        }
    }

    ASSERT_TRUE(found_logits) << "FATAL: Could not find logits output block in the compiled model.";
    const float* ov_logits = ov_logits_tensor.data<float>();
    
    int mismatch_count = 0;
    // Strict epsilon for floating-point precision parity validation (requires F16/F32 model)
    float epsilon = 0.015f; 

    for (int i = 0; i < vocab_size; i++) {
        float diff = std::abs(reference_logits[i] - ov_logits[i]);
        if (diff > epsilon) {
            if (mismatch_count < 5) { 
                std::cerr << "[MATH MISMATCH] Index " << i 
                          << " | Llama (Gold): " << reference_logits[i] 
                          << " | OV: " << ov_logits[i] << " | Diff: " << diff << "\n";
            }
            mismatch_count++;
        }
    }

    EXPECT_EQ(mismatch_count, 0) << "Translation failed! OpenVINO math does not match llama.cpp.";
}