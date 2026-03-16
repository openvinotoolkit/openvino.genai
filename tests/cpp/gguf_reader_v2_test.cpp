#include <gtest/gtest.h>
#include "gguf_utils/gguf_reader_v2.hpp"

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
    
    ov::genai::GGUFReaderV2 reader;
    
    // 1. Extract the full monolithic graph
    std::shared_ptr<ov::Model> ov_model = reader.read(model_path);
    ASSERT_NE(ov_model, nullptr) << "FATAL: Reader did not produce an OpenVINO model!";

    std::cout << "\n[GGUFReaderV2] Successfully captured " << ov_model->get_ops().size() << " OpenVINO operations.\n";

    // 2. Compile the model for the CPU
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(ov_model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 3. The Universal Auto-Feeder
    // We dynamically find every input the model needs and feed it empty zeros
    for (const auto& input : ov_model->inputs()) {
        ov::PartialShape pshape = input.get_partial_shape();
        ov::Shape static_shape;
        
        // Convert any dynamic dimensions (like '?') to 1
        for (const auto& dim : pshape) {
            if (dim.is_dynamic()) {
                static_shape.push_back(1);
            } else {
                static_shape.push_back(dim.get_length());
            }
        }
        
        // Create a dummy tensor and fill it with 0
        ov::Tensor dummy_tensor(input.get_element_type(), static_shape);
        std::memset(dummy_tensor.data(), 0, dummy_tensor.get_byte_size());
        
        infer_request.set_tensor(input, dummy_tensor);
    }

    // 4. Run the OpenVINO Math
    std::cout << "[GGUFReaderV2] Running OpenVINO inference...\n";
    ASSERT_NO_THROW(infer_request.infer()) << "OpenVINO inference crashed!";

    // 5. Compare the Outputs (Logits)
    ov::Tensor ov_logits_tensor;
    bool found_logits = false;

    // Search through all outputs to find the logits (f32 type, last dimension = 32000)
    for (const auto& output : compiled_model.outputs()) {
        auto shape = output.get_partial_shape();
        auto type = output.get_element_type();
        
        if (type == ov::element::f32 && shape.rank().is_static() && shape.rbegin()->get_length() == 32000) {
            ov_logits_tensor = infer_request.get_tensor(output);
            found_logits = true;
            std::cout << "[GGUFReaderV2] Found Logits Tensor on port: " 
                      << (output.get_names().empty() ? "UNKNOWN" : output.get_any_name()) << "\n";
            break;
        }
    }

    ASSERT_TRUE(found_logits) << "FATAL: Could not find an f32 output tensor with a vocab size of 32000!";
    const float* ov_logits = ov_logits_tensor.data<float>();
    float* llama_logits = reader.get_native_logits();
    
    ASSERT_NE(llama_logits, nullptr) << "FATAL: Failed to get llama logits";
    
    int vocab_size = 32000; // TinyLlama vocab size
    int mismatch_count = 0;
    
    // We use a small epsilon because floating-point math can drift slightly 
    // between OpenVINO hardware optimizations and llama.cpp C-code.
    float epsilon = 1e-3f; 

    for (int i = 0; i < vocab_size; i++) {
        float diff = std::abs(llama_logits[i] - ov_logits[i]);
        if (diff > epsilon) {
            if (mismatch_count < 5) { 
                std::cerr << "[MATH MISMATCH] Index " << i 
                          << " | Llama: " << llama_logits[i] 
                          << " | OV: " << ov_logits[i] << " | Diff: " << diff << "\n";
            }
            mismatch_count++;
        }
    }

    EXPECT_EQ(mismatch_count, 0) << "Translation failed! The OpenVINO math does not match llama.cpp.";
    if (mismatch_count == 0) {
        std::cout << "==================================================\n";
        std::cout << "🏆 SUCCESS: OpenVINO math is 100% equivalent to llama.cpp! 🏆\n";
        std::cout << "==================================================\n";
    }
}