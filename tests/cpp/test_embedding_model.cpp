// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/embedding/embedding_model.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <vector>


using namespace ov::genai;

// =============================================================================
// Helper Functions
// =============================================================================

static ov::Tensor create_test_tensor(const std::vector<float>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::memcpy(tensor.data<float>(), data.data(), data.size() * sizeof(float));
    return tensor;
}

static ov::Tensor create_hidden_states(size_t batch_size, size_t seq_length, size_t hidden_size, float base_value = 0.1f) {
    std::vector<float> data(batch_size * seq_length * hidden_size);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_length; ++s) {
            for (size_t h = 0; h < hidden_size; ++h) {
                data[b * seq_length * hidden_size + s * hidden_size + h] =
                    base_value + static_cast<float>(b) * 0.01f + 
                    static_cast<float>(s) * 0.001f +
                    static_cast<float>(h) * 0.0001f;
            }
        }
    }
    return create_test_tensor(data, {batch_size, seq_length, hidden_size});
}

static ov::Tensor create_attention_mask(size_t batch_size, size_t seq_length, size_t valid_length = 0) {
    if (valid_length == 0)
        valid_length = seq_length;

    std::vector<int64_t> mask_data(batch_size * seq_length);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_length; ++s) {
            mask_data[b * seq_length + s] = (s < valid_length) ? 1 : 0;
        }
    }

    ov::Tensor mask(ov::element::i64, {batch_size, seq_length});
    std::memcpy(mask.data<int64_t>(), mask_data.data(), mask_data.size() * sizeof(int64_t));
    return mask;
}

static float compute_l2_norm(const float* data, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum);
}

static bool has_invalid_values(const ov::Tensor& tensor) {
    const float* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return true;
        }
    }
    return false;
}

// =============================================================================
// Architecture String Conversion Tests
// =============================================================================

TEST(EmbeddingArchitectureTest, architecture_to_string_bge) {
    ASSERT_EQ(architecture_to_string(EmbeddingArchitecture::BGE), "bge");
}

TEST(EmbeddingArchitectureTest, architecture_to_string_bce) {
    ASSERT_EQ(architecture_to_string(EmbeddingArchitecture::BCE), "bce");
}

TEST(EmbeddingArchitectureTest, architecture_to_string_gte) {
    ASSERT_EQ(architecture_to_string(EmbeddingArchitecture::GTE), "gte");
}

TEST(EmbeddingArchitectureTest, architecture_to_string_e5) {
    ASSERT_EQ(architecture_to_string(EmbeddingArchitecture::E5), "e5");
}

TEST(EmbeddingArchitectureTest, architecture_to_string_instructor) {
    ASSERT_EQ(architecture_to_string(EmbeddingArchitecture::INSTRUCTOR), "instructor");
}

TEST(EmbeddingArchitectureTest, architecture_to_string_jina) {
    ASSERT_EQ(architecture_to_string(EmbeddingArchitecture::JINA), "jina");
}

TEST(EmbeddingArchitectureTest, architecture_to_string_nomic) {
    ASSERT_EQ(architecture_to_string(EmbeddingArchitecture::NOMIC), "nomic");
}

TEST(EmbeddingArchitectureTest, architecture_to_string_unknown) {
    ASSERT_EQ(architecture_to_string(EmbeddingArchitecture::UNKNOWN), "unknown");
}

TEST(EmbeddingArchitectureTest, architecture_from_string_bge) {
    ASSERT_EQ(architecture_from_string("bge"), EmbeddingArchitecture::BGE);
}

TEST(EmbeddingArchitectureTest, architecture_from_string_bce) {
    ASSERT_EQ(architecture_from_string("bce"), EmbeddingArchitecture::BCE);
}

TEST(EmbeddingArchitectureTest, architecture_from_string_gte) {
    ASSERT_EQ(architecture_from_string("gte"), EmbeddingArchitecture::GTE);
}

TEST(EmbeddingArchitectureTest, architecture_from_string_e5) {
    ASSERT_EQ(architecture_from_string("e5"), EmbeddingArchitecture::E5);
}

TEST(EmbeddingArchitectureTest, architecture_from_string_case_insensitive) {
    ASSERT_EQ(architecture_from_string("BGE"), EmbeddingArchitecture::BGE);
    ASSERT_EQ(architecture_from_string("Bge"), EmbeddingArchitecture::BGE);
    ASSERT_EQ(architecture_from_string("BCE"), EmbeddingArchitecture::BCE);
}

TEST(EmbeddingArchitectureTest, architecture_from_string_invalid) {
    ASSERT_EQ(architecture_from_string("invalid"), EmbeddingArchitecture::UNKNOWN);
    ASSERT_EQ(architecture_from_string(""), EmbeddingArchitecture::UNKNOWN);
    ASSERT_EQ(architecture_from_string("not_a_model"), EmbeddingArchitecture::UNKNOWN);
}

TEST(EmbeddingArchitectureTest, architecture_roundtrip_bge) {
    std::string str = architecture_to_string(EmbeddingArchitecture::BGE);
    ASSERT_EQ(architecture_from_string(str), EmbeddingArchitecture::BGE);
}

TEST(EmbeddingArchitectureTest, architecture_roundtrip_e5) {
    std::string str = architecture_to_string(EmbeddingArchitecture::E5);
    ASSERT_EQ(architecture_from_string(str), EmbeddingArchitecture::E5);
}

// =============================================================================
// Architecture Registry Tests
// =============================================================================

TEST(EmbeddingArchitectureRegistryTest, singleton_returns_same_instance) {
    auto& registry1 = EmbeddingArchitectureRegistry::instance();
    auto& registry2 = EmbeddingArchitectureRegistry::instance();
    ASSERT_EQ(&registry1, &registry2);
}

TEST(EmbeddingArchitectureRegistryTest, get_config_bge) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    auto config = registry.get_config(EmbeddingArchitecture::BGE);

    ASSERT_EQ(config.architecture, EmbeddingArchitecture::BGE);
    ASSERT_FALSE(config.lora_tensor_prefix.empty());
    ASSERT_EQ(config.default_pooling_mode, "cls");
    ASSERT_TRUE(config.normalize_embeddings);
}

TEST(EmbeddingArchitectureRegistryTest, get_config_bce) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    auto config = registry.get_config(EmbeddingArchitecture::BCE);

    ASSERT_EQ(config.architecture, EmbeddingArchitecture::BCE);
    ASSERT_FALSE(config.lora_tensor_prefix.empty());
}

TEST(EmbeddingArchitectureRegistryTest, get_config_gte) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    auto config = registry.get_config(EmbeddingArchitecture::GTE);

    ASSERT_EQ(config.architecture, EmbeddingArchitecture::GTE);
    ASSERT_FALSE(config.lora_tensor_prefix.empty());
}

TEST(EmbeddingArchitectureRegistryTest, get_config_e5) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    auto config = registry.get_config(EmbeddingArchitecture::E5);

    ASSERT_EQ(config.architecture, EmbeddingArchitecture::E5);
    ASSERT_EQ(config.default_pooling_mode, "mean");
    ASSERT_TRUE(config.query_instruction.has_value());
    ASSERT_TRUE(config.document_instruction.has_value());
}

TEST(EmbeddingArchitectureRegistryTest, get_config_unknown_returns_default) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    auto config = registry.get_config(EmbeddingArchitecture::UNKNOWN);

    ASSERT_FALSE(config.lora_tensor_prefix.empty());
    ASSERT_FALSE(config.default_pooling_mode.empty());
}

TEST(EmbeddingArchitectureRegistryTest, get_lora_prefix_bge) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    std::string prefix = registry.get_lora_prefix(EmbeddingArchitecture::BGE);
    ASSERT_FALSE(prefix.empty());
}

TEST(EmbeddingArchitectureRegistryTest, get_lora_prefix_e5) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    std::string prefix = registry.get_lora_prefix(EmbeddingArchitecture::E5);
    ASSERT_FALSE(prefix.empty());
}

TEST(EmbeddingArchitectureRegistryTest, get_registered_architectures_not_empty) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    auto archs = registry.get_registered_architectures();
    ASSERT_GT(archs.size(), 0);
}

TEST(EmbeddingArchitectureRegistryTest, get_registered_architectures_contains_bge) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    auto archs = registry.get_registered_architectures();

    bool found_bge = std::find(archs.begin(), archs.end(), EmbeddingArchitecture::BGE) != archs.end();
    ASSERT_TRUE(found_bge);
}

TEST(EmbeddingArchitectureRegistryTest, register_custom_architecture) {
    auto& registry = EmbeddingArchitectureRegistry::instance();

    EmbeddingArchitectureConfig custom_config;
    custom_config.architecture = EmbeddingArchitecture::CUSTOM;
    custom_config.lora_tensor_prefix = "test_custom_prefix";
    custom_config.default_pooling_mode = "max";
    custom_config.normalize_embeddings = false;

    registry.register_architecture(EmbeddingArchitecture::CUSTOM, custom_config);

    auto retrieved = registry.get_config(EmbeddingArchitecture::CUSTOM);
    ASSERT_EQ(retrieved.lora_tensor_prefix, "test_custom_prefix");
    ASSERT_EQ(retrieved.default_pooling_mode, "max");
    ASSERT_FALSE(retrieved.normalize_embeddings);
}

// =============================================================================
// EmbeddingModelConfig Tests
// =============================================================================

TEST(EmbeddingModelConfigTest, default_hidden_size) {
    EmbeddingModelConfig config;
    ASSERT_EQ(config.hidden_size, 768);
}

TEST(EmbeddingModelConfigTest, default_max_seq_length) {
    EmbeddingModelConfig config;
    ASSERT_EQ(config.max_seq_length, 512);
}

TEST(EmbeddingModelConfigTest, default_pooling_mode) {
    EmbeddingModelConfig config;
    ASSERT_EQ(config.pooling_mode, "mean");
}

TEST(EmbeddingModelConfigTest, default_normalize_embeddings) {
    EmbeddingModelConfig config;
    ASSERT_TRUE(config.normalize_embeddings);
}

TEST(EmbeddingModelConfigTest, default_architecture) {
    EmbeddingModelConfig config;
    ASSERT_EQ(config.architecture, EmbeddingArchitecture::UNKNOWN);
}

TEST(EmbeddingModelConfigTest, apply_architecture_defaults_bge) {
    EmbeddingModelConfig config;
    config.apply_architecture_defaults(EmbeddingArchitecture::BGE);

    ASSERT_EQ(config.architecture, EmbeddingArchitecture::BGE);
    ASSERT_FALSE(config.lora_tensor_prefix.empty());
    ASSERT_TRUE(config.query_instruction.has_value());
}

TEST(EmbeddingModelConfigTest, apply_architecture_defaults_e5) {
    EmbeddingModelConfig config;
    config.apply_architecture_defaults(EmbeddingArchitecture::E5);

    ASSERT_EQ(config.architecture, EmbeddingArchitecture::E5);
    ASSERT_EQ(config.pooling_mode, "mean");
    ASSERT_TRUE(config.query_instruction.has_value());
    ASSERT_EQ(*config.query_instruction, "query: ");
    ASSERT_TRUE(config.document_instruction.has_value());
    ASSERT_EQ(*config.document_instruction, "passage: ");
}

// =============================================================================
// EmbeddingModelConfigBuilder Tests
// =============================================================================

TEST(EmbeddingModelConfigBuilderTest, set_architecture) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_architecture(EmbeddingArchitecture::BGE)
                      .build();
    ASSERT_EQ(config.architecture, EmbeddingArchitecture::BGE);
}

TEST(EmbeddingModelConfigBuilderTest, set_hidden_size) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_hidden_size(384)
                      .build();
    ASSERT_EQ(config.hidden_size, 384);
}

TEST(EmbeddingModelConfigBuilderTest, set_max_seq_length) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_max_seq_length(256)
                      .build();
    ASSERT_EQ(config.max_seq_length, 256);
}

TEST(EmbeddingModelConfigBuilderTest, set_pooling_mode) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_pooling_mode("cls")
                      .build();
    ASSERT_EQ(config.pooling_mode, "cls");
}

TEST(EmbeddingModelConfigBuilderTest, set_normalize_false) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_normalize(false)
                      .build();
    ASSERT_FALSE(config.normalize_embeddings);
}

TEST(EmbeddingModelConfigBuilderTest, set_lora_prefix) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_lora_prefix("bert")
                      .build();
    ASSERT_EQ(config.lora_tensor_prefix, "bert");
}

TEST(EmbeddingModelConfigBuilderTest, add_lora_prefix_fallback) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_lora_prefix("primary")
                      .add_lora_prefix_fallback("fallback1")
                      .add_lora_prefix_fallback("fallback2")
                      .build();

    ASSERT_EQ(config.lora_tensor_prefix, "primary");
    ASSERT_EQ(config.lora_prefix_fallbacks.size(), 2);
    ASSERT_EQ(config.lora_prefix_fallbacks[0], "fallback1");
    ASSERT_EQ(config.lora_prefix_fallbacks[1], "fallback2");
}

TEST(EmbeddingModelConfigBuilderTest, set_query_instruction) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_query_instruction("Search: ")
                      .build();

    ASSERT_TRUE(config.query_instruction.has_value());
    ASSERT_EQ(*config.query_instruction, "Search: ");
}

TEST(EmbeddingModelConfigBuilderTest, set_document_instruction) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_document_instruction("Document: ")
                      .build();

    ASSERT_TRUE(config.document_instruction.has_value());
    ASSERT_EQ(*config.document_instruction, "Document: ");
}

TEST(EmbeddingModelConfigBuilderTest, chain_multiple_settings) {
    auto config = EmbeddingModelConfigBuilder()
                      .set_architecture(EmbeddingArchitecture::BGE)
                      .set_hidden_size(384)
                      .set_max_seq_length(256)
                      .set_pooling_mode("cls")
                      .set_normalize(true)
                      .build();

    ASSERT_EQ(config.architecture, EmbeddingArchitecture::BGE);
    ASSERT_EQ(config.hidden_size, 384);
    ASSERT_EQ(config.max_seq_length, 256);
    ASSERT_EQ(config.pooling_mode, "cls");
    ASSERT_TRUE(config.normalize_embeddings);
}

// =============================================================================
// Tensor Helper Tests
// =============================================================================

TEST(EmbeddingTensorTest, create_hidden_states_shape_batch_1) {
    auto tensor = create_hidden_states(1, 10, 384);
    auto shape = tensor.get_shape();

    ASSERT_EQ(shape.size(), 3);
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 10);
    ASSERT_EQ(shape[2], 384);
}

TEST(EmbeddingTensorTest, create_hidden_states_shape_batch_4) {
    auto tensor = create_hidden_states(4, 128, 768);
    auto shape = tensor.get_shape();

    ASSERT_EQ(shape.size(), 3);
    ASSERT_EQ(shape[0], 4);
    ASSERT_EQ(shape[1], 128);
    ASSERT_EQ(shape[2], 768);
}

TEST(EmbeddingTensorTest, create_hidden_states_no_invalid_values) {
    auto tensor = create_hidden_states(4, 128, 768);
    ASSERT_FALSE(has_invalid_values(tensor));
}

TEST(EmbeddingTensorTest, create_attention_mask_shape) {
    auto mask = create_attention_mask(2, 10);
    auto shape = mask.get_shape();

    ASSERT_EQ(shape.size(), 2);
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 10);
}

TEST(EmbeddingTensorTest, create_attention_mask_all_valid) {
    auto mask = create_attention_mask(1, 10, 10);
    const int64_t* data = mask.data<int64_t>();

    for (size_t i = 0; i < 10; ++i) {
        ASSERT_EQ(data[i], 1);
    }
}

TEST(EmbeddingTensorTest, create_attention_mask_partial_valid) {
    auto mask = create_attention_mask(1, 10, 7);
    const int64_t* data = mask.data<int64_t>();

    // First 7 should be 1
    for (size_t i = 0; i < 7; ++i) {
        ASSERT_EQ(data[i], 1);
    }
    // Remaining 3 should be 0
    for (size_t i = 7; i < 10; ++i) {
        ASSERT_EQ(data[i], 0);
    }
}

// =============================================================================
// L2 Normalization Tests
// =============================================================================

TEST(EmbeddingNormalizationTest, l2_norm_3_4_triangle) {
    std::vector<float> data = {3.0f, 4.0f};  // 3-4-5 triangle
    float norm = compute_l2_norm(data.data(), data.size());
    ASSERT_NEAR(norm, 5.0f, 1e-5f);
}

TEST(EmbeddingNormalizationTest, l2_norm_unit_vector) {
    std::vector<float> data = {1.0f, 0.0f, 0.0f};
    float norm = compute_l2_norm(data.data(), data.size());
    ASSERT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(EmbeddingNormalizationTest, l2_norm_zero_vector) {
    std::vector<float> data = {0.0f, 0.0f, 0.0f};
    float norm = compute_l2_norm(data.data(), data.size());
    ASSERT_NEAR(norm, 0.0f, 1e-5f);
}

TEST(EmbeddingNormalizationTest, l2_norm_negative_values) {
    std::vector<float> data = {-3.0f, -4.0f};
    float norm = compute_l2_norm(data.data(), data.size());
    ASSERT_NEAR(norm, 5.0f, 1e-5f);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(EmbeddingEdgeCaseTest, single_batch_single_token) {
    auto tensor = create_hidden_states(1, 1, 384);
    ASSERT_EQ(tensor.get_shape()[0], 1);
    ASSERT_EQ(tensor.get_shape()[1], 1);
    ASSERT_EQ(tensor.get_shape()[2], 384);
    ASSERT_FALSE(has_invalid_values(tensor));
}

TEST(EmbeddingEdgeCaseTest, large_batch_size) {
    auto tensor = create_hidden_states(64, 128, 768);
    ASSERT_EQ(tensor.get_shape()[0], 64);
    ASSERT_FALSE(has_invalid_values(tensor));
}

TEST(EmbeddingEdgeCaseTest, large_hidden_size) {
    auto tensor = create_hidden_states(2, 10, 4096);
    ASSERT_EQ(tensor.get_shape()[2], 4096);
    ASSERT_FALSE(has_invalid_values(tensor));
}

TEST(EmbeddingEdgeCaseTest, large_seq_length) {
    auto tensor = create_hidden_states(1, 2048, 384);
    ASSERT_EQ(tensor.get_shape()[1], 2048);
    ASSERT_FALSE(has_invalid_values(tensor));
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(EmbeddingRegistryThreadSafetyTest, concurrent_get_config) {
    auto& registry = EmbeddingArchitectureRegistry::instance();

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&registry, &success_count, i]() {
            try {
                auto arch = static_cast<EmbeddingArchitecture>(i % 5);
                auto config = registry.get_config(arch);
                if (!config.lora_tensor_prefix.empty()) {
                    success_count++;
                }
            } catch (...) {
                // Exception indicates thread safety issue
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    ASSERT_EQ(success_count.load(), 10);
}

TEST(EmbeddingRegistryThreadSafetyTest, concurrent_register_and_get) {
    auto& registry = EmbeddingArchitectureRegistry::instance();

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    // Mix of readers and writers
    for (int i = 0; i < 5; ++i) {
        // Reader threads
        threads.emplace_back([&registry, &success_count]() {
            try {
                auto config = registry.get_config(EmbeddingArchitecture::BGE);
                if (config.architecture == EmbeddingArchitecture::BGE) {
                    success_count++;
                }
            } catch (...) {}
        });

        // Writer threads
        threads.emplace_back([&registry, &success_count, i]() {
            try {
                EmbeddingArchitectureConfig cfg;
                cfg.architecture = EmbeddingArchitecture::CUSTOM;
                cfg.lora_tensor_prefix = "thread_" + std::to_string(i);
                registry.register_architecture(EmbeddingArchitecture::CUSTOM, cfg);
                success_count++;
            } catch (...) {}
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    ASSERT_EQ(success_count.load(), 10);
}
