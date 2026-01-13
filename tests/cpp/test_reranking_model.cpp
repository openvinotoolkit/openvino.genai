// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/reranking/reranking_model.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
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

static ov::Tensor create_model_output_1d(const std::vector<float>& scores) {
    return create_test_tensor(scores, {scores.size()});
}

static ov::Tensor create_model_output_2d(const std::vector<float>& scores, size_t num_labels = 1) {
    size_t batch_size = scores.size();
    std::vector<float> data;
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_labels; ++j) {
            data.push_back(scores[i]);
        }
    }
    return create_test_tensor(data, {batch_size, num_labels});
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static std::vector<float> softmax(const std::vector<float>& scores) {
    if (scores.empty())
        return {};

    float max_val = *std::max_element(scores.begin(), scores.end());
    std::vector<float> exp_scores;
    float sum = 0.0f;

    for (float s : scores) {
        float exp_s = std::exp(s - max_val);
        exp_scores.push_back(exp_s);
        sum += exp_s;
    }

    for (float& s : exp_scores) {
        s /= sum;
    }
    return exp_scores;
}

static std::vector<float> minmax_normalize(const std::vector<float>& scores) {
    if (scores.empty())
        return {};
    if (scores.size() == 1)
        return {1.0f};

    float min_val = *std::min_element(scores.begin(), scores.end());
    float max_val = *std::max_element(scores.begin(), scores.end());
    float range = max_val - min_val;

    if (range < 1e-8f) {
        return std::vector<float>(scores.size(), 0.5f);
    }

    std::vector<float> result;
    for (float s : scores) {
        result.push_back((s - min_val) / range);
    }
    return result;
}

static bool all_in_range_01(const std::vector<float>& values) {
    for (float v : values) {
        if (v < 0.0f || v > 1.0f) {
            return false;
        }
    }
    return true;
}

static bool has_invalid_values(const std::vector<float>& values) {
    for (float v : values) {
        if (std::isnan(v) || std::isinf(v)) {
            return true;
        }
    }
    return false;
}

static RerankingModelConfig create_default_config() {
    RerankingModelConfig config;
    config.variant = RerankVariant::AUTO;
    config.max_seq_length = 512;
    config.return_scores = true;
    config.normalize_scores = false;
    config.normalization_method = ScoreNormalization::SIGMOID;
    config.pooling = PoolingStrategy::NONE;
    config.separator = " [SEP] ";
    return config;
}

// =============================================================================
// Variant String Conversion Tests
// =============================================================================

TEST(RerankVariantTest, parse_variant_auto) {
    ASSERT_EQ(parse_rerank_variant("auto"), RerankVariant::AUTO);
    ASSERT_EQ(parse_rerank_variant("AUTO"), RerankVariant::AUTO);
}

TEST(RerankVariantTest, parse_variant_bge) {
    ASSERT_EQ(parse_rerank_variant("bge"), RerankVariant::BGE);
    ASSERT_EQ(parse_rerank_variant("BGE"), RerankVariant::BGE);
}

TEST(RerankVariantTest, parse_variant_bce) {
    ASSERT_EQ(parse_rerank_variant("bce"), RerankVariant::BCE);
    ASSERT_EQ(parse_rerank_variant("BCE"), RerankVariant::BCE);
}

TEST(RerankVariantTest, parse_variant_generic) {
    ASSERT_EQ(parse_rerank_variant("generic"), RerankVariant::GENERIC);
    ASSERT_EQ(parse_rerank_variant("GENERIC"), RerankVariant::GENERIC);
}

TEST(RerankVariantTest, parse_variant_invalid_throws) {
    ASSERT_THROW(parse_rerank_variant("invalid"), std::invalid_argument);
    ASSERT_THROW(parse_rerank_variant("unknown"), std::invalid_argument);
    ASSERT_THROW(parse_rerank_variant(""), std::invalid_argument);
}

TEST(RerankVariantTest, to_string_auto) {
    ASSERT_EQ(to_string(RerankVariant::AUTO), "auto");
}

TEST(RerankVariantTest, to_string_bge) {
    ASSERT_EQ(to_string(RerankVariant::BGE), "bge");
}

TEST(RerankVariantTest, to_string_bce) {
    ASSERT_EQ(to_string(RerankVariant::BCE), "bce");
}

TEST(RerankVariantTest, to_string_generic) {
    ASSERT_EQ(to_string(RerankVariant::GENERIC), "generic");
}

TEST(RerankVariantTest, roundtrip_auto) {
    std::string str = to_string(RerankVariant::AUTO);
    ASSERT_EQ(parse_rerank_variant(str), RerankVariant::AUTO);
}

TEST(RerankVariantTest, roundtrip_bge) {
    std::string str = to_string(RerankVariant::BGE);
    ASSERT_EQ(parse_rerank_variant(str), RerankVariant::BGE);
}

TEST(RerankVariantTest, roundtrip_bce) {
    std::string str = to_string(RerankVariant::BCE);
    ASSERT_EQ(parse_rerank_variant(str), RerankVariant::BCE);
}

TEST(RerankVariantTest, get_supported_variants_not_empty) {
    auto supported = get_supported_rerank_variants();
    ASSERT_GT(supported.size(), 0);
}

TEST(RerankVariantTest, get_supported_variants_contains_bge) {
    auto supported = get_supported_rerank_variants();
    bool found = std::find(supported.begin(), supported.end(), "bge") != supported.end();
    ASSERT_TRUE(found);
}

TEST(RerankVariantTest, get_supported_variants_contains_bce) {
    auto supported = get_supported_rerank_variants();
    bool found = std::find(supported.begin(), supported.end(), "bce") != supported.end();
    ASSERT_TRUE(found);
}

TEST(RerankVariantTest, get_supported_variants_contains_generic) {
    auto supported = get_supported_rerank_variants();
    bool found = std::find(supported.begin(), supported.end(), "generic") != supported.end();
    ASSERT_TRUE(found);
}

// =============================================================================
// RerankingModelConfig Tests
// =============================================================================

TEST(RerankingModelConfigTest, default_variant) {
    RerankingModelConfig config;
    ASSERT_EQ(config.variant, RerankVariant::AUTO);
}

TEST(RerankingModelConfigTest, default_max_seq_length) {
    RerankingModelConfig config;
    ASSERT_EQ(config.max_seq_length, 512);
}

TEST(RerankingModelConfigTest, default_return_scores) {
    RerankingModelConfig config;
    ASSERT_TRUE(config.return_scores);
}

TEST(RerankingModelConfigTest, default_normalize_scores) {
    RerankingModelConfig config;
    ASSERT_FALSE(config.normalize_scores);
}

TEST(RerankingModelConfigTest, default_normalization_method) {
    RerankingModelConfig config;
    ASSERT_EQ(config.normalization_method, ScoreNormalization::SIGMOID);
}

TEST(RerankingModelConfigTest, default_pooling) {
    RerankingModelConfig config;
    ASSERT_EQ(config.pooling, PoolingStrategy::NONE);
}

TEST(RerankingModelConfigTest, default_separator) {
    RerankingModelConfig config;
    ASSERT_EQ(config.separator, " [SEP] ");
}

TEST(RerankingModelConfigTest, custom_config) {
    RerankingModelConfig config;
    config.variant = RerankVariant::BGE;
    config.max_seq_length = 256;
    config.normalize_scores = true;
    config.normalization_method = ScoreNormalization::SOFTMAX;
    config.query_instruction = "Query: ";

    ASSERT_EQ(config.variant, RerankVariant::BGE);
    ASSERT_EQ(config.max_seq_length, 256);
    ASSERT_TRUE(config.normalize_scores);
    ASSERT_EQ(config.normalization_method, ScoreNormalization::SOFTMAX);
    ASSERT_EQ(config.query_instruction, "Query: ");
}

// =============================================================================
// RerankingResult Tests
// =============================================================================

TEST(RerankingResultTest, get_top_k_zero_returns_all) {
    RerankingResult result;
    result.scores = {0.9f, 0.7f, 0.8f, 0.6f, 0.5f};
    result.sorted_indices = {0, 2, 1, 3, 4};

    auto top_k = result.get_top_k(0);
    ASSERT_EQ(top_k.size(), 5);
    ASSERT_EQ(top_k, result.sorted_indices);
}

TEST(RerankingResultTest, get_top_k_3) {
    RerankingResult result;
    result.scores = {0.9f, 0.7f, 0.8f, 0.6f, 0.5f};
    result.sorted_indices = {0, 2, 1, 3, 4};

    auto top_k = result.get_top_k(3);
    ASSERT_EQ(top_k.size(), 3);
    ASSERT_EQ(top_k[0], 0);
    ASSERT_EQ(top_k[1], 2);
    ASSERT_EQ(top_k[2], 1);
}

TEST(RerankingResultTest, get_top_k_exceeds_size) {
    RerankingResult result;
    result.scores = {0.9f, 0.7f};
    result.sorted_indices = {0, 1};

    auto top_k = result.get_top_k(10);
    ASSERT_EQ(top_k.size(), 2);
}

TEST(RerankingResultTest, get_top_k_single_element) {
    RerankingResult result;
    result.scores = {0.5f};
    result.sorted_indices = {0};

    auto top_k = result.get_top_k(1);
    ASSERT_EQ(top_k.size(), 1);
    ASSERT_EQ(top_k[0], 0);
}

TEST(RerankingResultTest, get_top_k_empty) {
    RerankingResult result;

    auto top_k = result.get_top_k(5);
    ASSERT_TRUE(top_k.empty());
}

// =============================================================================
// Score Normalization Tests - Sigmoid
// =============================================================================

TEST(ScoreNormalizationTest, sigmoid_zero) {
    float result = sigmoid(0.0f);
    ASSERT_NEAR(result, 0.5f, 1e-5f);
}

TEST(ScoreNormalizationTest, sigmoid_positive) {
    float result = sigmoid(2.0f);
    ASSERT_GT(result, 0.5f);
    ASSERT_LT(result, 1.0f);
}

TEST(ScoreNormalizationTest, sigmoid_negative) {
    float result = sigmoid(-2.0f);
    ASSERT_LT(result, 0.5f);
    ASSERT_GT(result, 0.0f);
}

TEST(ScoreNormalizationTest, sigmoid_monotonic) {
    std::vector<float> scores = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> normalized;
    for (float s : scores) {
        normalized.push_back(sigmoid(s));
    }

    for (size_t i = 1; i < normalized.size(); ++i) {
        ASSERT_GT(normalized[i], normalized[i - 1]);
    }
}

TEST(ScoreNormalizationTest, sigmoid_all_in_range) {
    std::vector<float> scores = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    std::vector<float> normalized;
    for (float s : scores) {
        normalized.push_back(sigmoid(s));
    }
    ASSERT_TRUE(all_in_range_01(normalized));
}

// =============================================================================
// Score Normalization Tests - Softmax
// =============================================================================

TEST(ScoreNormalizationTest, softmax_sums_to_one) {
    std::vector<float> scores = {1.0f, 2.0f, 3.0f};
    auto normalized = softmax(scores);

    float sum = std::accumulate(normalized.begin(), normalized.end(), 0.0f);
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(ScoreNormalizationTest, softmax_all_in_range) {
    std::vector<float> scores = {1.0f, 2.0f, 3.0f};
    auto normalized = softmax(scores);
    ASSERT_TRUE(all_in_range_01(normalized));
}

TEST(ScoreNormalizationTest, softmax_monotonic) {
    std::vector<float> scores = {1.0f, 2.0f, 3.0f};
    auto normalized = softmax(scores);

    ASSERT_LT(normalized[0], normalized[1]);
    ASSERT_LT(normalized[1], normalized[2]);
}

TEST(ScoreNormalizationTest, softmax_negative_scores) {
    std::vector<float> scores = {-3.0f, -2.0f, -1.0f};
    auto normalized = softmax(scores);

    float sum = std::accumulate(normalized.begin(), normalized.end(), 0.0f);
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    ASSERT_TRUE(all_in_range_01(normalized));
}

TEST(ScoreNormalizationTest, softmax_large_values_stable) {
    std::vector<float> scores = {100.0f, 101.0f, 102.0f};
    auto normalized = softmax(scores);

    ASSERT_FALSE(has_invalid_values(normalized));
    float sum = std::accumulate(normalized.begin(), normalized.end(), 0.0f);
    ASSERT_NEAR(sum, 1.0f, 1e-4f);
}

TEST(ScoreNormalizationTest, softmax_empty) {
    std::vector<float> scores = {};
    auto normalized = softmax(scores);
    ASSERT_TRUE(normalized.empty());
}

// =============================================================================
// Score Normalization Tests - MinMax
// =============================================================================

TEST(ScoreNormalizationTest, minmax_range) {
    std::vector<float> scores = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    auto normalized = minmax_normalize(scores);

    ASSERT_NEAR(normalized[0], 0.0f, 1e-5f);
    ASSERT_NEAR(normalized[4], 1.0f, 1e-5f);
}

TEST(ScoreNormalizationTest, minmax_all_in_range) {
    std::vector<float> scores = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    auto normalized = minmax_normalize(scores);
    ASSERT_TRUE(all_in_range_01(normalized));
}

TEST(ScoreNormalizationTest, minmax_equal_scores) {
    std::vector<float> scores = {5.0f, 5.0f, 5.0f};
    auto normalized = minmax_normalize(scores);

    for (float v : normalized) {
        ASSERT_NEAR(v, 0.5f, 1e-5f);
    }
}

TEST(ScoreNormalizationTest, minmax_single_score) {
    std::vector<float> scores = {3.14f};
    auto normalized = minmax_normalize(scores);

    ASSERT_EQ(normalized.size(), 1);
    ASSERT_NEAR(normalized[0], 1.0f, 1e-5f);
}

TEST(ScoreNormalizationTest, minmax_empty) {
    std::vector<float> scores = {};
    auto normalized = minmax_normalize(scores);
    ASSERT_TRUE(normalized.empty());
}

// =============================================================================
// Adapter Factory Tests
// =============================================================================

TEST(RerankingAdapterFactoryTest, create_bge_adapter) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);

    ASSERT_NE(adapter, nullptr);
    ASSERT_EQ(adapter->get_variant(), RerankVariant::BGE);
}

TEST(RerankingAdapterFactoryTest, create_bce_adapter) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BCE);

    ASSERT_NE(adapter, nullptr);
    ASSERT_EQ(adapter->get_variant(), RerankVariant::BCE);
}

TEST(RerankingAdapterFactoryTest, create_generic_adapter) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);

    ASSERT_NE(adapter, nullptr);
    ASSERT_EQ(adapter->get_variant(), RerankVariant::GENERIC);
}

TEST(RerankingAdapterFactoryTest, create_auto_resolves) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::AUTO);

    ASSERT_NE(adapter, nullptr);
    // AUTO should resolve to GENERIC as fallback
    ASSERT_EQ(adapter->get_variant(), RerankVariant::GENERIC);
}

TEST(RerankingAdapterFactoryTest, bge_adapter_name) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    ASSERT_FALSE(adapter->get_variant_name().empty());
}

TEST(RerankingAdapterFactoryTest, bce_adapter_name) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BCE);
    ASSERT_FALSE(adapter->get_variant_name().empty());
}

TEST(RerankingAdapterFactoryTest, generic_adapter_name) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    ASSERT_FALSE(adapter->get_variant_name().empty());
}

TEST(RerankingAdapterFactoryTest, bge_default_max_seq_length) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    ASSERT_EQ(adapter->get_default_max_seq_length(), 512);
}

TEST(RerankingAdapterFactoryTest, bce_default_max_seq_length) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BCE);
    ASSERT_EQ(adapter->get_default_max_seq_length(), 512);
}

TEST(RerankingAdapterFactoryTest, generic_default_max_seq_length) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    ASSERT_EQ(adapter->get_default_max_seq_length(), 512);
}

TEST(RerankingAdapterFactoryTest, bge_default_normalize) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    ASSERT_TRUE(adapter->default_normalize_scores());
}

TEST(RerankingAdapterFactoryTest, generic_default_normalize) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    ASSERT_FALSE(adapter->default_normalize_scores());
}

// =============================================================================
// Adapter Input Formatting Tests
// =============================================================================

TEST(RerankingAdapterInputTest, bge_format_not_empty) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    auto config = create_default_config();

    std::string formatted = adapter->format_input("query", "document", config);
    ASSERT_FALSE(formatted.empty());
}

TEST(RerankingAdapterInputTest, bge_format_contains_query) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    auto config = create_default_config();

    std::string formatted = adapter->format_input("test query", "test document", config);
    ASSERT_NE(formatted.find("test query"), std::string::npos);
}

TEST(RerankingAdapterInputTest, bge_format_contains_document) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    auto config = create_default_config();

    std::string formatted = adapter->format_input("test query", "test document", config);
    ASSERT_NE(formatted.find("test document"), std::string::npos);
}

TEST(RerankingAdapterInputTest, generic_format_contains_separator) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    auto config = create_default_config();
    config.separator = " [SEP] ";

    std::string formatted = adapter->format_input("query", "document", config);
    ASSERT_NE(formatted.find("[SEP]"), std::string::npos);
}

TEST(RerankingAdapterInputTest, custom_separator) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    auto config = create_default_config();
    config.separator = " </s> ";

    std::string formatted = adapter->format_input("Q", "D", config);
    ASSERT_NE(formatted.find("</s>"), std::string::npos);
}

// =============================================================================
// Adapter Output Validation Tests
// =============================================================================

TEST(RerankingAdapterOutputTest, validate_correct_shape) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    ov::Shape valid_shape = {4};
    ASSERT_NO_THROW(adapter->validate_output_shape(valid_shape, 4));
}

TEST(RerankingAdapterOutputTest, validate_batch_mismatch_throws) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    ov::Shape shape = {3};
    ASSERT_THROW(adapter->validate_output_shape(shape, 5), std::runtime_error);
}

TEST(RerankingAdapterOutputTest, validate_empty_shape_throws) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    ov::Shape empty_shape = {};
    ASSERT_THROW(adapter->validate_output_shape(empty_shape, 1), std::runtime_error);
}

TEST(RerankingAdapterOutputTest, bge_expected_output_description) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    ASSERT_FALSE(adapter->get_expected_output_description().empty());
}

TEST(RerankingAdapterOutputTest, generic_expected_output_description) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    ASSERT_FALSE(adapter->get_expected_output_description().empty());
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(RerankingEdgeCaseTest, format_empty_query) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    auto config = create_default_config();

    std::string formatted = adapter->format_input("", "document", config);
    ASSERT_FALSE(formatted.empty());
}

TEST(RerankingEdgeCaseTest, format_empty_document) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    auto config = create_default_config();

    std::string formatted = adapter->format_input("query", "", config);
    ASSERT_FALSE(formatted.empty());
}

TEST(RerankingEdgeCaseTest, format_long_texts) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::BGE);
    auto config = create_default_config();

    std::string long_query(5000, 'q');
    std::string long_document(5000, 'd');

    std::string formatted = adapter->format_input(long_query, long_document, config);
    ASSERT_FALSE(formatted.empty());
    ASSERT_GT(formatted.length(), 0);
}

TEST(RerankingEdgeCaseTest, format_unicode_texts) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    auto config = create_default_config();

    std::string unicode_query = "什么是机器学习？";
    std::string unicode_document = "机器学习是人工智能的一个子集。";

    std::string formatted = adapter->format_input(unicode_query, unicode_document, config);
    ASSERT_NE(formatted.find(unicode_query), std::string::npos);
    ASSERT_NE(formatted.find(unicode_document), std::string::npos);
}

TEST(RerankingEdgeCaseTest, format_special_characters) {
    auto adapter = RerankingAdapterFactory::create(RerankVariant::GENERIC);
    auto config = create_default_config();

    std::string query = "query with <special> & \"chars\"";
    std::string document = "doc with\ttabs\nand\rnewlines";

    std::string formatted = adapter->format_input(query, document, config);
    ASSERT_FALSE(formatted.empty());
}

TEST(RerankingEdgeCaseTest, result_large_document_count) {
    size_t num_docs = 1000;

    RerankingResult result;
    result.scores.resize(num_docs);
    result.sorted_indices.resize(num_docs);

    for (size_t i = 0; i < num_docs; ++i) {
        result.scores[i] = static_cast<float>(num_docs - i) / num_docs;
        result.sorted_indices[i] = i;
    }

    auto top_10 = result.get_top_k(10);
    ASSERT_EQ(top_10.size(), 10);

    auto top_100 = result.get_top_k(100);
    ASSERT_EQ(top_100.size(), 100);
}

// =============================================================================
// Logging Callback Tests
// =============================================================================

TEST(RerankingLogCallbackTest, callback_invoked) {
    std::vector<std::string> logged_messages;

    RerankingLogCallback callback = [&logged_messages](const std::string& message) {
        logged_messages.push_back(message);
    };

    callback("Test message 1");
    callback("Test message 2");

    ASSERT_EQ(logged_messages.size(), 2);
    ASSERT_EQ(logged_messages[0], "Test message 1");
    ASSERT_EQ(logged_messages[1], "Test message 2");
}

TEST(RerankingLogCallbackTest, null_callback_is_false) {
    RerankingLogCallback callback = nullptr;
    ASSERT_FALSE(callback);
}

TEST(RerankingLogCallbackTest, valid_callback_is_true) {
    RerankingLogCallback callback = [](const std::string&) {};
    ASSERT_TRUE(callback);
}
