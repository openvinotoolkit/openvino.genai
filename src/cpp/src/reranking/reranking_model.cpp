// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/reranking/reranking_model.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <limits>
#include <sstream>
#include <mutex>

#include "openvino/openvino.hpp"
#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"

// Debug output macros - only active in debug builds
#ifndef NDEBUG
    #define RERANK_DEBUG(x) std::cout << x << std::endl
    #define RERANK_DEBUG_NO_ENDL(x) std::cout << x
#else
    #define RERANK_DEBUG(x)
    #define RERANK_DEBUG_NO_ENDL(x)
#endif
namespace ov {

namespace genai {

// ============================================================================
// Variant String Conversion Utilities
// ============================================================================
namespace {

std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}
}  // namespace
RerankVariant parse_rerank_variant(const std::string& variant_str) {
    std::string lower = to_lower(variant_str);
    if (lower == "auto" || lower == "automatic") {
        return RerankVariant::AUTO;
    }
    if (lower == "bge" || lower == "bge-reranker" || lower == "baai") {
        return RerankVariant::BGE;
    }
    if (lower == "bce" || lower == "bce-reranker" || lower == "netease") {
        return RerankVariant::BCE;
    }
    if (lower == "generic" || lower == "default" || lower == "cross-encoder") {
        return RerankVariant::GENERIC;
    }
    std::ostringstream oss;
    oss << "Unknown reranking variant: '" << variant_str << "'. "
        << "Supported variants: ";
    auto supported = get_supported_rerank_variants();
    for (size_t i = 0; i < supported.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "'" << supported[i] << "'";
    }
    throw std::invalid_argument(oss.str());
}
std::string to_string(RerankVariant variant) {
    switch (variant) {
        case RerankVariant::AUTO:    return "auto";
        case RerankVariant::BGE:     return "bge";
        case RerankVariant::BCE:     return "bce";
        case RerankVariant::GENERIC: return "generic";
        default:                     return "unknown";
    }
}
std::vector<std::string> get_supported_rerank_variants() {
    return {"auto", "bge", "bce", "generic"};
}
// ============================================================================
// Normalization Helper Functions
// ============================================================================
namespace {
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
std::vector<float> apply_sigmoid(const std::vector<float>& scores) {
    std::vector<float> result;
    result.reserve(scores.size());
    for (float s : scores) {
        result.push_back(sigmoid(s));
    }
    return result;
}
std::vector<float> apply_softmax(const std::vector<float>& scores) {
    if (scores.empty()) return {};
    float max_val = *std::max_element(scores.begin(), scores.end());
    std::vector<float> exp_scores;
    exp_scores.reserve(scores.size());
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
std::vector<float> apply_minmax(const std::vector<float>& scores) {
    if (scores.empty()) return {};
    if (scores.size() == 1) return {1.0f};
    float min_val = *std::min_element(scores.begin(), scores.end());
    float max_val = *std::max_element(scores.begin(), scores.end());
    float range = max_val - min_val;
    if (range < 1e-8f) {
        return std::vector<float>(scores.size(), 0.5f);
    }
    std::vector<float> result;
    result.reserve(scores.size());
    for (float s : scores) {
        result.push_back((s - min_val) / range);
    }
    return result;
}
std::vector<float> normalize_scores(const std::vector<float>& scores, 
                                    ScoreNormalization method) {
    switch (method) {
        case ScoreNormalization::SIGMOID:
            return apply_sigmoid(scores);
        case ScoreNormalization::SOFTMAX:
            return apply_softmax(scores);
        case ScoreNormalization::MINMAX:
            return apply_minmax(scores);
        case ScoreNormalization::NONE:
        default:
            return scores;
    }
}
}  // namespace
// ============================================================================
// Config Loading Helper
// ============================================================================
namespace {
RerankingModelConfig load_config(const std::filesystem::path& root_dir,
                                 const RerankingModelConfig& user_config) {
    RerankingModelConfig config = user_config;
    std::filesystem::path config_path = root_dir / "config.json";
    if (std::filesystem::exists(config_path)) {
        std::ifstream file(config_path);
        nlohmann::json data = nlohmann::json::parse(file);
        if (user_config.max_seq_length == 512) {
            if (data.contains("max_position_embeddings")) {
                config.max_seq_length = data["max_position_embeddings"].get<size_t>();
            }
        }
    }
    return config;
}

}  // namespace
// ============================================================================
// Concrete Adapter Implementations
// ============================================================================
class GenericRerankingAdapter : public IRerankingAdapter {
public:
    RerankVariant get_variant() const override { return RerankVariant::GENERIC; }
    std::string get_variant_name() const override { return "Generic Cross-Encoder"; }
    std::string format_input(const std::string& query, const std::string& document,
                             const RerankingModelConfig& config) const override {
        std::string sep = config.separator.empty() ? " [SEP] " : config.separator;
        return query + sep + document;
    }
    std::vector<float> parse_output(const ov::Tensor& output, size_t batch_size,
                                    const RerankingModelConfig& config) const override {
        const float* data = output.data<float>();
        auto shape = output.get_shape();
        std::vector<float> scores;
        scores.reserve(batch_size);
        if (shape.size() == 1) {
            for (size_t i = 0; i < batch_size; ++i) scores.push_back(data[i]);
        } else if (shape.size() == 2) {
            size_t num_labels = shape[1];
            for (size_t i = 0; i < batch_size; ++i) {
                scores.push_back(data[i * num_labels + num_labels - 1]);
            }
        } else {
            throw std::runtime_error("Unexpected output shape");
        }
        if (config.normalize_scores) scores = normalize_scores(scores, config.normalization_method);
        return scores;
    }
    size_t get_default_max_seq_length() const override { return 512; }
    bool default_normalize_scores() const override { return false; }
    ScoreNormalization default_normalization_method() const override { return ScoreNormalization::SIGMOID; }
    PoolingStrategy default_pooling() const override { return PoolingStrategy::NONE; }
    void validate_output_shape(const ov::Shape& output_shape, size_t batch_size) const override {
        if (output_shape.empty()) throw std::runtime_error("Empty output shape");
        if (output_shape[0] != batch_size) throw std::runtime_error("Batch size mismatch");
    }
    std::string get_expected_output_description() const override {
        return "Shape: [batch_size] or [batch_size, num_labels]";
    }
};

class BgeRerankingAdapter : public IRerankingAdapter {
public:
    RerankVariant get_variant() const override { return RerankVariant::BGE; }
    std::string get_variant_name() const override { return "BGE Reranker"; }
    std::string format_input(const std::string& query, const std::string& document,
                             const RerankingModelConfig& config) const override {
        return query + " " + document;
    }
    std::vector<float> parse_output(const ov::Tensor& output, size_t batch_size,
                                    const RerankingModelConfig& config) const override {
        const float* data = output.data<float>();
        auto shape = output.get_shape();
        std::vector<float> scores;
        scores.reserve(batch_size);
        if (shape.size() == 1) {
            for (size_t i = 0; i < batch_size; ++i) scores.push_back(data[i]);
        } else if (shape.size() == 2) {
            size_t num_labels = shape[1];
            for (size_t i = 0; i < batch_size; ++i) {
                scores.push_back(data[i * num_labels + num_labels - 1]);
            }
        } else {
            throw std::runtime_error("BGE: Unexpected output shape");
        }
        if (config.normalize_scores) scores = normalize_scores(scores, config.normalization_method);
        return scores;
    }
    size_t get_default_max_seq_length() const override { return 512; }
    bool default_normalize_scores() const override { return true; }
    ScoreNormalization default_normalization_method() const override { return ScoreNormalization::SIGMOID; }
    PoolingStrategy default_pooling() const override { return PoolingStrategy::NONE; }
    void validate_output_shape(const ov::Shape& output_shape, size_t batch_size) const override {
        if (output_shape.empty()) throw std::runtime_error("BGE: Empty output shape");
        if (output_shape[0] != batch_size) throw std::runtime_error("BGE: Batch size mismatch");
    }
    std::string get_expected_output_description() const override {
        return "BGE expects shape: [batch_size] or [batch_size, 1]";
    }
};

class BceRerankingAdapter : public IRerankingAdapter {
public:
    RerankVariant get_variant() const override { return RerankVariant::BCE; }
    std::string get_variant_name() const override { return "BCE Reranker"; }
    std::string format_input(const std::string& query, const std::string& document,
                             const RerankingModelConfig& config) const override {
        return query + " " + document;
    }
    std::vector<float> parse_output(const ov::Tensor& output, size_t batch_size,
                                    const RerankingModelConfig& config) const override {
        const float* data = output.data<float>();
        auto shape = output.get_shape();
        std::vector<float> scores;
        scores.reserve(batch_size);
        if (shape.size() == 1) {
            for (size_t i = 0; i < batch_size; ++i) scores.push_back(data[i]);
        } else if (shape.size() == 2) {
            size_t num_labels = shape[1];
            if (num_labels == 1) {
                for (size_t i = 0; i < batch_size; ++i) scores.push_back(data[i]);
            } else {
                for (size_t i = 0; i < batch_size; ++i) {
                    scores.push_back(data[i * num_labels + num_labels - 1]);
                }
            }
        } else if (shape.size() == 3) {
            size_t seq_len = shape[1];
            size_t hidden_size = shape[2];
            for (size_t i = 0; i < batch_size; ++i) {
                scores.push_back(data[i * seq_len * hidden_size]);
            }
        } else {
            throw std::runtime_error("BCE: Unexpected output shape");
        }
        if (config.normalize_scores) scores = normalize_scores(scores, config.normalization_method);
        return scores;
    }
    size_t get_default_max_seq_length() const override { return 512; }
    bool default_normalize_scores() const override { return true; }
    ScoreNormalization default_normalization_method() const override { return ScoreNormalization::SIGMOID; }
    PoolingStrategy default_pooling() const override { return PoolingStrategy::NONE; }
    void validate_output_shape(const ov::Shape& output_shape, size_t batch_size) const override {
        if (output_shape.empty()) throw std::runtime_error("BCE: Empty output shape");
        if (output_shape[0] != batch_size) throw std::runtime_error("BCE: Batch size mismatch");
    }
    std::string get_expected_output_description() const override {
        return "BCE expects shape: [batch_size], [batch_size, 1/2], or [batch_size, seq_len, hidden]";
    }
};
// ============================================================================
// Adapter Factory Implementation
// ============================================================================
std::map<RerankVariant, RerankingAdapterFactory::AdapterCreator>& 
RerankingAdapterFactory::get_registry() {
    static std::map<RerankVariant, AdapterCreator> registry;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        registry[RerankVariant::GENERIC] = []() { return std::make_shared<GenericRerankingAdapter>(); };
        registry[RerankVariant::BGE] = []() { return std::make_shared<BgeRerankingAdapter>(); };
        registry[RerankVariant::BCE] = []() { return std::make_shared<BceRerankingAdapter>(); };
    });
    return registry;
}
std::shared_ptr<IRerankingAdapter> RerankingAdapterFactory::create(RerankVariant variant) {
    if (variant == RerankVariant::AUTO) variant = RerankVariant::GENERIC;
    auto& registry = get_registry();
    auto it = registry.find(variant);
    if (it == registry.end()) throw std::invalid_argument("Unknown variant");
    return it->second();
}

RerankVariant RerankingAdapterFactory::detect_variant(const std::filesystem::path& root_dir) {
    std::filesystem::path config_path = root_dir / "config.json";
    if (!std::filesystem::exists(config_path)) return RerankVariant::GENERIC;
    try {
        std::ifstream file(config_path);
        nlohmann::json data = nlohmann::json::parse(file);
        std::string name_or_path;
        if (data.contains("_name_or_path")) {
            name_or_path = to_lower(data["_name_or_path"].get<std::string>());
        }
        if (name_or_path.find("bge") != std::string::npos || 
            name_or_path.find("baai") != std::string::npos) return RerankVariant::BGE;
        if (name_or_path.find("bce") != std::string::npos || 
            name_or_path.find("netease") != std::string::npos) return RerankVariant::BCE;
    } catch (const std::exception& e) {
        RERANK_DEBUG("[RerankingAdapterFactory::detect_variant] Warning: Failed to parse config.json: " << e.what());
    } catch (...) {
        RERANK_DEBUG("[RerankingAdapterFactory::detect_variant] Warning: Unknown error parsing config.json");
    }
    return RerankVariant::GENERIC;
}

void RerankingAdapterFactory::register_adapter(RerankVariant variant, AdapterCreator factory_func) {
    get_registry()[variant] = std::move(factory_func);
}
// ============================================================================
// Inference Implementation Classes
// ============================================================================
class RerankingModelInferenceBase {
public:
    virtual ~RerankingModelInferenceBase() = default;
    virtual void compile(std::shared_ptr<ov::Model> model, const std::string& device, 
                        const ov::AnyMap& properties) = 0;
    virtual void set_adapters(AdapterController& adapter_controller, const AdapterConfig& adapters) = 0;
    virtual ov::Tensor infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) = 0;
    virtual ov::InferRequest& get_request() = 0;
};

class RerankingModelInferenceDynamic : public RerankingModelInferenceBase {
public:
    void compile(std::shared_ptr<ov::Model> model, const std::string& device, 
                const ov::AnyMap& properties) override {
        ov::Core core;
        
        // =========================================================================
        // BUG FIX: Keep CompiledModel as member variable!
        // Previously it was a local variable that got destroyed, causing state
        // tensors to malfunction. This is critical for LoRA to work!
        // =========================================================================
        m_compiled_model = core.compile_model(model, device, properties);
        m_request = m_compiled_model.create_infer_request();
        
        // =========================================================================
        // DIAGNOSTIC: Check state tensors after compilation (debug builds only)
        // =========================================================================
#ifndef NDEBUG
        auto state = m_request.query_state();
        std::cout << "\n[COMPILE_DIAG] ========================================" << std::endl;
        std::cout << "[COMPILE_DIAG] Model compiled on device: " << device << std::endl;
        std::cout << "[COMPILE_DIAG] InferRequest state tensor count: " << state.size() << std::endl;
        
        if (state.empty()) {
            std::cout << "[COMPILE_DIAG] *** WARNING: No state tensors! ***" << std::endl;
            std::cout << "[COMPILE_DIAG] If LoRA was expected, model may have been" << std::endl;
            std::cout << "[COMPILE_DIAG] compiled before AdapterController modified it." << std::endl;
        } else {
            std::cout << "[COMPILE_DIAG] State tensor names (first 5):" << std::endl;
            for (size_t i = 0; i < std::min(state.size(), size_t(5)); ++i) {
                std::cout << "[COMPILE_DIAG]   " << i << ": " << state[i].get_name() << std::endl;
            }
            if (state.size() > 5) {
                std::cout << "[COMPILE_DIAG]   ... and " << (state.size() - 5) << " more" << std::endl;
            }
        }
        std::cout << "[COMPILE_DIAG] ========================================\n" << std::endl;
#endif
    }

    void set_adapters(AdapterController& adapter_controller, const AdapterConfig& adapters) override {
        OPENVINO_ASSERT(m_request, "Reranking model must be compiled first");
        
#ifndef NDEBUG
        // =========================================================================
        // DIAGNOSTIC: Check state BEFORE apply (debug builds only)
        // =========================================================================
        std::cout << "\n[SET_ADAPTERS_DIAG] ========================================" << std::endl;
        std::cout << "[SET_ADAPTERS_DIAG] set_adapters() called" << std::endl;
        
        auto state_before = m_request.query_state();
        std::cout << "[SET_ADAPTERS_DIAG] State count: " << state_before.size() << std::endl;
        
        if (state_before.empty()) {
            std::cout << "[SET_ADAPTERS_DIAG] *** CRITICAL: No state tensors! ***" << std::endl;
            std::cout << "[SET_ADAPTERS_DIAG] LoRA CANNOT work without state tensors!" << std::endl;
            std::cout << "[SET_ADAPTERS_DIAG] ========================================\n" << std::endl;
            return;
        }
        
        // Count non-zero state tensors BEFORE apply
        int non_zero_before = 0;
        for (size_t i = 0; i < std::min(state_before.size(), size_t(20)); ++i) {
            auto tensor = state_before[i].get_state();
            if (tensor.get_size() > 0 && tensor.get_element_type() == ov::element::f32) {
                float* data = tensor.data<float>();
                float sum = 0;
                for (size_t j = 0; j < std::min(tensor.get_size(), size_t(100)); ++j) {
                    sum += std::abs(data[j]);
                }
                if (sum > 1e-10) non_zero_before++;
            }
        }
        std::cout << "[SET_ADAPTERS_DIAG] Non-zero tensors BEFORE: " << non_zero_before << std::endl;
        std::cout << "[SET_ADAPTERS_DIAG] Calling adapter_controller.apply()..." << std::endl;
#endif
        
        // Apply LoRA weights
        adapter_controller.apply(m_request, adapters);
        
#ifndef NDEBUG
        std::cout << "[SET_ADAPTERS_DIAG] apply() completed" << std::endl;
        
        // =========================================================================
        // DIAGNOSTIC: Check state AFTER apply (debug builds only)
        // =========================================================================
        auto state_after = m_request.query_state();
        
        int non_zero_after = 0;
        int zero_count = 0;
        std::vector<std::pair<std::string, float>> sample_non_zero;
        std::vector<std::string> sample_zero;
        
        for (size_t i = 0; i < state_after.size(); ++i) {
            auto& s = state_after[i];
            auto tensor = s.get_state();
            
            if (tensor.get_size() > 0 && tensor.get_element_type() == ov::element::f32) {
                float* data = tensor.data<float>();
                float sum = 0;
                for (size_t j = 0; j < std::min(tensor.get_size(), size_t(100)); ++j) {
                    sum += std::abs(data[j]);
                }
                
                if (sum > 1e-10) {
                    non_zero_after++;
                    if (sample_non_zero.size() < 3) {
                        sample_non_zero.push_back({s.get_name(), sum});
                    }
                } else {
                    zero_count++;
                    if (sample_zero.size() < 3) {
                        sample_zero.push_back(s.get_name());
                    }
                }
            }
        }
        
        std::cout << "[SET_ADAPTERS_DIAG] Non-zero tensors AFTER: " << non_zero_after << std::endl;
        std::cout << "[SET_ADAPTERS_DIAG] Zero tensors AFTER: " << zero_count << std::endl;
        
        if (!sample_non_zero.empty()) {
            std::cout << "[SET_ADAPTERS_DIAG] Sample non-zero tensors:" << std::endl;
            for (const auto& [name, sum] : sample_non_zero) {
                std::cout << "[SET_ADAPTERS_DIAG]   " << name << " (sum=" << sum << ")" << std::endl;
            }
        }
        
        if (!sample_zero.empty() && zero_count > 0) {
            std::cout << "[SET_ADAPTERS_DIAG] Sample ZERO tensors:" << std::endl;
            for (const auto& name : sample_zero) {
                std::cout << "[SET_ADAPTERS_DIAG]   " << name << std::endl;
            }
        }
        
        // Final verdict
        if (non_zero_after == 0 && state_after.size() > 0) {
            std::cout << "[SET_ADAPTERS_DIAG] *** FAILURE: ALL state tensors are ZEROS! ***" << std::endl;
            std::cout << "[SET_ADAPTERS_DIAG] LoRA weights were NOT applied." << std::endl;
            std::cout << "[SET_ADAPTERS_DIAG] Likely: LoRA tensor names don't match model node names." << std::endl;
        } else if (non_zero_after > 0) {
            std::cout << "[SET_ADAPTERS_DIAG] SUCCESS: " << non_zero_after << " tensors have LoRA data!" << std::endl;
        }
        
        std::cout << "[SET_ADAPTERS_DIAG] ========================================\n" << std::endl;
#endif
    }
    
    ov::InferRequest& get_request() override { return m_request; }

    ov::Tensor infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) override {
        OPENVINO_ASSERT(m_request, "Reranking model must be compiled first");
        
        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("attention_mask", attention_mask);
        
        try {
            ov::Shape shape = input_ids.get_shape();
            ov::Tensor token_type_ids(ov::element::i64, shape);
            std::fill_n(token_type_ids.data<int64_t>(), token_type_ids.get_size(), 0);
            m_request.set_tensor("token_type_ids", token_type_ids);
        } catch (const std::exception& e) {
            RERANK_DEBUG("[RerankingModelInferenceDynamic::infer] Warning: Failed to set token_type_ids: " << e.what());
        } catch (...) {
            RERANK_DEBUG("[RerankingModelInferenceDynamic::infer] Warning: Unknown error setting token_type_ids");
        }
        
        m_request.infer();
        return m_request.get_output_tensor(0);
    }

private:
    // =========================================================================
    // BUG FIX: CompiledModel MUST be kept alive as a member variable!
    // =========================================================================
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_request;
};

class RerankingModelInferenceStatic : public RerankingModelInferenceBase {
public:
    void compile(std::shared_ptr<ov::Model> model, const std::string& device, 
                const ov::AnyMap& properties) override {
        ov::Core core;
        m_compiled_model = core.compile_model(model, device, properties);
        m_request = m_compiled_model.create_infer_request();
    }

    void set_adapters(AdapterController& adapter_controller, const AdapterConfig& adapters) override {
        OPENVINO_ASSERT(m_request, "Reranking model must be compiled first");
        adapter_controller.apply(m_request, adapters);
    }
    
    ov::InferRequest& get_request() override { return m_request; }

    ov::Tensor infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) override {
        OPENVINO_ASSERT(m_request, "Reranking model must be compiled first");
        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("attention_mask", attention_mask);
        m_request.infer();
        return m_request.get_output_tensor(0);
    }

private: 
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_request;
};

// ============================================================================
// RerankingModel Implementation
// ============================================================================
RerankingModel::RerankingModel(const std::filesystem::path& root_dir) {
    m_config = load_config(root_dir, RerankingModelConfig{});
    m_tokenizer = Tokenizer(root_dir);
    ov::Core core;
    std::filesystem::path model_path = root_dir / "openvino_model.xml";
    m_model = core.read_model(model_path);
    initialize_adapter(root_dir);
}

RerankingModel::RerankingModel(const std::filesystem::path& root_dir,
                               const std::string& device,
                               const ov::AnyMap& properties) 
    : RerankingModel(root_dir) {
    compile(device, properties);
}
RerankingModel::RerankingModel(const std::filesystem::path& root_dir,
                               const std::string& device,
                               const ov::AnyMap& properties,
                               const RerankingModelConfig& config) {
    m_config = load_config(root_dir, config);
    m_tokenizer = Tokenizer(root_dir);
    ov::Core core;
    std::filesystem::path model_path = root_dir / "openvino_model.xml";
    m_model = core.read_model(model_path);
    initialize_adapter(root_dir);
    compile(device, properties);
}

RerankingModel::RerankingModel(const std::string& model,
                               const Tokenizer& tokenizer,
                               const RerankingModelConfig& config)
    : m_tokenizer(tokenizer), m_config(config) {
    ov::Core core;
    m_model = core.read_model(model);
    RerankVariant resolved = m_config.variant;
    if (resolved == RerankVariant::AUTO) resolved = RerankVariant::GENERIC;
    m_adapter = RerankingAdapterFactory::create(resolved);
    m_config.variant = resolved;
    log_message("Initialized with variant: " + m_adapter->get_variant_name());
}

void RerankingModel::initialize_adapter(const std::filesystem::path& root_dir) {
    RerankVariant resolved = m_config.variant;
    if (resolved == RerankVariant::AUTO) {
        resolved = RerankingAdapterFactory::detect_variant(root_dir);
        log_message("Auto-detected variant: " + to_string(resolved));
    }
    m_adapter = RerankingAdapterFactory::create(resolved);
    m_config.variant = resolved;
    if (m_config.max_seq_length == 512) {
        m_config.max_seq_length = m_adapter->get_default_max_seq_length();
    }
    log_message("Initialized " + m_adapter->get_variant_name() + 
                " with max_seq_length=" + std::to_string(m_config.max_seq_length));
}
RerankingModel::RerankingModel(const RerankingModel&) = default;
RerankingModel& RerankingModel::operator=(const RerankingModel&) = default;
RerankingModel::RerankingModel(RerankingModel&&) noexcept = default;
RerankingModel& RerankingModel::operator=(RerankingModel&&) noexcept = default;
RerankingModel::~RerankingModel() = default;
void RerankingModel::log_message(const std::string& message) const {
    if (m_log_callback) m_log_callback("[RerankingModel] " + message);
}
void RerankingModel::validate_output_shape(const ov::Tensor& output, size_t expected_batch) const {
    auto shape = output.get_shape();
    m_adapter->validate_output_shape(shape, expected_batch);
}

RerankingModel& RerankingModel::reshape(int batch_size, int max_seq_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled");
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : m_model->inputs()) {
        std::string name = input.get_any_name();
        new_shapes[name] = ov::PartialShape{batch_size, max_seq_length};
    }
    m_model->reshape(new_shapes);
    log_message("Reshaped to batch=" + std::to_string(batch_size) + 
                ", seq_len=" + std::to_string(max_seq_length));
    return *this;
}

RerankingModel& RerankingModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled");
    
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    
#ifndef NDEBUG
    std::cout << "[DEBUG] Model inputs BEFORE AdapterController: " << m_model->inputs().size() << std::endl;
    for (const auto& input : m_model->inputs()) {
        std::cout << "[DEBUG]   - " << input.get_any_name() << std::endl;
    }
#endif
    
    if (adapters) {
        std::string prefix = adapters->get_tensor_name_prefix().value_or("");
        if (!adapters->get_tensor_name_prefix().has_value()) {
            adapters->set_tensor_name_prefix("");
            prefix = "";
        }
        
        RERANK_DEBUG("[DEBUG] Creating AdapterController with prefix='" << prefix << "'");
        RERANK_DEBUG("[DEBUG] Adapters provided: YES");
        
        m_adapter_controller = AdapterController(m_model, *adapters, device);
        
#ifndef NDEBUG
        std::cout << "[DEBUG] Model inputs AFTER AdapterController: " << m_model->inputs().size() << std::endl;
        std::cout << "[DEBUG] Model variables AFTER AdapterController: " << m_model->get_variables().size() << std::endl;
        std::cout << "[DEBUG] Model sinks AFTER AdapterController: " << m_model->get_sinks().size() << std::endl;



        if (m_model->get_variables().size() == 0) {
            std::cout << "[DEBUG] *** WARNING: NO LoRA variables added! ***" << std::endl;
        } else {
            std::cout << "[DEBUG] SUCCESS: " << m_model->get_variables().size() << " LoRA variables added" << std::endl;

            int count = 0;
            for (const auto& var : m_model->get_variables()) {
                if (count < 5) std::cout << "[DEBUG]   Variable: " << var->get_info().variable_id << std::endl;
                count++;
            }
            if (count > 5) std::cout << "[DEBUG]   ... and " << (count - 5) << " more" << std::endl;
        }
#endif
        
        log_message("AdapterController created with prefix='" + prefix + "'");
    } else {
        RERANK_DEBUG("[DEBUG] Adapters provided: NO (baseline mode)");
    }
    
    if (device.find("NPU") != std::string::npos) {
        m_impl = std::make_shared<RerankingModelInferenceStatic>();
        log_message("Using static inference for NPU");
    } else {
        m_impl = std::make_shared<RerankingModelInferenceDynamic>();
        log_message("Using dynamic inference for " + device);
    }
    
    m_impl->compile(m_model, device, *filtered_properties);
    
    if (adapters) {
        RERANK_DEBUG("[DEBUG] Calling set_adapters() to apply LoRA weights...");
        m_impl->set_adapters(m_adapter_controller, *adapters);
        RERANK_DEBUG("[DEBUG] LoRA weights applied");
        log_message("Initial LoRA adapters applied");
    }
    
    m_model.reset();
    log_message("Model compiled on " + device);
    return *this;
}

void RerankingModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_impl, "Reranking model must be compiled first");
    if (adapters.has_value()) {
        m_impl->set_adapters(m_adapter_controller, *adapters);
        log_message("LoRA adapters applied");
    } else {
        m_adapter_controller.apply(m_impl->get_request(), std::nullopt);
        log_message("LoRA adapters disabled");
    }
}

RerankingResult RerankingModel::rerank(const std::string& query, 
                                       const std::vector<std::string>& documents,
                                       size_t top_k) {
    OPENVINO_ASSERT(m_impl, "Reranking model must be compiled first");
    OPENVINO_ASSERT(m_adapter, "Reranking adapter not initialized");
    OPENVINO_ASSERT(!query.empty(), "Query cannot be empty");
    OPENVINO_ASSERT(!documents.empty(), "Documents list cannot be empty");
    
    RerankingResult result;
    result.scores.reserve(documents.size());
    
    std::vector<std::string> formatted_inputs;
    formatted_inputs.reserve(documents.size());
    for (const auto& doc : documents) {
        formatted_inputs.push_back(m_adapter->format_input(query, doc, m_config));
    }
    
    auto tokenized = m_tokenizer.encode(formatted_inputs);
    ov::Tensor logits = infer(tokenized.input_ids, tokenized.attention_mask);
    validate_output_shape(logits, documents.size());
    result.scores = m_adapter->parse_output(logits, documents.size(), m_config);
    
    result.sorted_indices.resize(documents.size());
    std::iota(result.sorted_indices.begin(), result.sorted_indices.end(), 0);
    std::sort(result.sorted_indices.begin(), result.sorted_indices.end(),
              [&result](size_t a, size_t b) { return result.scores[a] > result.scores[b]; });
    
    if (top_k > 0 && top_k < result.sorted_indices.size()) {
        result.sorted_indices.resize(top_k);
    }
    
    log_message("Reranked " + std::to_string(documents.size()) + " documents");
    return result;
}

float RerankingModel::score(const std::string& query, const std::string& document) {
    auto result = rerank(query, {document});
    return result.scores[0];
}

ov::Tensor RerankingModel::infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) {
    OPENVINO_ASSERT(m_impl, "Reranking model must be compiled first");
    return m_impl->infer(input_ids, attention_mask);
}

RerankingModelConfig RerankingModel::get_config() const { return m_config; }

Tokenizer RerankingModel::get_tokenizer() const { return m_tokenizer; }

RerankVariant RerankingModel::get_variant() const { return m_config.variant; }
void RerankingModel::set_log_callback(RerankingLogCallback callback) { m_log_callback = std::move(callback); }
std::shared_ptr<IRerankingAdapter> RerankingModel::get_adapter() const { return m_adapter; }

}  // namespace genai
}  // namespace ov