// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <openvino/openvino.hpp>
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "qwen3_moe_config.hpp"

namespace ov {
namespace genai {

/**
 * @brief Metadata structure for Qwen3-MoE model information.
 * 
 * Contains model statistics and configuration details useful for
 * understanding model characteristics and resource requirements.
 */
struct Qwen3MoeModelMetadata {
    std::string model_type = "qwen3_moe";
    std::string version;
    size_t num_parameters = 0;           // Total parameter count
    size_t num_active_parameters = 0;    // Parameters used per forward pass
    int num_moe_layers = 0;              // Count of MoE layers
    int num_mlp_layers = 0;              // Count of standard MLP layers
    int num_experts = 0;                 // Total number of experts
    int num_experts_per_tok = 0;         // Active experts per token
    int vocab_size = 0;
    int hidden_size = 0;
    int num_hidden_layers = 0;
};

/**
 * @brief Build Qwen3-MoE computation graph from configuration and weights.
 * 
 * Constructs the complete Qwen3-MoE model computation graph using OpenVINO
 * operator APIs. The graph includes embeddings, decoder layers with conditional
 * MLP/MoE selection, attention with Q/K normalization, RoPE, and output projection.
 * 
 * @param config_path Path to model configuration JSON file (config.json)
 * @param weights_path Path to model weights checkpoint (directory or file)
 * @param device Target device for compilation hints (e.g., "CPU", "GPU")
 * @return std::shared_ptr<ov::Model> Constructed OpenVINO model ready for compilation
 * @throws std::runtime_error if configuration is invalid or weights cannot be loaded
 * 
 * Example usage:
 * @code
 * auto model = build_qwen3_moe_model(
 *     "path/to/config.json",
 *     "path/to/weights",
 *     "CPU"
 * );
 * @endcode
 */
std::shared_ptr<ov::Model> build_qwen3_moe_model(
    const std::filesystem::path& config_path,
    const std::filesystem::path& weights_path,
    const std::string& device = "CPU"
);

/**
 * @brief Load and compile Qwen3-MoE model for inference.
 * 
 * Builds the model graph from configuration and weights, then compiles it
 * for the specified device with optional optimization properties.
 * 
 * @param model_path Path to model directory containing config.json and weights
 * @param device Target device for inference (e.g., "CPU", "GPU", "NPU")
 * @param config Optional compilation configuration properties
 * @return ov::CompiledModel Compiled model ready for inference
 * @throws std::runtime_error if model cannot be built or compiled
 * 
 * Example usage:
 * @code
 * ov::AnyMap properties = {
 *     {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
 *     {ov::hint::inference_precision(ov::element::f16)}
 * };
 * auto compiled = load_and_compile_qwen3_moe(
 *     "path/to/model",
 *     "CPU",
 *     properties
 * );
 * @endcode
 */
ov::CompiledModel load_and_compile_qwen3_moe(
    const std::filesystem::path& model_path,
    const std::string& device = "CPU",
    const ov::AnyMap& config = {}
);

/**
 * @brief Create a complete LLM pipeline for Qwen3-MoE model.
 * 
 * Builds the model, compiles it, loads the tokenizer, and creates a ready-to-use
 * LLMPipeline instance for text generation. This is the recommended high-level
 * API for using Qwen3-MoE models.
 * 
 * @param model_path Path to model directory containing all required files
 * @param device Target device for inference
 * @param properties Optional compilation and pipeline properties
 * @param generation_config Optional generation configuration (defaults loaded from model)
 * @return std::shared_ptr<LLMPipeline> Ready-to-use pipeline for text generation
 * @throws std::runtime_error if pipeline creation fails
 * 
 * Example usage:
 * @code
 * auto pipeline = create_qwen3_moe_pipeline(
 *     "path/to/model",
 *     "CPU",
 *     {},  // default properties
 *     {}   // default generation config
 * );
 * 
 * auto result = pipeline->generate("Hello, how are you?");
 * std::cout << result.texts[0] << std::endl;
 * @endcode
 */
std::shared_ptr<LLMPipeline> create_qwen3_moe_pipeline(
    const std::filesystem::path& model_path,
    const std::string& device = "CPU",
    const ov::AnyMap& properties = {},
    const ov::genai::GenerationConfig& generation_config = {}
);

/**
 * @brief Get metadata information about a Qwen3-MoE model.
 * 
 * Parses model configuration and computes statistics about model structure,
 * parameter counts, and layer composition.
 * 
 * @param model_path Path to model directory or config.json file
 * @return Qwen3MoeModelMetadata Structure containing model information
 * @throws std::runtime_error if configuration cannot be parsed
 * 
 * Example usage:
 * @code
 * auto metadata = get_qwen3_moe_metadata("path/to/model");
 * std::cout << "Model type: " << metadata.model_type << std::endl;
 * std::cout << "Total parameters: " << metadata.num_parameters << std::endl;
 * std::cout << "Active parameters: " << metadata.num_active_parameters << std::endl;
 * std::cout << "MoE layers: " << metadata.num_moe_layers << std::endl;
 * @endcode
 */
Qwen3MoeModelMetadata get_qwen3_moe_metadata(
    const std::filesystem::path& model_path
);

/**
 * @brief Validate Qwen3-MoE checkpoint integrity.
 * 
 * Checks that all required files exist and have valid formats:
 * - config.json with valid Qwen3-MoE configuration
 * - Weight files (safetensors, pytorch, or gguf format)
 * - tokenizer files (tokenizer.json, tokenizer_config.json)
 * 
 * @param checkpoint_path Path to model checkpoint directory
 * @return true if checkpoint is valid, false otherwise
 * 
 * Example usage:
 * @code
 * if (validate_qwen3_moe_checkpoint("path/to/model")) {
 *     std::cout << "Checkpoint is valid" << std::endl;
 * } else {
 *     std::cerr << "Invalid checkpoint" << std::endl;
 * }
 * @endcode
 */
bool validate_qwen3_moe_checkpoint(
    const std::filesystem::path& checkpoint_path
);

/**
 * @brief Convert HuggingFace checkpoint to OpenVINO format.
 * 
 * Loads a HuggingFace Qwen3-MoE checkpoint and converts it to optimized
 * OpenVINO format for faster loading and inference.
 * 
 * @param hf_path Path to HuggingFace checkpoint directory
 * @param ov_path Output path for OpenVINO format model
 * @param compress_weights Whether to compress weights (default: true)
 * @throws std::runtime_error if conversion fails
 * 
 * Example usage:
 * @code
 * convert_hf_checkpoint_to_ov(
 *     "path/to/hf/model",
 *     "path/to/ov/model",
 *     true  // compress weights
 * );
 * @endcode
 */
void convert_hf_checkpoint_to_ov(
    const std::filesystem::path& hf_path,
    const std::filesystem::path& ov_path,
    bool compress_weights = true
);

/**
 * @brief Register Qwen3-MoE model type with OpenVINO GenAI infrastructure.
 * 
 * Registers the Qwen3-MoE model type so it can be automatically detected
 * and loaded by the generic model loading APIs. This function should be
 * called during library initialization.
 * 
 * Note: Currently OpenVINO GenAI does not have a formal model type registry,
 * so this function is reserved for future use when such infrastructure is added.
 */
void register_qwen3_moe_model_type();

}  // namespace genai
}  // namespace ov