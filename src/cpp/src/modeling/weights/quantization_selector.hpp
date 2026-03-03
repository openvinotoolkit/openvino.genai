// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/quantization_config.hpp"
#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

/**
 * @brief Helper class for weight finalizers that support quantization
 * 
 * This class provides common selection logic that can be used by any
 * weight finalizer implementation (Safetensors, GGUF, etc.)
 * 
 * Implements NNCF-compatible quantization strategy:
 * - Primary precision (INT4) for attention and MLP layers
 * - Backup precision (INT8_ASYM) for lm_head and embeddings
 * - Set backup_mode=primary_mode to use same mode for all layers
 * - Set backup_mode=NONE to skip quantizing sensitive layers
 * 
 * Usage:
 * ```cpp
 * class MyWeightFinalizer : public WeightFinalizer {
 * public:
 *     MyWeightFinalizer(const QuantizationConfig& config)
 *         : selector_(config) {}
 *     
 *     Tensor finalize(const std::string& name, WeightSource& source, OpContext& ctx) override {
 *         auto tensor = source.get_tensor(name);
 *         auto quant_mode = selector_.get_quantization_mode(name, tensor.get_shape());
 *         if (quant_mode != QuantizationConfig::Mode::NONE) {
 *             // Apply quantization with quant_mode
 *         }
 *         // Return result
 *     }
 * 
 * private:
 *     QuantizationSelector selector_;
 * };
 * ```
 */
class QuantizationSelector {
public:
    QuantizationSelector() = default;
    explicit QuantizationSelector(const QuantizationConfig& config);
    
    /**
     * @brief Check if a weight should be quantized based on selection config
     * @param name Weight name
     * @param shape Weight shape
     * @param dtype Weight data type (optional, for dtype-based filtering)
     * @return true if weight should be quantized
     */
    bool should_quantize(const std::string& name, 
                        const ov::Shape& shape,
                        ov::element::Type dtype = ov::element::undefined) const;
    
    /**
     * @brief Get the quantization mode for a specific weight (NNCF-style)
     * 
     * Returns:
     * - NONE: if weight should not be quantized
     * - Primary mode (e.g., INT4_SYM): for regular transformer layers
     * - Backup mode (e.g., INT8_ASYM): for lm_head and embeddings (when backup_mode != primary_mode)
     * 
     * @param name Weight name
     * @param shape Weight shape
     * @param dtype Weight data type (optional)
     * @return Quantization mode for this weight
     */
    QuantizationConfig::Mode get_quantization_mode(const std::string& name,
                                                   const ov::Shape& shape,
                                                   ov::element::Type dtype = ov::element::undefined) const;
    
    /**
     * @brief Get group size for a weight (may vary by layer type in future)
     * @param name Weight name
     * @return Group size to use
     */
    int get_group_size(const std::string& name) const;
    
    /**
     * @brief Get the quantization configuration
     */
    const QuantizationConfig& config() const { return config_; }
    
    /**
     * @brief Check if quantization is enabled
     */
    bool enabled() const { return config_.enabled(); }
    
private:
    /**
     * @brief Check if layer is considered "sensitive" (needs backup precision)
     */
    bool is_sensitive_layer(const std::string& name) const;

    QuantizationConfig config_;
};

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
