// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <optional>

#include "modeling/ops/context.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

/**
 * @brief Result of weight finalization, supporting both single and multi-tensor returns
 * 
 * For non-quantized weights:
 *   - primary: the weight tensor
 *   - auxiliary: empty
 * 
 * For quantized weights:
 *   - primary: the dequantized/decompressed weight (ready to use)
 *   - auxiliary: optional map of related tensors (scales, zero-points, compressed weights)
 */
struct FinalizedWeight {
    Tensor primary;  ///< Primary tensor (the actual weight to use in the model)
    std::unordered_map<std::string, Tensor> auxiliary;  ///< Optional auxiliary tensors (scales, zps, etc.)
    
    // Constructor for single tensor (backward compatibility)
    explicit FinalizedWeight(const Tensor& t) : primary(t) {}
    
    // Constructor for multi-tensor result
    FinalizedWeight(const Tensor& primary_tensor, 
                   const std::unordered_map<std::string, Tensor>& aux_tensors = {})
        : primary(primary_tensor), auxiliary(aux_tensors) {}
    
    // Implicit conversion to Tensor for backward compatibility
    operator Tensor() const { return primary; }
    
    // Check if auxiliary tensors are present
    bool has_auxiliary() const { return !auxiliary.empty(); }
    
    // Get auxiliary tensor by key
    std::optional<Tensor> get_auxiliary(const std::string& key) const {
        auto it = auxiliary.find(key);
        if (it != auxiliary.end()) {
            return it->second;
        }
        return std::nullopt;
    }
};

/**
 * @brief Base class for weight finalization strategies
 * 
 * Responsible for converting raw weight tensors from WeightSource into
 * finalized Tensor objects ready to use in the model graph. This may involve:
 * - Type conversions (F16/BF16 -> F32)
 * - Quantization/dequantization
 * - Zero-copy optimizations
 * - Caching
 */
class WeightFinalizer {
public:
    virtual ~WeightFinalizer() = default;

    /**
     * @brief Finalize a weight tensor
     * 
     * @param name Weight name/key
     * @param source Weight source to load from
     * @param ctx Operation context
     * @return Finalized weight (may include auxiliary tensors for quantization)
     */
    virtual FinalizedWeight finalize(const std::string& name, WeightSource& source, OpContext& ctx) = 0;
    
    /**
     * @brief Legacy finalize method for backward compatibility
     * 
     * @deprecated Use finalize() that returns FinalizedWeight instead
     */
    virtual Tensor finalize_single(const std::string& name, WeightSource& source, OpContext& ctx) {
        return finalize(name, source, ctx).primary;
    }
};

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
