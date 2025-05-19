// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "openvino/core/except.hpp"

namespace ov::genai {
/**
 * @brief Represents the mode of how anchor points are formed in KVCrush Cache eviction algorithm
 */
enum class KVCrushAnchorPointMode {
    RANDOM, /**<In this mode the anchor point is a random binary vector of 0s and 1s > */
    ZEROS,  /**<In this mode the anchor point is a vector of 0s */
    ONES,   /**<In this mode the anchor point is a vector of 1s */
    MEAN, /**<In this mode the anchor point is a random binary vector of 0s and 1s, where individual values are decided
             based on majority value */
    ALTERNATE /**In this mode the anchor point is a vector of alternate 0s and 1s */
};

/**
 * @brief Configuration struct for the cache eviction algorithm.
 */
class KVCrushConfig {
public:
    KVCrushConfig() = default;

    KVCrushConfig(size_t budget_, KVCrushAnchorPointMode anchor_point_mode_, size_t rng_seed_ = 0)
        : budget(budget_),
          anchor_point_mode(anchor_point_mode_),
          rng_seed(rng_seed_) {}

    /*KVCrush Cache budget - number of tokens*/
    std::size_t budget = 128;
    /*KVCrush Anchor point mode*/
    KVCrushAnchorPointMode anchor_point_mode = KVCrushAnchorPointMode::RANDOM;
    size_t rng_seed = 0;
    std::size_t get_budget() const {
        return budget;
    }
};

}  // namespace ov::genai
