// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "layer_selection_strategy.hpp"
#include "logger.hpp"

#include <algorithm>
#include <sstream>
#include <openvino/core/except.hpp>

namespace ov {
namespace genai {

LayerSelectionStrategy::LayerSelectionStrategy(const Qwen3MoeConfig& config)
    : config_(config) {
    // Precompute layer types for all layers
    precompute_layer_types();
    
    // Validate configuration
    OPENVINO_ASSERT(validate(), "LayerSelectionStrategy configuration validation failed");
    
    // Log layer type schedule
    auto stats = get_layer_statistics();
    GENAI_INFO("LayerSelectionStrategy initialized: %d MoE layers, %d MLP layers (total: %d layers)",
               stats.first, stats.second, config_.num_hidden_layers);
    
    // Log detailed schedule if debug logging is enabled
    if (Logger::get_instance().should_log(ov::log::Level::DEBUG)) {
        auto schedule = get_layer_type_schedule();
        std::ostringstream oss;
        oss << "Layer type schedule: [";
        for (size_t i = 0; i < schedule.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << schedule[i];
        }
        oss << "]";
        GENAI_DEBUG("%s", oss.str().c_str());
    }
}

bool LayerSelectionStrategy::is_moe_layer(int layer_idx) const {
    OPENVINO_ASSERT(
        layer_idx >= 0 && layer_idx < static_cast<int>(layer_types_.size()),
        "Layer index ", layer_idx, " is out of range [0, ", layer_types_.size(), ")"
    );
    return layer_types_[layer_idx];
}

std::vector<std::string> LayerSelectionStrategy::get_layer_type_schedule() const {
    std::vector<std::string> schedule;
    schedule.reserve(layer_types_.size());
    
    for (bool is_moe : layer_types_) {
        schedule.push_back(is_moe ? "moe" : "mlp");
    }
    
    return schedule;
}

bool LayerSelectionStrategy::validate() const {
    // Check mlp_only_layers validity
    for (int layer_idx : config_.mlp_only_layers) {
        if (layer_idx < 0 || layer_idx >= config_.num_hidden_layers) {
            GENAI_ERR("mlp_only_layers contains invalid layer index: %d (must be in range [0, %d))",
                      layer_idx, config_.num_hidden_layers);
            return false;
        }
    }
    
    // Check for duplicate indices in mlp_only_layers
    std::vector<int> sorted_mlp_layers = config_.mlp_only_layers;
    std::sort(sorted_mlp_layers.begin(), sorted_mlp_layers.end());
    auto it = std::adjacent_find(sorted_mlp_layers.begin(), sorted_mlp_layers.end());
    if (it != sorted_mlp_layers.end()) {
        GENAI_ERR("mlp_only_layers contains duplicate index: %d", *it);
        return false;
    }
    
    // Check decoder_sparse_step validity
    if (config_.decoder_sparse_step <= 0) {
        GENAI_WARN("decoder_sparse_step is %d (non-positive), all layers will use MLP",
                   config_.decoder_sparse_step);
    } else if (config_.num_experts <= 0) {
        GENAI_WARN("num_experts is %d (non-positive), all layers will use MLP",
                   config_.num_experts);
    } else {
        // Check if configuration results in at least one MoE layer
        auto stats = get_layer_statistics();
        if (stats.first == 0) {
            GENAI_WARN("Configuration results in no MoE layers (all %d layers use MLP). "
                       "Check decoder_sparse_step=%d and mlp_only_layers size=%zu",
                       config_.num_hidden_layers, config_.decoder_sparse_step,
                       config_.mlp_only_layers.size());
        }
    }
    
    // Check for conflicts: decoder_sparse_step suggests MoE but mlp_only_layers excludes all
    if (config_.decoder_sparse_step > 0 && config_.num_experts > 0) {
        int potential_moe_count = 0;
        for (int i = 0; i < config_.num_hidden_layers; ++i) {
            if ((i + 1) % config_.decoder_sparse_step == 0) {
                potential_moe_count++;
            }
        }
        
        auto stats = get_layer_statistics();
        if (potential_moe_count > 0 && stats.first == 0) {
            GENAI_WARN("decoder_sparse_step=%d suggests %d MoE layers, but mlp_only_layers excludes all of them",
                       config_.decoder_sparse_step, potential_moe_count);
        }
    }
    
    return true;
}

std::pair<int, int> LayerSelectionStrategy::get_layer_statistics() const {
    int num_moe = 0;
    int num_mlp = 0;
    
    for (bool is_moe : layer_types_) {
        if (is_moe) {
            num_moe++;
        } else {
            num_mlp++;
        }
    }
    
    return {num_moe, num_mlp};
}

void LayerSelectionStrategy::precompute_layer_types() {
    layer_types_.clear();
    layer_types_.reserve(config_.num_hidden_layers);
    
    for (int layer_idx = 0; layer_idx < config_.num_hidden_layers; ++layer_idx) {
        // Check if layer is explicitly marked as MLP-only
        if (is_in_mlp_only_layers(layer_idx)) {
            layer_types_.push_back(false);  // Use MLP
            continue;
        }
        
        // If mlp_only_layers is empty and decoder_sparse_step > 0 and num_experts > 0,
        // use decoder_sparse_step to determine MoE layers
        if (config_.mlp_only_layers.empty() && 
            config_.decoder_sparse_step > 0 && 
            config_.num_experts > 0) {
            // Layer uses MoE if (layer_idx + 1) is divisible by decoder_sparse_step
            bool is_moe = ((layer_idx + 1) % config_.decoder_sparse_step) == 0;
            layer_types_.push_back(is_moe);
        } else {
            // If mlp_only_layers is not empty, only layers not in mlp_only_layers
            // and passing the sparse_step check should use MoE
            if (config_.decoder_sparse_step > 0 && config_.num_experts > 0) {
                bool is_moe = ((layer_idx + 1) % config_.decoder_sparse_step) == 0;
                layer_types_.push_back(is_moe);
            } else {
                // decoder_sparse_step <= 0 or num_experts <= 0: all layers use MLP
                layer_types_.push_back(false);
            }
        }
    }
}

bool LayerSelectionStrategy::is_in_mlp_only_layers(int layer_idx) const {
    return std::find(config_.mlp_only_layers.begin(), 
                     config_.mlp_only_layers.end(), 
                     layer_idx) != config_.mlp_only_layers.end();
}

}  // namespace genai
}  // namespace ov