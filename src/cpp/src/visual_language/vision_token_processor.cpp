// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_token_processor.hpp"
#include <iostream>

namespace ov::genai {

VisionTokenProcessor::VisionTokenProcessor(const std::string& device, 
                                          const cdpruner::Config& config) {
    // Initialize CDPruner with provided configuration
    cdpruner::Config pruner_config = config;
    pruner_config.device = device;
    m_pruner = std::make_unique<cdpruner::CDPruner>(pruner_config);
}

ov::Tensor VisionTokenProcessor::process(const std::vector<ov::Tensor>& visual_features,
                                        const ov::Tensor& text_features) {
    if (!m_pruner) {
        // If pruner is not available, return empty tensor
        return ov::Tensor();
    }

    // Delegate to CDPruner for processing
    return m_pruner->apply_pruning(visual_features, text_features);
}

void VisionTokenProcessor::set_config(const cdpruner::Config& config) {
    if (!m_pruner) {
        // Create pruner if it doesn't exist
        m_pruner = std::make_unique<cdpruner::CDPruner>(config);
    } else {
        // Update existing pruner configuration
        if (!m_pruner->update_config(config)) {
            // If update fails, recreate the pruner
            m_pruner = std::make_unique<cdpruner::CDPruner>(config);
        }
    }
}

cdpruner::Config VisionTokenProcessor::get_config() const {
    if (!m_pruner) {
        return cdpruner::Config{};
    }
    return m_pruner->get_config();
}

std::optional<cdpruner::PruningStatistics> VisionTokenProcessor::get_last_statistics() const {
    if (!m_pruner) {
        return std::nullopt;
    }

    try {
        return m_pruner->get_last_pruning_statistics();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get pruning statistics: " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::vector<std::vector<size_t>> VisionTokenProcessor::get_last_selected_tokens() const {
    if (!m_pruner) {
        return {};
    }

    try {
        return m_pruner->get_last_selected_tokens();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get selected token indices: " << e.what() << std::endl;
        return {};
    }
}

} // namespace ov::genai
