// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/weights/quantization_config.hpp"

#include <regex>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

// ============================================================================
// WeightSelectionConfig Implementation
// ============================================================================

bool WeightSelectionConfig::matches_pattern(const std::string& name, 
                                            const std::vector<std::string>& patterns) const {
    if (patterns.empty()) {
        return false;
    }
    
    for (const auto& pattern : patterns) {
        // Convert wildcard pattern to regex
        std::string regex_pattern = pattern;
        
        // Escape special regex characters except * and ?
        std::string escaped;
        for (char c : regex_pattern) {
            if (c == '*') {
                escaped += ".*";  // * matches any sequence
            } else if (c == '?') {
                escaped += ".";   // ? matches single char
            } else if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' || 
                       c == '{' || c == '}' || c == '+' || c == '|' || c == '^' || 
                       c == '$' || c == '\\') {
                escaped += '\\';
                escaped += c;
            } else {
                escaped += c;
            }
        }
        
        try {
            std::regex re(escaped, std::regex::icase);
            if (std::regex_search(name, re)) {
                return true;
            }
        } catch (const std::regex_error& e) {
            std::cerr << "[WeightSelection] Invalid pattern '" << pattern 
                      << "': " << e.what() << std::endl;
        }
    }
    
    return false;
}

std::optional<int> WeightSelectionConfig::extract_layer_index(const std::string& name) const {
    // Match patterns like "layers[5]" or "layers.5"
    std::regex layer_regex(R"(layers[\[\.](\d+)[\]\.])", std::regex::icase);
    std::smatch match;
    
    if (std::regex_search(name, match, layer_regex)) {
        try {
            return std::stoi(match[1].str());
        } catch (...) {
            return std::nullopt;
        }
    }
    
    return std::nullopt;
}

QuantizationConfig parse_quantization_config_from_env() {
    QuantizationConfig quant_config;
    
    // Parse OV_GENAI_INFLIGHT_QUANT
    const char* quant_mode_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_MODE");
    if (quant_mode_env) {
        std::string mode_str(quant_mode_env);
        
        // Convert to uppercase for case-insensitive comparison
        std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
        
        if (mode_str == "INT4_SYM") {
            quant_config.mode = QuantizationConfig::Mode::INT4_SYM;
        } else if (mode_str == "INT4_ASYM") {
            quant_config.mode = QuantizationConfig::Mode::INT4_ASYM;
        } else if (mode_str == "INT8_SYM") {
            quant_config.mode = QuantizationConfig::Mode::INT8_SYM;
        } else if (mode_str == "INT8_ASYM") {
            quant_config.mode = QuantizationConfig::Mode::INT8_ASYM;
        } else {
            quant_config.mode = QuantizationConfig::Mode::NONE;
        }
    }
    
    // Parse group size
    const char* group_size_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE");
    if (group_size_env) {
        quant_config.group_size = std::atoi(group_size_env);
    }
    
    // Parse weight selection criteria
    WeightSelectionConfig& selection = quant_config.selection;
    
    // Include patterns
    const char* include_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_INCLUDE");
    if (include_env) {
        std::string patterns(include_env);
        std::istringstream iss(patterns);
        std::string pattern;
        while (std::getline(iss, pattern, ',')) {
            // Trim whitespace
            pattern.erase(0, pattern.find_first_not_of(" \t"));
            pattern.erase(pattern.find_last_not_of(" \t") + 1);
            if (!pattern.empty()) {
                selection.include_patterns.push_back(pattern);
            }
        }
    }
    
    // Exclude patterns
    const char* exclude_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_EXCLUDE");
    if (exclude_env) {
        std::string patterns(exclude_env);
        std::istringstream iss(patterns);
        std::string pattern;
        while (std::getline(iss, pattern, ',')) {
            pattern.erase(0, pattern.find_first_not_of(" \t"));
            pattern.erase(pattern.find_last_not_of(" \t") + 1);
            if (!pattern.empty()) {
                selection.exclude_patterns.push_back(pattern);
            }
        }
    }
    
    // Layer range
    const char* layer_range_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_LAYER_RANGE");
    if (layer_range_env) {
        std::string range(layer_range_env);
        size_t dash_pos = range.find('-');
        if (dash_pos != std::string::npos) {
            int start = std::atoi(range.substr(0, dash_pos).c_str());
            int end = std::atoi(range.substr(dash_pos + 1).c_str());
            selection.layer_range = std::make_pair(start, end);
        }
    }
    
    // Explicit weight names
    const char* names_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_WEIGHT_NAMES");
    if (names_env) {
        std::string names(names_env);
        std::istringstream iss(names);
        std::string name;
        while (std::getline(iss, name, ',')) {
            name.erase(0, name.find_first_not_of(" \t"));
            name.erase(name.find_last_not_of(" \t") + 1);
            if (!name.empty()) {
                selection.include_weights.push_back(name);
            }
        }
    }
    
    // Size thresholds
    const char* min_size_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_MIN_SIZE");
    if (min_size_env) {
        selection.min_weight_size = std::atoll(min_size_env);
    }
    
    const char* max_size_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_MAX_SIZE");
    if (max_size_env) {
        selection.max_weight_size = std::atoll(max_size_env);
    }
    
    // NNCF-style options
    // backup_mode: quantization mode for sensitive layers (lm_head, embeddings)
    // Default is INT8_ASYM (matching NNCF default)
    // Set to primary mode to quantize all layers with same mode
    // Set to NONE to skip quantizing sensitive layers
    const char* backup_mode_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE");
    if (backup_mode_env) {
        std::string mode_str(backup_mode_env);
        std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
        
        if (mode_str == "INT4_SYM") {
            quant_config.backup_mode = QuantizationConfig::Mode::INT4_SYM;
        } else if (mode_str == "INT4_ASYM") {
            quant_config.backup_mode = QuantizationConfig::Mode::INT4_ASYM;
        } else if (mode_str == "INT8_SYM") {
            quant_config.backup_mode = QuantizationConfig::Mode::INT8_SYM;
        } else if (mode_str == "INT8_ASYM") {
            quant_config.backup_mode = QuantizationConfig::Mode::INT8_ASYM;
        } else if (mode_str == "NONE") {
            quant_config.backup_mode = QuantizationConfig::Mode::NONE;
        }
    }
    
    // Verbose mode for debugging
    const char* verbose_env = std::getenv("OV_GENAI_INFLIGHT_QUANT_VERBOSE");
    if (verbose_env) {
        std::string val(verbose_env);
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        selection.verbose = (val == "1" || val == "true");
    }
    
    return quant_config;
}

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
