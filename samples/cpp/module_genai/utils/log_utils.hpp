// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>

namespace log_utils {

// ============================================================================
// Verbose Logging
// ============================================================================

enum class LogLevel { QUIET = 0, ERROR = 1, INFO = 2, DEBUG = 3 };

// Global log level - default to INFO
inline LogLevel& get_log_level() {
    static LogLevel g_log_level = LogLevel::INFO;
    return g_log_level;
}

inline void set_log_level(LogLevel level) {
    get_log_level() = level;
}

inline void set_log_level(int level) {
    get_log_level() = static_cast<LogLevel>(level);
}

}  // namespace log_utils

// Logging macros - can be used directly in code
#define LOG_ERROR(msg) do { if (log_utils::get_log_level() >= log_utils::LogLevel::ERROR) { std::cerr << "[ERROR] " << msg << std::endl; } } while(0)
#define LOG_INFO(msg)  do { if (log_utils::get_log_level() >= log_utils::LogLevel::INFO)  { std::cout << "[INFO] " << msg << std::endl; } } while(0)
#define LOG_DEBUG(msg) do { if (log_utils::get_log_level() >= log_utils::LogLevel::DEBUG) { std::cout << "[DEBUG] " << msg << std::endl; } } while(0)
#define LOG_SUCCESS(msg) do { if (log_utils::get_log_level() >= log_utils::LogLevel::INFO) { std::cout << "[SUCCESS] " << msg << std::endl; } } while(0)
#define LOG_WARNING(msg) do { if (log_utils::get_log_level() >= log_utils::LogLevel::INFO) { std::cout << "[WARNING] " << msg << std::endl; } } while(0)
