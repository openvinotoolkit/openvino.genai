// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ctime>
#include <iomanip>

#include "logger.hpp"

std::ostream& operator << (std::ostream& stream, const LogLevel& value) {
    switch (value) {
    case LogLevel::INFO:
        stream << "INFO";
        break;
    case LogLevel::DEBUG:
        stream << "DEBUG";
        break;
    case LogLevel::WARNING:
        stream << "WARNING";
        break;
    case LogLevel::ERROR:
        stream << "ERROR";
        break;
    default:
        stream << "UNKNOWN";
        break;
    }
    return stream;
}

Logger::Logger(const std::string& filename) {
    m_log_file.open(filename, std::ios::app);
}

std::string Logger::get_current_timestamp() {
    std::time_t currentTime = std::time(nullptr);
    struct std::tm* localTime = std::localtime(&currentTime);

    std::ostringstream oss;
    oss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void Logger::log_time(LogLevel level) {
    if (m_logging_enabled) {
        m_log_file << "[" << level << "] [" << get_current_timestamp() << "] " << std::endl;
    }
}

void Logger::log_string(LogLevel level, const std::string& message) {
    if (m_logging_enabled) {
        m_log_file << "[" << level << "] " << message << std::endl;
    }
}

void Logger::set_logging_enabled(bool enabled) {
    m_logging_enabled = enabled;
}

Logger::~Logger() {
    m_log_file.close();
}
