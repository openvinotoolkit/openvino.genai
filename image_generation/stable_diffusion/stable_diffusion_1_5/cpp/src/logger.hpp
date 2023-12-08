// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief a header file for logger
 * @file logger.hpp
 */


#pragma once

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

enum class LogLevel { INFO, DEBUG, WARNING, ERROR };

class Logger {
public:
    Logger(const std::string& filename) {
        logFile.open(filename, std::ios::app);
    }
    void log_time(LogLevel level) {
        if (loggingEnabled) {
            std::string levelStr = logLevelToString(level);
            std::string timestamp = getCurrentTimestamp();
            logFile << "[" << levelStr << "] [" << timestamp << "] " << std::endl;
        }
    }
    void log_string(LogLevel level, const std::string& message) {
        if (loggingEnabled) {
            std::string levelStr = logLevelToString(level);
            logFile << "[" << levelStr << "] " << message << std::endl;
        }
    }
    template <typename T>
    void log_value(LogLevel level, const std::string& message, T value) {
        if (loggingEnabled) {
            std::string levelStr = logLevelToString(level);
            logFile << "[" << levelStr << "] " << message << value << std::endl;
        }
    }
    template <typename T>
    void log_vector(LogLevel level,
                    const std::string& message,
                    const std::vector<T>& input_vec,
                    int32_t start_id,
                    int32_t num = 5) {
        if (loggingEnabled) {
            std::string levelStr = logLevelToString(level);
            logFile << "[" << levelStr << "] " << message;

            for (int32_t i = start_id; i < start_id + num && i < (int32_t)input_vec.size(); ++i) {
                logFile << input_vec[i] << " ";
            }
            logFile << std::endl;
        }
    }
    void setLoggingEnabled(bool enabled) {
        loggingEnabled = enabled;
    }

    ~Logger() {
        logFile.close();
    }

private:
    std::ofstream logFile;
    bool loggingEnabled;
    std::string logLevelToString(LogLevel level) {
        switch (level) {
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
        }
    }

    std::string getCurrentTimestamp() {
        std::time_t currentTime = std::time(nullptr);
        struct std::tm* localTime = std::localtime(&currentTime);

        std::ostringstream oss;
        oss << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }
};