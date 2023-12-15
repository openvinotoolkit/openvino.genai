// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <string>
#include <vector>

enum class LogLevel { 
    INFO,
    DEBUG,
    WARNING,
    ERROR
};

std::ostream& operator << (std::ostream& stream, const LogLevel& value);

class Logger {
public:
    explicit Logger(const std::string& filename);

    void log_time(LogLevel level);

    void log_string(LogLevel level, const std::string& message);

    template <typename T>
    void log_value(LogLevel level, const std::string& message, T value) {
        if (m_logging_enabled) {
            m_log_file << "[" << level << "] " << message << value << std::endl;
        }
    }

    template <typename T>
    void log_vector(LogLevel level,
                    const std::string& message,
                    const std::vector<T>& input_vec,
                    int32_t start_id,
                    int32_t num = 5) {
        if (m_logging_enabled) {
            m_log_file << "[" << level << "] " << message;

            for (int32_t i = start_id; i < start_id + num && i < (int32_t)input_vec.size(); ++i) {
                m_log_file << input_vec[i] << " ";
            }
            m_log_file << std::endl;
        }
    }

    void set_logging_enabled(bool enabled);

    ~Logger();

private:
    std::ofstream m_log_file;
    bool m_logging_enabled;

    static std::string get_current_timestamp();
};
