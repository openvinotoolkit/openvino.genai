// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <iostream>
#include <string>

#include "openvino/openvino.hpp"

namespace ov::genai {
inline ov::log::Level get_openvino_env_log_level() {
    const char* env = std::getenv("OPENVINO_LOG_LEVEL");
    if (!env)
        return ov::log::Level::NO;
    try {
        std::string env_str(env);
        size_t idx = 0;
        int env_var_value = std::stoi(env_str, &idx);
        if (idx != env_str.size()) {
            return ov::log::Level::NO;
        }
        return static_cast<ov::log::Level>(env_var_value);
    } catch (...) {
        return ov::log::Level::NO;
    }
}

inline ov::log::Level get_cur_log_level() {
    static ov::log::Level cur_log_level = get_openvino_env_log_level();
    return cur_log_level;
}

class Logger {
public:
    Logger() {
        log_level = get_cur_log_level();
    }
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    static Logger* get_instance() {
        static Logger instance;
        return &instance;
    }
    void do_log(ov::log::Level level, const char* file, int line, const std::string& msg) {
        static std::mutex log_mutex;
        std::lock_guard<std::mutex> lock(log_mutex);
        if (level <= log_level) {
            std::cout << format_prefix(level, file, line) << msg << std::endl;
        }
    }

private:
    ov::log::Level log_level = ov::log::Level::NO;
    std::string get_filename(const std::string& filePath) const {
        auto index = filePath.find_last_of("/\\");
        if (std::string::npos == index) {
            return filePath;
        }
        return filePath.substr(index + 1);
    }
    std::string format_prefix(ov::log::Level level, const char* file, int line) const {
        std::string level_str;
        switch (level) {
        case ov::log::Level::DEBUG:
            level_str = "[DEBUG][" + get_filename(file) + ":" + std::to_string(line) + "] ";
            break;
        case ov::log::Level::INFO:
            level_str = "[INFO]";
            break;
        case ov::log::Level::WARNING:
            level_str = "[WARNING]";
            break;
        case ov::log::Level::ERR:
            level_str = "[ERROR]";
            break;
        default:
            level_str = "[LOG]";
            break;
        }
        return level_str;
    }
};
#define GenAILogger ov::genai::Logger::get_instance()

#define GenAILogPrint(level, msg) GenAILogger->do_log(level, __FILE__, __LINE__, msg)

#define GENAI_DEBUG_LOG(msg)   GenAILogPrint(ov::log::Level::DEBUG, msg)
#define GENAI_INFO_LOG(msg)    GenAILogPrint(ov::log::Level::INFO, msg)
#define GENAI_WARNING_LOG(msg) GenAILogPrint(ov::log::Level::WARNING, msg)
#define GENAI_ERROR_LOG(msg)   GenAILogPrint(ov::log::Level::ERR, msg)

}  // namespace ov::genai
