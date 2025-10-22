// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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
        log_level.store(get_cur_log_level(), std::memory_order_relaxed);
    }
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    static Logger* get_instance() {
        static Logger instance;
        return &instance;
    }
    void do_log(ov::log::Level level, const char* file, int line, const std::string& msg) {
        if (!should_log(level)) {
            return;
        }
        write_message(level, file, line, msg);
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 5, 6)))
#endif
    void log_format(ov::log::Level level, const char* file, int line, const char* format, ...) {
        if (!should_log(level)) {
            return;
        }
        va_list args;
        va_start(args, format);
        try {
            log_format_impl(level, file, line, format, args);
        } catch (const std::exception& ex) {
            va_end(args);
            throw std::runtime_error(std::string{"Logger format error: "} + ex.what());
        } catch (...) {
            va_end(args);
            throw std::runtime_error("Logger format error: unknown exception");
        }
        va_end(args);
    }

    inline void validate_format_string(const char* format, size_t arg_count) const {
        if (!format) {
            if (arg_count != 0) {
                throw std::runtime_error("Logger format error: null format string with arguments");
            }
            return;
        }

        size_t placeholder_count = 0;
        const char* ptr = format;
        while (*ptr != '\0') {
            if (*ptr != '%') {
                ++ptr;
                continue;
            }
            ++ptr;
            if (*ptr == '%') {
                ++ptr;
                continue;
            }
            if (*ptr == '\0') {
                throw std::runtime_error("Logger format error: dangling '%' at end of format string");
            }

            while (*ptr && std::strchr("-+ #0'", *ptr)) {
                ++ptr;
            }

            if (*ptr == '*') {
                throw std::runtime_error("Logger format error: '*' width specifier is not supported");
            }
            while (*ptr && std::isdigit(static_cast<unsigned char>(*ptr))) {
                ++ptr;
            }

            if (*ptr == '.') {
                ++ptr;
                if (*ptr == '*') {
                    throw std::runtime_error("Logger format error: '*' precision specifier is not supported");
                }
                while (*ptr && std::isdigit(static_cast<unsigned char>(*ptr))) {
                    ++ptr;
                }
            }

            if (*ptr == 'h' || *ptr == 'l' || *ptr == 'j' || *ptr == 'z' || *ptr == 't' || *ptr == 'L') {
                char length_modifier = *ptr;
                ++ptr;
                if ((length_modifier == 'h' || length_modifier == 'l') && *ptr == length_modifier) {
                    ++ptr;
                }
            }

            if (*ptr == '\0') {
                throw std::runtime_error("Logger format error: incomplete format specifier");
            }

            char conversion = *ptr;
            const char* allowed = "diuoxXfFeEgGaAcsp";
            if (conversion == 'n') {
                throw std::runtime_error("Logger format error: '%n' specifier is not supported");
            }
            if (!std::strchr(allowed, conversion)) {
                throw std::runtime_error(std::string{"Logger format error: unsupported specifier '%"} + conversion +
                                         "'");
            }

            ++placeholder_count;
            ++ptr;
        }

        if (placeholder_count != arg_count) {
            throw std::runtime_error("Logger format error: argument count mismatch");
        }
    }

    void set_log_level(ov::log::Level level) {
        log_level.store(level, std::memory_order_relaxed);
    }

    inline bool should_log(ov::log::Level level) const {
        const auto current_level = log_level.load(std::memory_order_relaxed);
        return current_level != ov::log::Level::NO && level != ov::log::Level::NO && level <= current_level;
    }

private:
    std::atomic<ov::log::Level> log_level{ov::log::Level::NO};
    std::mutex log_mutex;
    void write_message(ov::log::Level level, const char* file, int line, const std::string& msg) {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::cout << format_prefix(level, file, line) << msg;
        if (msg.empty() || msg.back() != '\n') {
            std::cout << std::endl;
        }
    }
    void log_format_impl(ov::log::Level level, const char* file, int line, const char* format, va_list args) {
        std::string formatted = format_from_variadic(format, args);
        write_message(level, file, line, formatted);
    }
    std::string format_from_variadic(const char* format, va_list args) const {
        if (!format) {
            return {};
        }
        va_list args_copy;
        va_copy(args_copy, args);
        int required = std::vsnprintf(nullptr, 0, format, args_copy);
        va_end(args_copy);
        if (required < 0) {
            throw std::runtime_error("Failed to format log message");
        }
        if (required == 0) {
            return {};
        }
        std::vector<char> buffer(static_cast<size_t>(required) + 1u, '\0');
        std::vsnprintf(buffer.data(), buffer.size(), format, args);
        return std::string(buffer.data(), static_cast<size_t>(required));
    }
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
            level_str = "[INFO] ";
            break;
        case ov::log::Level::WARNING:
            level_str = "[WARNING] ";
            break;
        case ov::log::Level::ERR:
            level_str = "[ERROR] ";
            break;
        default:
            level_str = "[LOG] ";
            break;
        }
        return level_str;
    }
};
namespace detail {

inline void log_message(ov::log::Level level, const char* file, int line, const std::string& msg) {
    Logger* logger = Logger::get_instance();
    logger->do_log(level, file, line, msg);
}

inline void log_message(ov::log::Level level, const char* file, int line, const char* msg) {
    Logger* logger = Logger::get_instance();
    if (msg) {
        logger->validate_format_string(msg, 0);
    }
    logger->do_log(level, file, line, msg ? std::string(msg) : std::string());
}

template <typename... Args, typename = std::enable_if_t<(sizeof...(Args) > 0)>>
inline void log_message(ov::log::Level level, const char* file, int line, const char* format, Args&&... args) {
    Logger* logger = Logger::get_instance();
    logger->validate_format_string(format, sizeof...(Args));
    logger->log_format(level, file, line, format, std::forward<Args>(args)...);
}

}  // namespace detail

#define GenAILogger ov::genai::Logger::get_instance()

#define GenAILogPrint(level, ...) ::ov::genai::detail::log_message(level, __FILE__, __LINE__, __VA_ARGS__)

#define GENAI_DEBUG_LOG(...)   GenAILogPrint(ov::log::Level::DEBUG, __VA_ARGS__)
#define GENAI_INFO_LOG(...)    GenAILogPrint(ov::log::Level::INFO, __VA_ARGS__)
#define GENAI_WARNING_LOG(...) GenAILogPrint(ov::log::Level::WARNING, __VA_ARGS__)
#define GENAI_ERROR_LOG(...)   GenAILogPrint(ov::log::Level::ERR, __VA_ARGS__)

}  // namespace ov::genai
