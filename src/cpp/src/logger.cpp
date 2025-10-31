// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "logger.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ov::genai {

ov::log::Level get_openvino_env_log_level() {
    const char* env = std::getenv("OPENVINO_LOG_LEVEL");
    if (!env) {
        return ov::log::Level::NO;
    }
    try {
        std::string env_str(env);
        size_t idx = 0;
        const auto env_var_value = std::stoi(env_str, &idx);
        if (idx != env_str.size()) {
            return ov::log::Level::NO;
        }
        return static_cast<ov::log::Level>(env_var_value);
    } catch (...) {
        return ov::log::Level::NO;
    }
}

ov::log::Level get_cur_log_level() {
    static ov::log::Level cur_log_level = get_openvino_env_log_level();
    return cur_log_level;
}

std::once_flag Logger::init_flag;

Logger::Logger() {
    log_level.store(get_cur_log_level(), std::memory_order_relaxed);
}

void Logger::do_log(ov::log::Level level, const char* file, int line, const std::string& msg) {
    if (!should_log(level)) {
        return;
    }
    write_message(level, file, line, msg);
}

void Logger::log_format(ov::log::Level level, const char* file, int line, const char* format, ...) {
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

void Logger::validate_format_string(const char* format, size_t arg_count) const {
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
            throw std::runtime_error(std::string{"Logger format error: unsupported specifier '%"} + conversion + "'");
        }

        ++placeholder_count;
        ++ptr;
    }

    if (placeholder_count != arg_count) {
        throw std::runtime_error("Logger format error: argument count mismatch");
    }
}

void Logger::set_log_level(ov::log::Level level) {
    log_level.store(level, std::memory_order_relaxed);
}

bool Logger::should_log(ov::log::Level level) const {
    const auto current_level = log_level.load(std::memory_order_relaxed);
    return current_level != ov::log::Level::NO && level != ov::log::Level::NO && level <= current_level;
}

void Logger::write_message(ov::log::Level level, const char* file, int line, const std::string& msg) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::ostream& out = (level == ov::log::Level::ERR) ? std::cerr : std::cout;
    format_prefix(out, level, file, line);
    out << msg;
    if (msg.empty() || msg.back() != '\n') {
        out << std::endl;
    }
}

void Logger::log_format_impl(ov::log::Level level, const char* file, int line, const char* format, va_list args) {
    std::string formatted = format_from_variadic(format, args);
    write_message(level, file, line, formatted);
}

std::string Logger::format_from_variadic(const char* format, va_list args) const {
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

std::string_view Logger::get_filename(std::string_view file_path) const {
    const auto index = file_path.find_last_of("/\\");
    if (index == std::string_view::npos) {
        return file_path;
    }
    return file_path.substr(index + 1);
}

std::ostream& Logger::format_prefix(std::ostream& out, ov::log::Level level, const char* file, int line) const {
    switch (level) {
    case ov::log::Level::DEBUG:
        out << "[DEBUG][" << get_filename(file) << ":" << line << "] ";
        break;
    case ov::log::Level::INFO:
        out << "[INFO] ";
        break;
    case ov::log::Level::WARNING:
        out << "[WARNING] ";
        break;
    case ov::log::Level::ERR:
        out << "[ERROR] ";
        break;
    default:
        out << "[LOG] ";
        break;
    }
    return out;
}

}  // namespace ov::genai
