// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <atomic>
#include <cstdarg>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "openvino/openvino.hpp"

namespace ov::genai {
ov::log::Level get_openvino_env_log_level();
ov::log::Level get_cur_log_level();

class Logger {
public:
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;
    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }
    ~Logger() = default;
    void do_log(ov::log::Level level, const char* file, int line, const std::string& msg);

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 5, 6)))
#endif
    void log_format(ov::log::Level level, const char* file, int line, const char* format, ...);

    void set_log_level(ov::log::Level level);

    bool should_log(ov::log::Level level) const;

private:
    Logger();
    std::atomic<ov::log::Level> log_level{ov::log::Level::NO};
    std::mutex log_mutex;
    void write_message(ov::log::Level level, const char* file, int line, const std::string& msg);
    void log_format_impl(ov::log::Level level, const char* file, int line, const char* format, va_list args);
    std::string format_from_variadic(const char* format, va_list args) const;
    std::string_view get_filename(std::string_view file_path) const;
    std::ostream& format_prefix(std::ostream& out, ov::log::Level level, const char* file, int line) const;
};
namespace detail {

inline void log_message(ov::log::Level level, const char* file, int line, const std::string& msg) {
    Logger::get_instance().do_log(level, file, line, msg);
}

inline void log_message(ov::log::Level level, const char* file, int line, const char* msg) {
    Logger::get_instance().do_log(level, file, line, msg ? std::string(msg) : std::string());
}

template <typename... Args, typename = std::enable_if_t<(sizeof...(Args) > 0)>>
inline void log_message(ov::log::Level level, const char* file, int line, const char* format, Args&&... args) {
    Logger::get_instance().log_format(level, file, line, format, std::forward<Args>(args)...);
}

}  // namespace detail

#define GENAI_CHECK_LOG_LEVEL(LOG_LEVEL, ...)                                         \
    if (::ov::genai::Logger::get_instance().should_log(LOG_LEVEL)) {                  \
        ::ov::genai::detail::log_message(LOG_LEVEL, __FILE__, __LINE__, __VA_ARGS__); \
    }

#define GENAI_DEBUG(...) GENAI_CHECK_LOG_LEVEL(ov::log::Level::DEBUG, __VA_ARGS__)
#define GENAI_INFO(...)  GENAI_CHECK_LOG_LEVEL(ov::log::Level::INFO, __VA_ARGS__)
#define GENAI_WARN(...)  GENAI_CHECK_LOG_LEVEL(ov::log::Level::WARNING, __VA_ARGS__)
#define GENAI_ERR(...)   GENAI_CHECK_LOG_LEVEL(ov::log::Level::ERR, __VA_ARGS__)

}  // namespace ov::genai
