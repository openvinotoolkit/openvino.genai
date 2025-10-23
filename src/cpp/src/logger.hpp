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
    Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    static std::shared_ptr<Logger>& get_instance() {
        static std::shared_ptr<Logger> instance;
        std::call_once(init_flag, [&]() {
            auto* obj_ptr = new Logger();
            OPENVINO_ASSERT(obj_ptr != nullptr);
            instance.reset(obj_ptr);
        });
        return instance;
    }
    virtual ~Logger() = default;
    void do_log(ov::log::Level level, const char* file, int line, const std::string& msg);

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 5, 6)))
#endif
    void log_format(ov::log::Level level, const char* file, int line, const char* format, ...);

    void validate_format_string(const char* format, size_t arg_count) const;

    void set_log_level(ov::log::Level level);

    bool should_log(ov::log::Level level) const;

private:
    std::atomic<ov::log::Level> log_level{ov::log::Level::NO};
    std::mutex log_mutex;
    static std::once_flag init_flag;
    void write_message(ov::log::Level level, const char* file, int line, const std::string& msg);
    void log_format_impl(ov::log::Level level, const char* file, int line, const char* format, va_list args);
    std::string format_from_variadic(const char* format, va_list args) const;
    std::string_view get_filename(std::string_view file_path) const;
    std::ostream& format_prefix(std::ostream& out, ov::log::Level level, const char* file, int line) const;
};
namespace detail {

inline void log_message(ov::log::Level level, const char* file, int line, const std::string& msg) {
    auto& logger = Logger::get_instance();
    logger->do_log(level, file, line, msg);
}

inline void log_message(ov::log::Level level, const char* file, int line, const char* msg) {
    auto& logger = Logger::get_instance();
    if (!logger->should_log(level)) {
        return;
    }
    if (msg) {
        logger->validate_format_string(msg, 0);
    }
    logger->do_log(level, file, line, msg ? std::string(msg) : std::string());
}

template <typename... Args, typename = std::enable_if_t<(sizeof...(Args) > 0)>>
inline void log_message(ov::log::Level level, const char* file, int line, const char* format, Args&&... args) {
    auto& logger = Logger::get_instance();
    if (!logger->should_log(level)) {
        return;
    }
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
