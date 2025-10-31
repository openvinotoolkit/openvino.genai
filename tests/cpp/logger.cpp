// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logger.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace {

void expect_contains(const std::string& haystack, const std::string& needle) {
    EXPECT_NE(haystack.find(needle), std::string::npos);
}

class LoggerTests : public ::testing::Test {
protected:
    void SetUp() override {
        GenAILogger->set_log_level(ov::log::Level::DEBUG);
    }

    void TearDown() override {
        GenAILogger->set_log_level(ov::log::Level::NO);
    }
};

}  // namespace

TEST_F(LoggerTests, SupportsPrintfStyleFormatting) {
    testing::internal::CaptureStdout();
    GenAILogPrint(ov::log::Level::INFO, "The value of %s is %d", "alpha", 42);
    std::string output = testing::internal::GetCapturedStdout();

    expect_contains(output, "[INFO] ");
    expect_contains(output, "The value of alpha is 42");
    ASSERT_FALSE(output.empty());
    EXPECT_EQ('\n', output.back());
}

TEST_F(LoggerTests, KeepsSingleTrailingNewline) {
    testing::internal::CaptureStdout();
    GenAILogPrint(ov::log::Level::INFO, "Message with newline\n");
    std::string output = testing::internal::GetCapturedStdout();

    expect_contains(output, "[INFO] ");
    EXPECT_EQ(1, std::count(output.begin(), output.end(), '\n'));
}

TEST_F(LoggerTests, InvalidFormatThrows) {
    EXPECT_THROW(GenAILogPrint(ov::log::Level::DEBUG, "%q"), std::runtime_error);
    EXPECT_THROW(GenAILogPrint(ov::log::Level::INFO, "The value of %s"), std::runtime_error);
    EXPECT_THROW(GenAILogPrint(ov::log::Level::INFO, "The value of %s is %d", "alpha"), std::runtime_error);
    EXPECT_THROW(GenAILogPrint(ov::log::Level::INFO, "The value of %s", "alpha", 42), std::runtime_error);
    EXPECT_THROW(GenAILogPrint(ov::log::Level::INFO, "The value with width %*d", 5, 42), std::runtime_error);
    EXPECT_THROW(GenAILogPrint(ov::log::Level::INFO, "Incomplete percent %"), std::runtime_error);
    EXPECT_THROW(GenAILogPrint(ov::log::Level::INFO, "Unsupported spec %k", 1), std::runtime_error);
    EXPECT_THROW(GenAILogPrint(ov::log::Level::INFO, "Disallowed spec %n"), std::runtime_error);
}

TEST_F(LoggerTests, ValidFormatDoesNotThrow) {
    EXPECT_NO_THROW(GenAILogPrint(ov::log::Level::INFO, "Hello, %s", "world"));
    EXPECT_NO_THROW(GenAILogPrint(ov::log::Level::INFO, "Hello, %s\n", "world"));
    EXPECT_NO_THROW(GenAILogPrint(ov::log::Level::INFO, "%d + %d = %d", 1, 2, 3));
    EXPECT_NO_THROW(GenAILogPrint(ov::log::Level::INFO, "Pi is approximately %.2f", 3.14159));
    EXPECT_NO_THROW(GenAILogPrint(ov::log::Level::INFO, "Hex: 0x%X", 255));
    EXPECT_NO_THROW(GenAILogPrint(ov::log::Level::INFO, "Pointer %p", static_cast<const void*>(this)));
}

TEST_F(LoggerTests, NoOutputWhenLevelIsNo) {
    GenAILogger->set_log_level(ov::log::Level::NO);
    testing::internal::CaptureStdout();
    GenAILogPrint(ov::log::Level::INFO, "Should not appear");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.empty());
}

TEST_F(LoggerTests, RespectsLogLevelFiltering) {
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();

    GenAILogger->set_log_level(ov::log::Level::WARNING);
    GenAILogPrint(ov::log::Level::DEBUG, "debug message");
    GenAILogPrint(ov::log::Level::INFO, "info message");
    GenAILogPrint(ov::log::Level::WARNING, "warn message");
    GenAILogPrint(ov::log::Level::ERR, "error message");

    std::string output = testing::internal::GetCapturedStdout();
    std::string error_output = testing::internal::GetCapturedStderr();

    EXPECT_EQ(output.find("debug message"), std::string::npos);
    EXPECT_EQ(output.find("info message"), std::string::npos);
    expect_contains(output, "[WARNING] warn message");
    EXPECT_EQ(error_output.find("debug message"), std::string::npos);
    EXPECT_EQ(error_output.find("info message"), std::string::npos);
    EXPECT_TRUE(error_output.find("[WARNING]") == std::string::npos);
    expect_contains(error_output, "[ERROR] error message");
}

TEST_F(LoggerTests, EmitsAllLogLevelsWithoutErrors) {
    GenAILogger->set_log_level(ov::log::Level::DEBUG);

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();

    GenAILogPrint(ov::log::Level::DEBUG, "debug level message");
    GenAILogPrint(ov::log::Level::INFO, "info level message");
    GenAILogPrint(ov::log::Level::WARNING, "warning level message");
    GenAILogPrint(ov::log::Level::ERR, "error level message");

    std::string std_output = testing::internal::GetCapturedStdout();
    std::string err_output = testing::internal::GetCapturedStderr();

    expect_contains(std_output, "[DEBUG][");
    expect_contains(std_output, ":");
    expect_contains(std_output, "debug level message");
    expect_contains(std_output, "[INFO] info level message");
    expect_contains(std_output, "[WARNING] warning level message");

    EXPECT_TRUE(err_output.find("debug level message") == std::string::npos);
    EXPECT_TRUE(err_output.find("info level message") == std::string::npos);
    EXPECT_TRUE(err_output.find("warning level message") == std::string::npos);
    expect_contains(err_output, "[ERROR] error level message");
}
