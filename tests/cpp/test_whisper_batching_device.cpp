// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "whisper/whisper_utils.hpp"

using ov::genai::utils::is_whisper_batching_supported_device;

TEST(WhisperBatchingDevice, AcceptsCpuGpuAndGpuDeviceIds) {
    EXPECT_TRUE(is_whisper_batching_supported_device("CPU"));
    EXPECT_TRUE(is_whisper_batching_supported_device("GPU"));
    EXPECT_TRUE(is_whisper_batching_supported_device("GPU.0"));
    EXPECT_TRUE(is_whisper_batching_supported_device("GPU.1"));
}

TEST(WhisperBatchingDevice, RejectsNpuIncludingDeviceIds) {
    EXPECT_FALSE(is_whisper_batching_supported_device("NPU"));
    EXPECT_FALSE(is_whisper_batching_supported_device("NPU.0"));
}

TEST(WhisperBatchingDevice, RejectsMetaDevices) {
    EXPECT_FALSE(is_whisper_batching_supported_device("AUTO"));
    EXPECT_FALSE(is_whisper_batching_supported_device("AUTO:CPU,GPU"));
    EXPECT_FALSE(is_whisper_batching_supported_device("AUTO:GPU.0,CPU"));
    EXPECT_FALSE(is_whisper_batching_supported_device("AUTO:NPU,CPU"));
    EXPECT_FALSE(is_whisper_batching_supported_device("HETERO:CPU,GPU"));
    EXPECT_FALSE(is_whisper_batching_supported_device("MULTI:CPU,GPU"));
    EXPECT_FALSE(is_whisper_batching_supported_device("BATCH:GPU"));
}
