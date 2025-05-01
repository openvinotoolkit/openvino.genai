// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/openvino.hpp"
using AttentionScoresForCacheOfSubsequence = ov::Tensor;
using AttentionScoresForEachDecoderLayer = std::vector<AttentionScoresForCacheOfSubsequence>;
using AttentionScoresForEachSubsequence = std::map<size_t, AttentionScoresForEachDecoderLayer>;
