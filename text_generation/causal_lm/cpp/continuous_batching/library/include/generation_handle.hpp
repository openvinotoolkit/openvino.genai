// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>

#include "generation_config.hpp"


enum class GenerationResultStatus {
    FINISHED = 0,
    IGNORED = 1,
    ABORTED = 2 // Currently not used, TODO: implement abort functionality
};

struct GenerationResult {
    // request ID - obsolete when handle API is approved as handle will connect results with prompts.
    uint64_t m_request_id;

    // in a generic case we have multiple generation results per initial prompt
    // depending on sampling parameters (e.g. beam search or parallel sampling)
    std::vector<std::string> m_generation_ids;
    // scores
    std::vector<float> m_scores;

    // Status of generation
    GenerationResultStatus m_status;
};

struct GenerationOutput {
    std::vector<int64_t> generated_token_ids;
    float score;
};

using GenerationOutputs = std::unordered_map<uint64_t, GenerationOutput>;

class GenerationStream;

class GenerationHandleImpl {
    std::shared_ptr<GenerationStream> m_generation_stream;
    GenerationConfig m_sampling_params;
 
public:
    GenerationHandleImpl(std::shared_ptr<GenerationStream> generation_stream, const GenerationConfig& sampling_params) :
    m_generation_stream(generation_stream),
    m_sampling_params(sampling_params) {};

    ~GenerationHandleImpl();

    // There can be only one handle for a request
    GenerationHandleImpl(const GenerationHandleImpl&) = delete;
    GenerationHandleImpl& operator=(const GenerationHandleImpl&) = delete;

    bool generation_finished();

    GenerationResultStatus get_finish_status();

    bool can_read();

    // Reads result of a generation for single iteration
    GenerationOutputs read();
    // Reads all generated tokens for all sequences
    std::vector<GenerationOutput> read_all();
};

using GenerationHandle = std::unique_ptr<GenerationHandleImpl>;
