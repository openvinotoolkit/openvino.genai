// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/perf_metrics.hpp"

namespace ov::genai {
enum class GenerationStatus {
    RUNNING = 0, // Default status for ongoing generation
    FINISHED = 1, // Status set when generation has been finished
    IGNORED = 2, // Status set when generation run into out-of-memory condition and could not be continued
    DROPPED_BY_PIPELINE = 3, // Currently not used, TODO: implement abort functionality
    DROPPED_BY_HANDLE = 4 // Status set when generation handle is dropped
};

struct EncodedGenerationResult {
    // request ID - obsolete when handle API is approved as handle will connect results with prompts.
    uint64_t m_request_id;

    // in a generic case we have multiple generation results per initial prompt
    // depending on sampling parameters (e.g. beam search or parallel sampling)
    std::vector<std::vector<int64_t>> m_generation_ids;
    // scores
    std::vector<float> m_scores;

    // Status of generation
    GenerationStatus m_status = GenerationStatus::RUNNING;
    
    // PerfMetrics but with empty tokenization/detokenization durations.
    PerfMetrics perf_metrics;
};

enum class GenerationFinishReason {
    NONE = 0, // Default value, when generation is not yet finished
    STOP = 1, // Generation finished naturally, by reaching end of sequence token
    LENGTH = 2 // Generation finished by reaching max_new_tokens limit
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
    GenerationStatus m_status = GenerationStatus::RUNNING;

    // PerfMetrics
    PerfMetrics perf_metrics;
};

struct GenerationOutput {
    std::vector<int64_t> generated_ids;
    std::vector<float> generated_log_probs;
    float score;
    GenerationFinishReason finish_reason;
};

using GenerationOutputs = std::unordered_map<uint64_t, GenerationOutput>;

class GenerationStream;

class OPENVINO_GENAI_EXPORTS GenerationHandleImpl {
    std::shared_ptr<GenerationStream> m_generation_stream;
    ov::genai::GenerationConfig m_sampling_params;

    bool is_dropped();
 
public:
    GenerationHandleImpl(std::shared_ptr<GenerationStream> generation_stream, const ov::genai::GenerationConfig& sampling_params) :
    m_generation_stream(std::move(generation_stream)),
    m_sampling_params(sampling_params) {};

    ~GenerationHandleImpl();

    // There can be only one handle for a request
    GenerationHandleImpl(const GenerationHandleImpl&) = delete;
    GenerationHandleImpl& operator=(const GenerationHandleImpl&) = delete;

    GenerationStatus get_status();

    bool can_read();

    void drop();

    GenerationOutputs back();
    // Reads result of a generation for single iteration
    GenerationOutputs read();
    // Reads all generated tokens for all sequences
    std::vector<GenerationOutput> read_all();
};

using GenerationHandle = std::shared_ptr<GenerationHandleImpl>;
}
