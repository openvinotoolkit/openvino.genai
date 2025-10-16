// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include "openvino/genai/visibility.hpp"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <optional>

namespace ov {
namespace genai {

using TimePoint = std::chrono::steady_clock::time_point;
using MicroSeconds = std::chrono::duration<float, std::ratio<1, 1000000>>;

/**
 * @brief Structure with raw performance metrics for each generation before any statistics are calculated.
 *
 * @param generate_durations Durations for each generate call in microseconds.
 * @param tokenization_durations Durations for the tokenization process in microseconds.
 * @param detokenization_durations Durations for the detokenization process in microseconds.
 * @param m_times_to_first_token Times to the first token for each call in microseconds.
 * @param m_new_token_times Time points for each new token generated.
 * @param m_token_infer_durations Inference time for each token in microseconds.
 * @param m_batch_sizes Batch sizes for each generate call.
 * @param m_durations Total durations for each generate call in microseconds.
 * @param m_inference_durations Total inference duration for each generate call in microseconds.
 * @param m_grammar_compile_times Time to compile the grammar in microseconds.
 */
struct OPENVINO_GENAI_EXPORTS RawPerfMetrics {
    std::vector<MicroSeconds> generate_durations;
    std::vector<MicroSeconds> tokenization_durations;
    std::vector<MicroSeconds> detokenization_durations;

    std::vector<MicroSeconds> m_times_to_first_token;
    std::vector<TimePoint> m_new_token_times;
    std::vector<MicroSeconds> m_token_infer_durations;
    std::vector<size_t> m_batch_sizes;
    std::vector<MicroSeconds> m_durations;
    std::vector<MicroSeconds> m_inference_durations;

    std::vector<MicroSeconds> m_grammar_compile_times;
};

/**
* @brief Structure to store mean and standard deviation values.
*/
struct OPENVINO_GENAI_EXPORTS MeanStdPair {
    float mean = 0;
    float std = 0;
};

/**
* @brief Structure to store list of durations in milliseconds.
*/
struct OPENVINO_GENAI_EXPORTS SummaryStats {
    float mean = 0;
    float std = 0;
    float min = 0;
    float max = 0;
};

/**
 * @brief Holds performance metrics for each generate call.
 *
 * PerfMetrics holds fields with mean and standard deviations for the following metrics:
 * - Time To the First Token (TTFT), ms
 * - Time per Output Token (TPOT), ms/token
 * - Generate total duration, ms
 * - Tokenization duration, ms
 * - Detokenization duration, ms
 * - Throughput, tokens/s
 *
 * Additional fields include:
 * - Load time, ms
 * - Number of generated tokens
 * - Number of tokens in the input prompt
 *
 * Preverable way to access values is via get functions. Getters calculate mean and std values from raw_metrics are return pairs.
 * If mean and std were already calculated getters return cached values.
 * @param get_load_time Returns the load time in milliseconds.
 * @param get_num_generated_tokens Returns the number of generated tokens.
 * @param get_num_input_tokens Returns the number of tokens in the input prompt.
 * @param get_ttft Returns the mean and standard deviation of TTFT.
 * @param get_tpot Returns the mean and standard deviation of TPOT.
 * @param get_ipot Returns the mean and standard deviation of IPOT.
 * @param get_throughput Returns the mean and standard deviation of throughput.
 * @param get_inference_duration Returns the mean and standard deviation of inference duration.
 * @param get_generate_duration Returns the mean and standard deviation of generate duration.
 * @param get_tokenization_duration Returns the mean and standard deviation of tokenization duration.
 * @param get_detokenization_duration Returns the mean and standard deviation of detokenization duration.
 * @param get_grammar_compiler_init_times Returns a map with the time to initialize the grammar compiler for each backend in milliseconds.
 * @param get_grammar_compile_time Returns the time to compile the grammar in milliseconds.
 * @param get_microsec Converts a duration to microseconds.
 * @param m_evaluated Flag indicating if raw metrics were evaluated.
 *        If false, current mean/std TTFT, TPOT, etc. are not actual and evaluate_statistics() should recalculate them.
 * @param evaluate_statistics Calculates mean and standard deviation values from raw_metrics.
 *        Optional start_time can be provided to update durations.
 * @param operator+ Adds two PerfMetrics objects.
 * @param operator+= Adds and assigns the right-hand PerfMetrics to the current object.
 * @param raw_metrics A structure of RawPerfMetrics type that holds raw metrics.
 * @param load_time Load time in milliseconds.
 *
 * Cached mean and standard deviations.
 * @param ttft Mean and standard deviation of Time to the First Token (TTFT) in milliseconds.
 * @param tpot Mean and standard deviation of Time per Output Token (TPOT) in milliseconds per token.
 * @param ipot Mean and standard deviation of Inference Time per Output Token (IPOT) in milliseconds per token.
 * @param throughput Mean and standard deviation of tokens per second.
 * @param inference_duration Mean and standard deviation of the time spent on model inference during generate call in milliseconds.
 * @param generate_duration Mean and standard deviation of the total duration of generate calls in milliseconds.
 * @param tokenization_duration Mean and standard deviation of the tokenization duration in milliseconds.
 * @param detokenization_duration Mean and standard deviation of the detokenization duration in milliseconds.
 * @param num_generated_tokens Number of generated tokens.
 * @param num_input_tokens Number of tokens in the input prompt.
 */
struct OPENVINO_GENAI_EXPORTS PerfMetrics {
    float load_time = 0;   // Load time in ms.
    MeanStdPair ttft = {0, 0};  // Time to the first token (in ms) (TTFT).
    MeanStdPair tpot = {0, 0};  // Time (in ms) per output token (TPOT).
    MeanStdPair ipot = {0, 0};  // Inference time (in ms) per output token.
    MeanStdPair throughput = {0, 0};  // Tokens per second.

    // Time to initialize grammar compiler for each backend in ms.
    std::map<std::string, float> grammar_compiler_init_times;     
    SummaryStats grammar_compile_time;    // Time to compile grammar in ms.

    MeanStdPair generate_duration = {0, 0};
    MeanStdPair inference_duration = {0, 0};
    MeanStdPair tokenization_duration = {-1.0f, -1.0f};
    MeanStdPair detokenization_duration = {-1.0f, -1.0f};

    size_t num_generated_tokens = 0;
    size_t num_input_tokens = 0;

    float get_load_time();         // Load time in ms.
    size_t get_num_generated_tokens();
    size_t get_num_input_tokens();
    MeanStdPair get_ttft();         // Time to the first token (in ms) (TTFT).
    MeanStdPair get_tpot();         // Time (in ms) per output token (TPOT).
    MeanStdPair get_ipot();         // Inference time (in ms) per output token.
    MeanStdPair get_throughput();   // Tokens per second.
    
    std::map<std::string, float> get_grammar_compiler_init_times();
    SummaryStats get_grammar_compile_time();    // in ms

    MeanStdPair get_inference_duration();       // in ms
    MeanStdPair get_generate_duration();        // in ms
    MeanStdPair get_tokenization_duration();    // in ms
    MeanStdPair get_detokenization_duration();  // in ms

    // Flag indicating if raw metrics were evaluated.
    // If false means current mean/std ttft, tpot, etc. are not actual
    // and evaluate_statistics() should recalculate them.
    bool m_evaluated = false;

    /**
     * @brief calculates mean/std values from raw_metrics.
     *
     * @param start_time optional start_time in case if duration needs to be updated.
     */
    virtual void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt);

    /**
     * @brief convert duration to microseconds
     *
     * @param duration steady clock duration
     */
    static float get_microsec(std::chrono::steady_clock::duration duration);
    PerfMetrics operator+(const PerfMetrics& metrics) const;
    PerfMetrics& operator+=(const PerfMetrics& right);

    RawPerfMetrics raw_metrics;
};

// interface for creating perf metrics for python API
struct OPENVINO_GENAI_EXPORTS ExtendedPerfMetrics : public ov::genai::PerfMetrics {};

} // namespace genai
} // namespace ov
