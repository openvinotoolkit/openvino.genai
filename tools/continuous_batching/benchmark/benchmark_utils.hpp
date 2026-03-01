// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace benchmark_utils {

struct ImageInfo {
    int width = 0;
    int height = 0;
};

struct Dataset {
    std::vector<std::string> m_prompts;
    std::vector<std::string> m_image_path;
    std::vector<std::vector<ImageInfo>> m_image_info;
    std::vector<ov::genai::GenerationConfig> m_sampling_params;
    std::vector<size_t> m_input_lens;
    std::vector<size_t> m_output_lens;

    size_t m_total_input_len = 0;
    size_t m_total_output_len = 0;

    void reserve(size_t size);
    void push_data(std::string prompt,
                   ov::genai::GenerationConfig sampling_params,
                   const std::optional<std::string>& image_path = std::nullopt);
    void push_lens(size_t input_len, size_t output_len);

    float get_average_input_len() const;
    float get_average_output_len() const;

    bool empty() const;
    size_t size() const;
};

class GenerationInfo {
public:
    struct GenerationMetrics {
        std::chrono::milliseconds mean_ttft = std::chrono::milliseconds::zero();
        std::chrono::milliseconds mean_tpot = std::chrono::milliseconds::zero();
        size_t num_output_tokens = 0;
        size_t num_input_tokens = 0;
        std::vector<ImageInfo> input_images_info;
    };

    GenerationInfo(ov::genai::GenerationHandle generation_handle,
                   size_t input_len,
                   std::vector<ImageInfo> input_img_info,
                   bool collect_generated_ids);

    void update(ov::genai::GenerationOutputs& outputs);
    ov::genai::GenerationOutputs read();
    bool can_read();
    bool is_finished();
    void set_inactive();
    bool is_active();

    GenerationMetrics get_metrics();
    std::vector<std::vector<int64_t>> get_generated_sequences() const;

private:
    struct SequenceInfo {
        std::chrono::milliseconds ttft = std::chrono::milliseconds::zero();
        std::chrono::milliseconds cumulated_tpot = std::chrono::milliseconds::zero();
        std::chrono::milliseconds mean_tpot = std::chrono::milliseconds::zero();
        size_t num_output_tokens = 0;
        std::vector<int64_t> generated_ids;

        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point last_read_time;

        explicit SequenceInfo(const std::chrono::steady_clock::time_point& start_time);
        void update();
        void update_generated_ids(const std::vector<int64_t>& gen_ids);
    };

    void update_sequence(int64_t sequence_id, ov::genai::GenerationOutput& gen_output);

    ov::genai::GenerationHandle m_generation_handle;
    std::chrono::steady_clock::time_point m_start_time;
    std::unordered_map<int64_t, SequenceInfo> m_sequences_info;
    bool m_active = true;
    size_t m_input_len = 0;
    std::vector<ImageInfo> m_input_img_info;
    bool m_collect_generated_ids = false;
};

class GenerationInfoCollector {
public:
    explicit GenerationInfoCollector(bool collect_generated_ids = false);

    void set_start_time(std::chrono::steady_clock::time_point start_time);

    void add_generation(ov::genai::ContinuousBatchingPipeline* pipe,
                        Dataset* dataset,
                        size_t request_id,
                        bool is_speculative_decoding_enabled = false,
                        const std::vector<ov::Tensor>* images = nullptr);

    size_t run();
    void output_generated_text(ov::genai::Tokenizer& tokenizer);
    void print_statistics(bool print_per_generation = false);

private:
    std::mutex m_mutex;
    std::vector<GenerationInfo> m_generations_info;
    size_t m_num_finished = 0;
    std::chrono::steady_clock::time_point m_start_time;
    bool m_collect_generated_ids = false;
};

std::vector<ImageInfo> check_images(const std::filesystem::path& input_path);

}  // namespace benchmark_utils
