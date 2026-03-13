// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "benchmark_utils.hpp"

#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>

#include "openvino/core/except.hpp"
#include "stb_image.h"

namespace benchmark_utils {

namespace fs = std::filesystem;

namespace {

std::optional<ImageInfo> get_image_dimensions(const fs::path& image_path) {
    ImageInfo info;
    int channels = 0;

    if (stbi_info(image_path.string().c_str(), &info.width, &info.height, &channels)) {
        return info;
    }

    std::cerr << "Failed to read info for image '" << image_path.string() << "': " << stbi_failure_reason() << std::endl;
    return std::nullopt;
}

}  // namespace

std::vector<ImageInfo> check_images(const fs::path& input_path) {
    OPENVINO_ASSERT(!input_path.empty() && fs::exists(input_path), "Input path is empty or does not exist.");

    std::vector<ImageInfo> images;
    if (fs::is_directory(input_path)) {
        const std::set<fs::path> sorted_entries(fs::directory_iterator{input_path}, fs::directory_iterator{});
        images.reserve(sorted_entries.size());

        for (const fs::path& entry_path : sorted_entries) {
            if (!fs::is_regular_file(entry_path)) {
                continue;
            }
            if (auto info = get_image_dimensions(entry_path)) {
                images.push_back(*info);
            }
        }
    } else if (fs::is_regular_file(input_path)) {
        if (auto info = get_image_dimensions(input_path)) {
            images.push_back(*info);
        }
    }

    return images;
}

void Dataset::reserve(size_t size) {
    m_prompts.reserve(size);
    m_image_path.reserve(size);
    m_image_info.reserve(size);
    m_sampling_params.reserve(size);
    m_input_lens.reserve(size);
    m_output_lens.reserve(size);
}

void Dataset::push_data(std::string prompt,
                        ov::genai::GenerationConfig sampling_params,
                        const std::optional<std::string>& image_path) {
    m_prompts.push_back(std::move(prompt));
    m_sampling_params.push_back(std::move(sampling_params));

    if (image_path.has_value()) {
        m_image_path.push_back(*image_path);
        m_image_info.push_back(check_images(*image_path));
    } else {
        m_image_path.emplace_back();
        m_image_info.emplace_back();
    }
}

void Dataset::push_lens(size_t input_len, size_t output_len) {
    m_input_lens.push_back(input_len);
    m_output_lens.push_back(output_len);

    m_total_input_len += input_len;
    m_total_output_len += output_len;
}

float Dataset::get_average_input_len() const {
    OPENVINO_ASSERT(!empty());
    return static_cast<float>(m_total_input_len) / static_cast<float>(size());
}

float Dataset::get_average_output_len() const {
    OPENVINO_ASSERT(!empty());
    return static_cast<float>(m_total_output_len) / static_cast<float>(size());
}

bool Dataset::empty() const {
    return size() == 0;
}

size_t Dataset::size() const {
    return m_prompts.size();
}

GenerationInfo::SequenceInfo::SequenceInfo(const std::chrono::steady_clock::time_point& start_time) : start_time(start_time) {}

void GenerationInfo::SequenceInfo::update() {
    const std::chrono::steady_clock::time_point new_read_time = std::chrono::steady_clock::now();
    if (last_read_time.time_since_epoch() == std::chrono::milliseconds::zero()) {
        ttft = std::chrono::duration_cast<std::chrono::milliseconds>(new_read_time - start_time);
    } else {
        cumulated_tpot += std::chrono::duration_cast<std::chrono::milliseconds>(new_read_time - last_read_time);
        if (num_output_tokens > 0) {
            mean_tpot = cumulated_tpot / num_output_tokens;
        }
    }
    num_output_tokens++;
    last_read_time = new_read_time;
}

void GenerationInfo::SequenceInfo::update_generated_ids(const std::vector<int64_t>& gen_ids) {
    generated_ids.insert(generated_ids.end(), gen_ids.begin(), gen_ids.end());
}

GenerationInfo::GenerationInfo(ov::genai::GenerationHandle generation_handle,
                               size_t input_len,
                               std::vector<ImageInfo> input_img_info,
                               bool collect_generated_ids)
    : m_generation_handle(std::move(generation_handle)),
      m_start_time(std::chrono::steady_clock::now()),
      m_input_len(input_len),
      m_input_img_info(std::move(input_img_info)),
      m_collect_generated_ids(collect_generated_ids) {}

void GenerationInfo::update_sequence(int64_t sequence_id, ov::genai::GenerationOutput& gen_output) {
    if (m_sequences_info.find(sequence_id) == m_sequences_info.end()) {
        m_sequences_info.emplace(sequence_id, SequenceInfo(m_start_time));
    }

    SequenceInfo& sequence_info = m_sequences_info.at(sequence_id);
    sequence_info.update();
    if (m_collect_generated_ids) {
        sequence_info.update_generated_ids(gen_output.generated_ids);
    }
}

void GenerationInfo::update(ov::genai::GenerationOutputs& outputs) {
    for (auto& output : outputs) {
        update_sequence(output.first, output.second);
    }
}

ov::genai::GenerationOutputs GenerationInfo::read() {
    return m_generation_handle->read();
}

bool GenerationInfo::can_read() {
    return m_generation_handle->can_read();
}

bool GenerationInfo::is_finished() {
    return m_generation_handle->get_status() == ov::genai::GenerationStatus::FINISHED;
}

void GenerationInfo::set_inactive() {
    m_active = false;
}

bool GenerationInfo::is_active() {
    return m_active;
}

GenerationInfo::GenerationMetrics GenerationInfo::get_metrics() {
    GenerationMetrics generation_metrics;
    if (!m_sequences_info.empty()) {
        for (auto& sequence_info_pair : m_sequences_info) {
            generation_metrics.mean_ttft += sequence_info_pair.second.ttft;
            generation_metrics.mean_tpot += sequence_info_pair.second.mean_tpot;
            generation_metrics.num_output_tokens += sequence_info_pair.second.num_output_tokens;
        }
        generation_metrics.mean_ttft /= m_sequences_info.size();
        generation_metrics.mean_tpot /= m_sequences_info.size();
    }

    generation_metrics.num_input_tokens = m_input_len;
    generation_metrics.input_images_info = m_input_img_info;
    return generation_metrics;
}

std::vector<std::vector<int64_t>> GenerationInfo::get_generated_sequences() const {
    std::vector<std::vector<int64_t>> generated_sequences;
    generated_sequences.reserve(m_sequences_info.size());
    for (const auto& sequence_info_pair : m_sequences_info) {
        generated_sequences.push_back(sequence_info_pair.second.generated_ids);
    }
    return generated_sequences;
}

GenerationInfoCollector::GenerationInfoCollector(bool collect_generated_ids) : m_collect_generated_ids(collect_generated_ids) {}

void GenerationInfoCollector::set_start_time(std::chrono::steady_clock::time_point start_time) {
    m_start_time = start_time;
}

void GenerationInfoCollector::add_generation(ov::genai::ContinuousBatchingPipeline* pipe,
                                             Dataset* dataset,
                                             size_t request_id,
                                             bool is_speculative_decoding_enabled,
                                             const std::vector<ov::Tensor>* images) {
    auto sampling_params = dataset->m_sampling_params[request_id];
    if (is_speculative_decoding_enabled) {
        sampling_params.num_assistant_tokens = 5;
    }

    ov::genai::GenerationHandle generation_handle;
    if (images == nullptr) {
        generation_handle = pipe->add_request(request_id, dataset->m_prompts[request_id], sampling_params);
    } else {
        generation_handle = pipe->add_request(request_id, dataset->m_prompts[request_id], *images, sampling_params);
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_generations_info.emplace_back(std::move(generation_handle),
                                    dataset->m_input_lens[request_id],
                                    dataset->m_image_info[request_id],
                                    m_collect_generated_ids);
}

size_t GenerationInfoCollector::run() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (GenerationInfo& generation_info : m_generations_info) {
        if (!generation_info.is_active()) {
            continue;
        }

        if (generation_info.is_finished()) {
            m_num_finished++;
            generation_info.set_inactive();
        } else if (generation_info.can_read()) {
            auto outputs = generation_info.read();
            generation_info.update(outputs);
        }
    }
    return m_num_finished;
}

void GenerationInfoCollector::output_generated_text(ov::genai::Tokenizer& tokenizer) {
    std::lock_guard<std::mutex> lock(m_mutex);
    size_t request_idx = 0;
    for (const GenerationInfo& generation_info : m_generations_info) {
        const auto generated_sequences = generation_info.get_generated_sequences();
        for (const auto& generated_ids : generated_sequences) {
            const auto text = tokenizer.decode(generated_ids);
            std::cout << "[" << request_idx << "] generated text: " << text << std::endl;
        }
        request_idx++;
    }
}

void GenerationInfoCollector::print_statistics(bool print_per_generation) {
    const std::chrono::seconds total_duration =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_start_time);
    std::chrono::milliseconds mean_ttft = std::chrono::milliseconds::zero();
    std::chrono::milliseconds mean_tpot = std::chrono::milliseconds::zero();
    size_t total_input_len = 0;
    size_t total_output_len = 0;

    std::lock_guard<std::mutex> lock(m_mutex);
    std::cout << "Benchmark duration: " << total_duration.count() << " s" << std::endl;

    size_t generation_idx = 0;
    for (GenerationInfo& generation_info : m_generations_info) {
        auto generation_metrics = generation_info.get_metrics();
        mean_ttft += generation_metrics.mean_ttft;
        mean_tpot += generation_metrics.mean_tpot;
        total_input_len += generation_metrics.num_input_tokens;
        total_output_len += generation_metrics.num_output_tokens;

        if (print_per_generation) {
            std::cout << "[" << generation_idx << "] Input prompt tokens: " << generation_metrics.num_input_tokens << std::endl;
            size_t image_idx = 0;
            for (const ImageInfo& image_info : generation_metrics.input_images_info) {
                std::cout << "[" << generation_idx << "] Input image[" << image_idx << "]: width:" << image_info.width
                          << ", height:" << image_info.height << std::endl;
                image_idx++;
            }
            std::cout << "[" << generation_idx << "] Number of output tokens: " << generation_metrics.num_output_tokens
                      << std::endl;
        }

        generation_idx++;
    }

    OPENVINO_ASSERT(!m_generations_info.empty(), "No generation info collected");
    mean_ttft /= m_generations_info.size();
    mean_tpot /= m_generations_info.size();

    std::cout << "Total number of input tokens: " << total_input_len << std::endl;
    std::cout << "Total number of output tokens: " << total_output_len << std::endl;

    if (total_duration.count() > 0) {
        std::cout << "Input throughput: " << total_input_len / total_duration.count() << " tokens / s" << std::endl;
        std::cout << "Output throughput: " << total_output_len / total_duration.count() << " tokens / s" << std::endl;
    }

    std::cout << "Mean TTFT: " << mean_ttft.count() << " ms" << std::endl;
    std::cout << "Mean TPOT: " << mean_tpot.count() << " ms" << std::endl;
}

}  // namespace benchmark_utils
