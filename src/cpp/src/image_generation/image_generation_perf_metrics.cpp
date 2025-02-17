// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <numeric>
#include <cmath>

#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"

namespace ov {
namespace genai {
ov::genai::MeanStdPair calculation(const std::vector<ov::genai::MicroSeconds>& durations) {
    if (durations.size() == 0) {
        return {-1, -1};
    }
    // Accepts time durations in microseconds and returns standard deviation and mean in milliseconds.
    float mean = std::accumulate(durations.begin(),
                                 durations.end(),
                                 0.0f,
                                 [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                     return acc + duration.count() / 1000.0f;
                                 });
    mean /= durations.size();

    float sum_square_durations =
        std::accumulate(durations.begin(),
                        durations.end(),
                        0.0f,
                        [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                            auto d = duration.count() / 1000.0f;
                            return acc + d * d;
                        });
    float std = std::sqrt(sum_square_durations / durations.size() - mean * mean);
    return {mean, std};
}

void ImageGenerationPerfMetrics::clean_up() {
    m_evaluated = false;
    load_time = 0.f;
    generate_duration = 0.f;
    vae_encoder_inference_duration = 0.f;
    vae_decoder_inference_duration = 0.f;
    encoder_inference_duration.clear();
    raw_metrics.unet_inference_durations.clear();
    raw_metrics.transformer_inference_durations.clear();
    raw_metrics.iteration_durations.clear();
}

void ImageGenerationPerfMetrics::evaluate_statistics() {
    if (m_evaluated) {
        return;
    }

    // calc_mean_and_std will convert microsecond to milliseconds.
    unet_inference_duration = calculation(raw_metrics.unet_inference_durations);
    transformer_inference_duration = calculation(raw_metrics.transformer_inference_durations);
    iteration_duration = calculation(raw_metrics.iteration_durations);

    m_evaluated = true;
}

MeanStdPair ImageGenerationPerfMetrics::get_unet_infer_duration() {
    evaluate_statistics();
    return unet_inference_duration;
}

MeanStdPair ImageGenerationPerfMetrics::get_transformer_infer_duration() {
    evaluate_statistics();
    return transformer_inference_duration;
}

MeanStdPair ImageGenerationPerfMetrics::get_iteration_duration() {
    evaluate_statistics();
    return iteration_duration;
}

void ImageGenerationPerfMetrics::get_first_and_other_unet_infer_duration(float &first_infer, float &other_infer_avg) {
    first_infer = 0.0f;
    other_infer_avg = 0.0f;
    if (!raw_metrics.unet_inference_durations.empty()) {
        first_infer = raw_metrics.unet_inference_durations[0].count() / 1000.0f;
    }
    if (raw_metrics.unet_inference_durations.size() > 1) {
        float total = std::accumulate(raw_metrics.unet_inference_durations.begin() + 1,
                                      raw_metrics.unet_inference_durations.end(),
                                      0.0f,
                                      [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                          return acc + duration.count() / 1000.0f;
                                      });
        other_infer_avg = total / (raw_metrics.unet_inference_durations.size() - 1);
    }
}

void ImageGenerationPerfMetrics::get_first_and_other_trans_infer_duration(float &first_infer, float &other_infer_avg) {
    first_infer = 0.0f;
    other_infer_avg = 0.0f;
    if (!raw_metrics.transformer_inference_durations.empty()) {
        first_infer = raw_metrics.transformer_inference_durations[0].count() / 1000.0f;
    }
    if (raw_metrics.transformer_inference_durations.size() > 1) {
        float total = std::accumulate(raw_metrics.transformer_inference_durations.begin() + 1,
                                      raw_metrics.transformer_inference_durations.end(),
                                      0.0f,
                                      [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                          return acc + duration.count() / 1000.0f;
                                      });
        other_infer_avg = total / (raw_metrics.transformer_inference_durations.size() - 1);
    }
}

std::map<std::string, float> ImageGenerationPerfMetrics::get_text_encoder_infer_duration() const {
    return encoder_inference_duration;
}

float ImageGenerationPerfMetrics::get_vae_decoder_infer_duration() const {
    return vae_decoder_inference_duration;
}

float ImageGenerationPerfMetrics::get_vae_encoder_infer_duration() const {
    return vae_encoder_inference_duration;
}

float ImageGenerationPerfMetrics::get_inference_duration() {
    float total_duration = 0;
    if (!raw_metrics.unet_inference_durations.empty()) {
        float total = std::accumulate(raw_metrics.unet_inference_durations.begin(),
                                      raw_metrics.unet_inference_durations.end(),
                                      0.0f,
                                      [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                          return acc + duration.count() / 1000.0f;
                                      });
        total_duration += total;
    } else if (!raw_metrics.transformer_inference_durations.empty()) {
        float total = std::accumulate(raw_metrics.transformer_inference_durations.begin(),
                                      raw_metrics.transformer_inference_durations.end(),
                                      0.0f,
                                      [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                          return acc + duration.count() / 1000.0f;
                                      });
        total_duration += total;
    }

    for (auto encoder = encoder_inference_duration.begin(); encoder != encoder_inference_duration.end(); encoder++) {
        total_duration += encoder->second;
    }

    total_duration += vae_encoder_inference_duration;

    total_duration += vae_decoder_inference_duration;

    // Return milliseconds
    return total_duration;
}

float ImageGenerationPerfMetrics::get_load_time() const {
    return load_time;
}

float ImageGenerationPerfMetrics::get_generate_duration() {
    return generate_duration;
}

void ImageGenerationPerfMetrics::get_first_and_other_iter_duration(float &first_iter, float &other_iter_avg) {
    first_iter = 0.0f;
    other_iter_avg = 0.0f;
    if (!raw_metrics.iteration_durations.empty()) {
        first_iter = raw_metrics.iteration_durations[0].count() / 1000.0f;
    }
    if (raw_metrics.iteration_durations.size() > 1) {
        float total = std::accumulate(raw_metrics.iteration_durations.begin() + 1,
                                      raw_metrics.iteration_durations.end(),
                                      0.0f,
                                      [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                          return acc + duration.count() / 1000.0f;
                                      });
        other_iter_avg = total / (raw_metrics.iteration_durations.size() - 1);
    }
}

}  // namespace genai
}  // namespace ov