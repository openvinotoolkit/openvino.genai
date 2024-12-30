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

MeanStdPair ImageGenerationPerfMetrics::get_unet_inference_duration() {
    evaluate_statistics();
    return unet_inference_duration;
}

MeanStdPair ImageGenerationPerfMetrics::get_transformer_inference_duration() {
    evaluate_statistics();
    return transformer_inference_duration;
}
MeanStdPair ImageGenerationPerfMetrics::get_iteration_duration() {
    evaluate_statistics();
    return iteration_duration;
}

float ImageGenerationPerfMetrics::get_inference_total_duration() {
    float total_duration = 0;
    if (!raw_metrics.unet_inference_durations.empty()) {
        float total = std::accumulate(raw_metrics.unet_inference_durations.begin(),
                                      raw_metrics.unet_inference_durations.end(),
                                      0.0f,
                                      [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                          return acc + duration.count();
                                      });
        total_duration += total;
    } else if (!raw_metrics.transformer_inference_durations.empty()) {
        float total = std::accumulate(raw_metrics.transformer_inference_durations.begin(),
                                      raw_metrics.transformer_inference_durations.end(),
                                      0.0f,
                                      [](const float& acc, const ov::genai::MicroSeconds& duration) -> float {
                                          return acc + duration.count();
                                      });
        total_duration += total;
    }

    total_duration += vae_decoder_inference_duration;

    for (auto encoder = encoder_inference_duration.begin(); encoder != encoder_inference_duration.end(); encoder++) {
        total_duration += encoder->second;
    }

    return total_duration / 1000.0f;
}
}  // namespace genai
}  // namespace ov