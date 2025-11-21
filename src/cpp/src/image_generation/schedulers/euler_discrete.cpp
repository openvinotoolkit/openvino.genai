// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/schedulers/euler_discrete.hpp"

#include <cassert>
#include <fstream>
#include <iterator>
#include <random>

#include "image_generation/numpy_utils.hpp"
#include "json_utils.hpp"

namespace ov {
namespace genai {

EulerDiscreteScheduler::Config::Config(const std::filesystem::path& scheduler_config_path) {
    std::ifstream file(scheduler_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", scheduler_config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "num_train_timesteps", num_train_timesteps);
    read_json_param(data, "beta_start", beta_start);
    read_json_param(data, "beta_end", beta_end);
    read_json_param(data, "beta_schedule", beta_schedule);
    read_json_param(data, "trained_betas", trained_betas);
    read_json_param(data, "final_sigmas_type", final_sigmas_type);
    read_json_param(data, "interpolation_type", interpolation_type);
    read_json_param(data, "sigma_max", sigma_max);
    read_json_param(data, "sigma_min", sigma_min);
    read_json_param(data, "steps_offset", steps_offset);
    read_json_param(data, "prediction_type", prediction_type);
    read_json_param(data, "timestep_spacing", timestep_spacing);
    read_json_param(data, "timestep_type", timestep_type);
    read_json_param(data, "rescale_betas_zero_snr", rescale_betas_zero_snr);
    read_json_param(data, "use_karras_sigmas", use_karras_sigmas);
    read_json_param(data, "use_exponential_sigmas", use_exponential_sigmas);
    read_json_param(data, "use_beta_sigmas", use_beta_sigmas);
}

EulerDiscreteScheduler::EulerDiscreteScheduler(const std::filesystem::path& scheduler_config_path)
    : EulerDiscreteScheduler(Config(scheduler_config_path)) {}

EulerDiscreteScheduler::EulerDiscreteScheduler(const Config& scheduler_config) : 
    m_config(scheduler_config), 
    m_num_inference_steps(0),
    m_step_index(-1),
    m_begin_index(-1) {
    std::vector<float> alphas, betas;

    using numpy_utils::linspace;

    if (!m_config.trained_betas.empty()) {
        betas = m_config.trained_betas;
    } else if (m_config.beta_schedule == BetaSchedule::LINEAR) {
        betas = linspace<float>(m_config.beta_start, m_config.beta_end, m_config.num_train_timesteps);
    } else if (m_config.beta_schedule == BetaSchedule::SCALED_LINEAR) {
        float start = std::sqrt(m_config.beta_start);
        float end = std::sqrt(m_config.beta_end);
        betas = linspace<float>(start, end, m_config.num_train_timesteps);
        std::for_each(betas.begin(), betas.end(), [](float& x) {
            x *= x;
        });
    } else {
        OPENVINO_THROW(
            "'beta_schedule' must be one of 'LINEAR' or 'SCALED_LINEAR'. Please, add support of other types");
    }

    if (m_config.rescale_betas_zero_snr) {
        using numpy_utils::rescale_zero_terminal_snr;
        rescale_zero_terminal_snr(betas);
    }

    std::transform(betas.begin(), betas.end(), std::back_inserter(alphas), [](float b) {
        return 1.0f - b;
    });

    for (size_t i = 1; i <= alphas.size(); ++i) {
        float alpha_cumprod =
            std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
        m_alphas_cumprod.push_back(alpha_cumprod);
    }

    if (m_config.rescale_betas_zero_snr) {
        m_alphas_cumprod.back() = std::pow(2, -24);
    }

    for (auto it = m_alphas_cumprod.rbegin(); it != m_alphas_cumprod.rend(); ++it) {
        float sigma = std::pow(((1 - (*it)) / (*it)), 0.5);
        m_sigmas.push_back(sigma);
    }

    auto linspaced =
        linspace<float>(0.0f, static_cast<float>(m_config.num_train_timesteps - 1), m_config.num_train_timesteps, true);
    for (auto it = linspaced.rbegin(); it != linspaced.rend(); ++it) {
        m_timesteps.push_back(static_cast<int64_t>(std::round(*it)));
    }

    OPENVINO_ASSERT(
        m_config.timestep_type != TimestepType::CONTINUOUS || m_config.prediction_type != PredictionType::V_PREDICTION,
        "This case isn't supported: `timestep_type=continuous` and `prediction_type=v_prediction`. Please, add "
        "support.");

    m_sigmas.push_back(0);
}

void EulerDiscreteScheduler::set_timesteps(size_t num_inference_steps, float strength) {
    // TODO: support `timesteps` and `sigmas` inputs
    m_timesteps.clear();
    m_sigmas.clear();
    m_step_index = m_begin_index = -1;

    m_num_inference_steps = num_inference_steps;
    std::vector<float> sigmas;

    OPENVINO_ASSERT(
        m_config.timestep_type != TimestepType::CONTINUOUS || m_config.prediction_type != PredictionType::V_PREDICTION,
        "This case isn't supported: `timestep_type=continuous` and `prediction_type=v_prediction`. Please, add "
        "support.");

    switch (m_config.timestep_spacing) {
    case TimestepSpacing::LINSPACE: {
        using numpy_utils::linspace;
        float end = static_cast<float>(m_config.num_train_timesteps - 1);
        auto linspaced = linspace<float>(0.0f, end, num_inference_steps, true);
        for (auto it = linspaced.rbegin(); it != linspaced.rend(); ++it) {
            m_timesteps.push_back(static_cast<int64_t>(std::round(*it)));
        }
        break;
    }
    case TimestepSpacing::LEADING: {
        size_t step_ratio = m_config.num_train_timesteps / m_num_inference_steps;
        for (size_t i = num_inference_steps - 1; i != -1; --i) {
            m_timesteps.push_back(i * step_ratio + m_config.steps_offset);
        }
        break;
    }
    case TimestepSpacing::TRAILING: {
        float step_ratio = static_cast<float>(m_config.num_train_timesteps) / static_cast<float>(m_num_inference_steps);
        for (float i = m_config.num_train_timesteps; i > 0; i -= step_ratio) {
            m_timesteps.push_back(static_cast<int64_t>(std::round(i)) - 1);
        }
        break;
    }
    default:
        OPENVINO_THROW("Unsupported value for 'timestep_spacing'");
    }

    for (const float& i : m_alphas_cumprod) {
        float sigma = std::pow(((1 - i) / i), 0.5);
        sigmas.push_back(sigma);
    }

    switch (m_config.interpolation_type) {
        case InterpolationType::LINEAR: {
            using numpy_utils::interp;

            std::vector<size_t> x_data_points(sigmas.size());
            std::iota(x_data_points.begin(), x_data_points.end(), 0);
            m_sigmas = interp(m_timesteps, x_data_points, sigmas);
            break;
        }
        case InterpolationType::LOG_LINEAR: {
            using numpy_utils::linspace;

            m_sigmas = linspace<float>(std::log(sigmas.back()), std::log(sigmas[0]), num_inference_steps + 1, true);
            std::transform(m_sigmas.begin(), m_sigmas.end(), m_sigmas.begin(), [](float x) {
                return std::exp(x);
            });
            break;
        }
        default:
            OPENVINO_THROW("Unsupported value for 'interpolation_type'");
    }

    OPENVINO_ASSERT(!m_config.use_karras_sigmas,
                    "Parameter 'use_karras_sigmas' is not supported. Please, add support.");

    OPENVINO_ASSERT(!m_config.use_exponential_sigmas,
                    "Parameter 'use_exponential_sigmas' is not supported. Please, add support.");

    OPENVINO_ASSERT(!m_config.use_beta_sigmas, "Parameter 'use_beta_sigmas' is not supported. Please, add support.");

    float sigma_last = 0;
    switch (m_config.final_sigmas_type) {
    case FinalSigmaType::SIGMA_MIN:
        sigma_last = std::pow(((1 - m_alphas_cumprod[0]) / m_alphas_cumprod[0]), 0.5);
        break;
    case FinalSigmaType::ZERO:
        break;
    default:
        OPENVINO_THROW("Unsupported value for 'final_sigmas_type'");
    }
    m_sigmas.push_back(sigma_last);

    // apply 'strength' used in image generation
    // in diffusers, it's https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L650
    {
        size_t init_timestep = std::min<size_t>(num_inference_steps * strength, num_inference_steps);
        size_t t_start = std::max<size_t>(num_inference_steps - init_timestep, 0);
        // keep original timesteps
        m_schedule_timesteps = m_timesteps;
        // while return patched ones by 'strength' parameter
        m_timesteps = std::vector<int64_t>(m_timesteps.begin() + t_start, m_timesteps.end());
        m_begin_index = t_start;

        OPENVINO_ASSERT(!m_timesteps.empty(),
                        "After adjusting the num_inference_steps by strength parameter: ", strength,
                        " the number of pipeline steps is less then 1 and not appropriate for this pipeline. Please set a different strength value.");
    }
}

std::map<std::string, ov::Tensor> EulerDiscreteScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) {
    // noise_pred - model_output
    // latents - sample
    // inference_step

    size_t timestep = m_timesteps[inference_step];

    if (m_step_index == -1)
        m_step_index = m_begin_index;

    float sigma = m_sigmas[m_step_index];
    // TODO: hardcoded gamma
    float gamma = 0.0f;
    float sigma_hat = sigma * (gamma + 1);

    float* model_output_data = noise_pred.data<float>();
    float* sample_data = latents.data<float>();

    ov::Tensor pred_original_sample(noise_pred.get_element_type(), noise_pred.get_shape());
    float* pred_original_sample_data = pred_original_sample.data<float>();

    ov::Tensor prev_sample(noise_pred.get_element_type(), noise_pred.get_shape());
    float* prev_sample_data = prev_sample.data<float>();

    // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    switch (m_config.prediction_type) {
    case PredictionType::EPSILON:
        for (size_t i = 0; i < noise_pred.get_size(); ++i) {
            pred_original_sample_data[i] = sample_data[i] - model_output_data[i] * sigma_hat;
        }
        break;
    case PredictionType::SAMPLE:
        for (size_t i = 0; i < noise_pred.get_size(); ++i) {
            pred_original_sample_data[i] = model_output_data[i];
        }
        break;
    case PredictionType::V_PREDICTION:
        for (size_t i = 0; i < noise_pred.get_size(); ++i) {
            pred_original_sample_data[i] = model_output_data[i] * (-sigma / std::pow((std::pow(sigma, 2) + 1), 0.5)) +
                                           (sample_data[i] / (std::pow(sigma, 2) + 1));
        }
        break;
    default:
        OPENVINO_THROW("Unsupported value for 'PredictionType'");
    }

    float dt = m_sigmas[m_step_index + 1] - sigma_hat;

    // 2. Convert to an ODE derivative
    for (size_t i = 0; i < prev_sample.get_size(); ++i) {
        prev_sample_data[i] = ((sample_data[i] - pred_original_sample_data[i]) / sigma_hat) * dt + sample_data[i];
    }

    m_step_index += 1;

    return {{"latent", prev_sample}, {"denoised", pred_original_sample}};
}

std::vector<int64_t> EulerDiscreteScheduler::get_timesteps() const {
    OPENVINO_ASSERT(!m_timesteps.empty(), "'timesteps' have not yet been set.");

    return m_timesteps;
}

float EulerDiscreteScheduler::get_init_noise_sigma() const {
    float max_sigma = *std::max_element(m_sigmas.begin(), m_sigmas.end());

    if (m_config.timestep_spacing == TimestepSpacing::LINSPACE ||
        m_config.timestep_spacing == TimestepSpacing::TRAILING) {
        return max_sigma;
    }

    return std::sqrt(max_sigma * max_sigma + 1);
}

void EulerDiscreteScheduler::scale_model_input(ov::Tensor sample, size_t inference_step) {
    if (m_step_index == -1)
        m_step_index = m_begin_index;

    float sigma = m_sigmas[m_step_index];
    float* sample_data = sample.data<float>();
    for (size_t i = 0; i < sample.get_size(); i++) {
        sample_data[i] /= std::pow((std::pow(sigma, 2) + 1), 0.5);
    }
}

size_t EulerDiscreteScheduler::_index_for_timestep(int64_t timestep) const {
    for (size_t i = 0; i < m_schedule_timesteps.size(); ++i) {
        if (timestep == m_schedule_timesteps[i]) {
            return i;
        }
    }

    OPENVINO_THROW("Failed to find index for timestep ", timestep);
}

void EulerDiscreteScheduler::add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const {
    const float sigma = m_sigmas[_index_for_timestep(latent_timestep)];

    float * init_latent_data = init_latent.data<float>();
    const float * noise_data = noise.data<float>();

    for (size_t i = 0; i < init_latent.get_size(); ++i) {
        init_latent_data[i] = init_latent_data[i] + sigma * noise_data[i];
    }
}

}  // namespace genai
}  // namespace ov
