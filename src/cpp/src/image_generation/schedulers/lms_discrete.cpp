// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/schedulers/lms_discrete.hpp"

#include <cmath>
#include <fstream>

#include "json_utils.hpp"
namespace {

// https://gist.github.com/lorenzoriano/5414671
template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N - 1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

// adaptive trapezoidal integral function
template <class F, class Real>
Real trapezoidal(F f, Real a, Real b, Real tol = 1e-6, int max_refinements = 100) {
    Real h = (b - a) / 2.0;
    Real ya = f(a);
    Real yb = f(b);
    Real I0 = (ya + yb) * h;

    for (int k = 1; k <= max_refinements; ++k) {
        Real sum = 0.0;
        for (int j = 1; j <= (1 << (k - 1)); ++j) {
            sum += f(a + (2 * j - 1) * h);
        }

        Real I1 = 0.5 * I0 + h * sum;
        if (k > 1 && std::abs(I1 - I0) < tol) {
            return I1;
        }

        I0 = I1;
        h /= 2.0;
    }

    // If the desired accuracy is not achieved, return the best estimate
    return I0;
}

float lms_derivative(float tau, int32_t order, int32_t curr_order, const std::vector<float>& sigma_vec, int32_t t) {
    float prod = 1.0f;

    for (int32_t k = 0; k < order; k++) {
        if (curr_order == k) {
            continue;
        }
        prod *= (tau - sigma_vec[t - k]) / (sigma_vec[t - curr_order] - sigma_vec[t - k]);
    }
    return prod;
}

}  // namespace


namespace ov {
namespace genai {

int64_t LMSDiscreteScheduler::_sigma_to_t(float sigma) const {
    double log_sigma = std::log(sigma);
    std::vector<float> dists(1000);
    for (int32_t i = 0; i < 1000; i++) {
        if (log_sigma - m_log_sigmas[i] >= 0)
            dists[i] = 1;
        else
            dists[i] = 0;
        if (i == 0)
            continue;
        dists[i] += dists[i - 1];
    }

    // get sigmas range
    int32_t low_idx = std::min(int(std::max_element(dists.begin(), dists.end()) - dists.begin()), 1000 - 2);
    int32_t high_idx = low_idx + 1;
    float low = m_log_sigmas[low_idx];
    float high = m_log_sigmas[high_idx];
    // interpolate sigmas
    double w = (low - log_sigma) / (low - high);
    w = std::max(0.0, std::min(1.0, w));
    int64_t timestep = std::llround((1 - w) * low_idx + w * high_idx);

    return timestep;
}

LMSDiscreteScheduler::Config::Config(const std::filesystem::path& scheduler_config_path) {
    std::ifstream file(scheduler_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", scheduler_config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "beta_start", beta_start);
    read_json_param(data, "beta_end", beta_end);
    read_json_param(data, "trained_betas", trained_betas);
    read_json_param(data, "beta_schedule", beta_schedule);
    read_json_param(data, "prediction_type", prediction_type);
    read_json_param(data, "num_train_timesteps", num_train_timesteps);
    read_json_param(data, "steps_offset", steps_offset);
}

LMSDiscreteScheduler::LMSDiscreteScheduler(const std::filesystem::path& scheduler_config_path)
    : LMSDiscreteScheduler(Config(scheduler_config_path)) {
}

LMSDiscreteScheduler::LMSDiscreteScheduler(const Config& scheduler_config)
    : m_config(scheduler_config) {
    if (!m_config.trained_betas.empty()) {
        m_betas = m_config.trained_betas;
    } else if (m_config.beta_schedule == BetaSchedule::LINEAR) {
        m_betas = linspace<float>(m_config.beta_start, m_config.beta_end, m_config.num_train_timesteps);
    } else if (m_config.beta_schedule == BetaSchedule::SCALED_LINEAR) {
        float start = std::sqrt(m_config.beta_start);
        float end = std::sqrt(m_config.beta_end);
        m_betas = linspace<float>(start, end, m_config.num_train_timesteps);
        std::for_each(m_betas.begin(), m_betas.end(), [] (float & x) { x *= x; });
    } else {
        OPENVINO_THROW("'beta_schedule' must be one of 'EPSILON' or 'SCALED_LINEAR'. Please, add support of other types");
    }

    // generates alphas
    std::transform(m_betas.begin(), m_betas.end(), std::back_inserter(m_alphas), [] (float b) { return 1.0f - b; });

    std::vector<float> log_sigma_vec;
    for (size_t i = 1; i <= m_alphas.size(); i++) {
        float alphas_cumprod =
            std::accumulate(m_alphas.begin(), m_alphas.begin() + i, 1.0f, std::multiplies<float>{});
        float sigma = std::sqrt((1 - alphas_cumprod) / alphas_cumprod);
        m_log_sigmas.push_back(std::log(sigma));
    }
}

float LMSDiscreteScheduler::get_init_noise_sigma() const {
    float max_sigma = *std::max_element(m_sigmas.begin(), m_sigmas.end());

    if (m_config.timestep_spacing == TimestepSpacing::LINSPACE ||
        m_config.timestep_spacing == TimestepSpacing::TRAILING) {
        return max_sigma;
    }

    return std::sqrt(max_sigma * max_sigma + 1);
}

void LMSDiscreteScheduler::scale_model_input(ov::Tensor sample, size_t inference_step) {
    const double scale = 1.0 / std::sqrt((m_sigmas[inference_step] * m_sigmas[inference_step] + 1));

    float* sample_data = sample.data<float>();
    for (size_t i = 0; i < sample.get_size(); i++) {
        sample_data[i] *= scale;
    }
}

void LMSDiscreteScheduler::set_timesteps(size_t num_inference_steps, float strength) {
    m_timesteps.clear();
    m_derivative_list.clear();

    float delta = -999.0f / (num_inference_steps - 1);
    // transform interpolation to time range
    for (size_t i = 0; i < num_inference_steps; i++) {
        float t = 999.0 + i * delta;
        int32_t low_idx = std::floor(t);
        int32_t high_idx = std::ceil(t);
        float w = t - low_idx;
        float sigma = std::exp((1 - w) * m_log_sigmas[low_idx] + w * m_log_sigmas[high_idx]);
        m_sigmas.push_back(sigma);
    }

    m_sigmas.push_back(0.f);

    // initialize timesteps
    for (size_t i = 0; i < num_inference_steps; ++i) {
        int64_t timestep = _sigma_to_t(m_sigmas[i]);
        m_timesteps.push_back(timestep);
    }
}

std::vector<int64_t> LMSDiscreteScheduler::get_timesteps() const {
    OPENVINO_ASSERT(!m_timesteps.empty(), "'timesteps' have not yet been set.");

    return m_timesteps;
}

std::map<std::string, ov::Tensor> LMSDiscreteScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) {
    const float sigma = m_sigmas[inference_step];

    // LMS step function:
    std::vector<float> derivative;
    derivative.reserve(latents.get_size());

    for (size_t j = 0; j < latents.get_size(); j++) {
        // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        float pred_latent = 0;
        switch (m_config.prediction_type) {
            case PredictionType::EPSILON:
                pred_latent = latents.data<float>()[j] - sigma * noise_pred.data<float>()[j];
                break;
            case PredictionType::SAMPLE:
                pred_latent = noise_pred.data<float>()[j];
                break;
            case PredictionType::V_PREDICTION:
                // pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
                pred_latent = noise_pred.data<float>()[j] * (-sigma / std::sqrt(sigma * sigma + 1.0f) +
                    latents.data<float>()[j] / (sigma * sigma + 1.0f));
                break;
            default:
                OPENVINO_THROW("Unsupported value for 'PredictionType'");
        }
        // 2. Convert to an ODE derivative
        derivative.push_back((latents.data<float>()[j] - pred_latent) / sigma);
    }

    m_derivative_list.push_back(derivative);
    // keep the list size within 4
    size_t order = 4;
    if (order < m_derivative_list.size()) {
        m_derivative_list.pop_front();
    }

    // 3. Compute linear multistep coefficients
    order = std::min(inference_step + 1, order);

    std::vector<float> lms_coeffs(order);
    for (size_t curr_order = 0; curr_order < order; curr_order++) {
        auto lms_derivative_functor = [order, curr_order, sigmas = this->m_sigmas, inference_step] (float tau) {
            return lms_derivative(tau, order, curr_order, sigmas, inference_step);
        };
        // integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]
        lms_coeffs[curr_order] = trapezoidal(lms_derivative_functor, static_cast<double>(sigma), static_cast<double>(m_sigmas[inference_step + 1]), 1e-4);
    }

    // 4. Compute previous sample based on the derivatives path
    // prev_sample = sample + sum(coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives)))
    ov::Tensor prev_sample(latents.get_element_type(), latents.get_shape());
    float* prev_sample_data = prev_sample.data<float>();
    const float* latents_data = latents.data<const float>();
    for (size_t i = 0; i < prev_sample.get_size(); ++i) {
        float derivative_sum = 0.0f;
        auto derivative_it = m_derivative_list.begin();
        for (size_t curr_order = 0; curr_order < order; derivative_it++, curr_order++) {
            derivative_sum += (*derivative_it)[i] * lms_coeffs[order - curr_order - 1];
        }
        prev_sample_data[i] = latents_data[i] + derivative_sum;
    }

    std::map<std::string, ov::Tensor> result{{"latent", prev_sample}};

    return result;
}

void LMSDiscreteScheduler::add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const {
    // use https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_ddim.py#L474
    OPENVINO_THROW("Not implemented");
}

} // namespace genai
} // namespace ov
