// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <vector>
#include <string>
#include <map>
#include <openvino/runtime/tensor.hpp>
#include <openvino/runtime/compiled_model.hpp>

namespace ov::genai::module {

class UniPCMultistepScheduler {
public:
    enum class BetaSchedule {
        LINEAR,
        SCALED_LINEAR,
        SQUAREDCOS_CAP_V2
    };

    enum class PredictionType {
        EPSILON,
        SAMPLE,
        V_PREDICTION,
        FLOW_PREDICTION
    };

    enum class SolverType {
        BH1,
        BH2,
        MIDPOINT,
        HEUN,
        LOGRHO
    };

    enum class TimestepSpacing {
        LINSPACE,
        LEADING,
        TRAILING
    };

    enum class FinalSigmaType {
        ZERO,
        SIGMA_MIN
    };

    enum class TimeShiftType {
        EXPONENTIAL
    };

    struct Config {
        int num_train_timesteps = 1000;
        float beta_start = 0.0001f;
        float beta_end = 0.02f;
        BetaSchedule beta_schedule = BetaSchedule::LINEAR;
        std::optional<ov::Tensor> trained_betas = std::nullopt;
        int solver_order = 2;
        PredictionType prediction_type = PredictionType::EPSILON;
        bool thresholding = false;
        float dynamic_thresholding_ratio = 0.995f;
        float sample_max_value = 1.0f;
        bool predict_x0 = true;
        SolverType solver_type = SolverType::BH2;
        bool lower_order_final = true;
        std::vector<int> disable_corrector = {};
        bool use_karras_sigmas = false;
        bool use_exponential_sigmas = false;
        bool use_beta_sigmas = false;
        bool use_flow_sigmas = false;
        float flow_shift = 1.0f;
        TimestepSpacing timestep_spacing = TimestepSpacing::LINSPACE;
        int steps_offset = 0;
        FinalSigmaType final_sigma_type = FinalSigmaType::ZERO;
        bool rescale_betas_zero_snr = false;
        bool use_dynamic_shifting = false;
        TimeShiftType time_shift_type = TimeShiftType::EXPONENTIAL;
        std::optional<float> sigma_min = std::nullopt;
        std::optional<float> sigma_max = std::nullopt;

        Config() = default;
        explicit Config(const std::filesystem::path &config_path);
    };

    explicit UniPCMultistepScheduler(const std::filesystem::path &config_path, const std::string &device = "CPU");
    
    explicit UniPCMultistepScheduler(const Config &config, const std::string &device = "CPU");

    UniPCMultistepScheduler(const UniPCMultistepScheduler &) = delete;
    UniPCMultistepScheduler &operator=(const UniPCMultistepScheduler &) = delete;

    void set_timesteps(size_t num_inference_steps, std::optional<float> mu = std::nullopt);

    std::map<std::string, ov::Tensor> step(ov::Tensor &model_output, int64_t timestep, ov::Tensor sample);

    [[nodiscard]]
    const ov::Tensor &get_sigmas() const {
        return m_sigmas;
    }

    [[nodiscard]]
    const std::vector<int64_t> &get_timesteps() const {
        return m_timesteps;
    }

private:
    Config m_config;
    ov::InferRequest m_c_solver;
    ov::InferRequest m_p_solver;
    std::string m_device;
    ov::Tensor m_betas;
    ov::Tensor m_alphas;
    ov::Tensor m_alphas_cumprod;
    ov::Tensor m_alpha_t;
    ov::Tensor m_sigma_t;
    ov::Tensor m_lambda_t;
    ov::Tensor m_sigmas;
    float m_init_noise_sigma;
    std::vector<int64_t> m_timesteps;
    std::vector<std::optional<ov::Tensor>> m_model_outputs;
    std::vector<std::optional<int64_t>> m_timestep_list;
    std::optional<ov::Tensor> m_last_sample;
    std::optional<size_t> m_num_inference_steps = std::nullopt;
    std::optional<size_t> m_step_index = std::nullopt;
    std::optional<size_t> m_begin_index = std::nullopt;
    std::optional<size_t> m_lower_order_nums = std::nullopt;
    std::optional<int> m_this_order = std::nullopt;

    static BetaSchedule parse_beta_schedule(const std::string &value);
    static PredictionType parse_prediction_type(const std::string &value);
    static SolverType parse_solver_type(const std::string &value);
    static TimestepSpacing parse_timestep_spacing(const std::string &value);
    static FinalSigmaType parse_final_sigma_type(const std::string &value);
    static TimeShiftType parse_time_shift_type(const std::string &value);
    static std::string to_string(BetaSchedule schedule);
    static std::string to_string(PredictionType type);
    static std::string to_string(SolverType type);
    static std::string to_string(TimestepSpacing spacing);
    static std::string to_string(FinalSigmaType type);
    static std::string to_string(TimeShiftType type);
    void init_solver_models();
    ov::Tensor betas_for_alpha_bar(size_t num_diffusion_timesteps, float max_beta = 0.999f, const std::string &alpha_transform_type = "cosine");
    void rescale_zero_terminal_snr();
    float sigma_to_t(float sigma, const ov::Tensor &log_sigmas);
    std::vector<int64_t> sigma_to_t(const ov::Tensor &sigmas, const ov::Tensor &log_sigmas);
    std::pair<ov::Tensor, std::vector<int64_t>> convert_to_karras(ov::Tensor &sigmas, size_t num_inference_steps);
    std::pair<ov::Tensor, std::vector<int64_t>> convert_to_exponential(ov::Tensor &sigmas, size_t num_inference_steps);
    std::pair<ov::Tensor, std::vector<int64_t>> convert_to_flow(size_t num_inference_steps);
    std::pair<ov::Tensor, ov::Tensor> sigma_to_alpha_sigma_t(const ov::Tensor &sigma);
    std::pair<float, float> sigma_to_alpha_sigma_t(float sigma);
    size_t index_for_timestep(int64_t timestep, std::optional<std::vector<int64_t>> schedule_timesteps = std::nullopt);
    void init_step_index(int64_t timestep);
    ov::Tensor convert_model_output(const ov::Tensor &model_output, const ov::Tensor &sample);
    void thresholding_sample(ov::Tensor &sample);
    static float compute_quantile(std::vector<float> &data, float quantile);
    static ov::Tensor einsum_k_bkc_to_bc(const ov::Tensor &D1s, const ov::Tensor &rhos);
    ov::Tensor multistep_uni_c_bh_update(ov::Tensor &this_model_output, ov::Tensor &last_sample, ov::Tensor &this_sample, int order);
    ov::Tensor multistep_uni_p_bh_update(ov::Tensor &model_output, ov::Tensor &sample, int order);
};

}
