// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "unipc_multistep_scheduler.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <functional>
#include "json_utils.hpp"
#include "image_generation/numpy_utils.hpp"
#include "logger.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/inverse.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "utils.hpp"
#include "module_genai/utils/tensor_utils.hpp"


namespace ov::genai::module {

UniPCMultistepScheduler::Config::Config(const std::filesystem::path &config_path) {
    if (!std::filesystem::exists(config_path)) {
        OPENVINO_THROW("UniPCMultistepScheduler config file does not exist: " + config_path.string());
    }

    std::ifstream config_file(config_path);
    nlohmann::json parsed = nlohmann::json::parse(config_file);
    if (parsed.contains("num_train_timesteps")) {
        utils::read_json_param(parsed, "num_train_timesteps", num_train_timesteps);
    }
    if (parsed.contains("beta_start")) {
        utils::read_json_param(parsed, "beta_start", beta_start);
    }
    if (parsed.contains("beta_end")) {
        utils::read_json_param(parsed, "beta_end", beta_end);
    }
    if (parsed.contains("beta_schedule")) {
        std::string beta_schedule_str;
        utils::read_json_param(parsed, "beta_schedule", beta_schedule_str);
        beta_schedule = parse_beta_schedule(beta_schedule_str);
    }
    if (parsed.contains("trained_betas")) {
        std::vector<float> trained_betas_vec;
        utils::read_json_param(parsed, "trained_betas", trained_betas_vec);
        ov::Shape shape = {trained_betas_vec.size()};
        trained_betas = ov::Tensor(ov::element::f32, shape);
        std::memcpy(trained_betas->data<float>(), trained_betas_vec.data(), trained_betas_vec.size() * sizeof(float));
    }
    if (parsed.contains("solver_order")) {
        utils::read_json_param(parsed, "solver_order", solver_order);
    }
    if (parsed.contains("prediction_type")) {
        std::string prediction_type_str;
        utils::read_json_param(parsed, "prediction_type", prediction_type_str);
        prediction_type = parse_prediction_type(prediction_type_str);
    }
    if (parsed.contains("thresholding")) {
        utils::read_json_param(parsed, "thresholding", thresholding);
    }
    if (parsed.contains("dynamic_thresholding_ratio")) {
        utils::read_json_param(parsed, "dynamic_thresholding_ratio", dynamic_thresholding_ratio);
    }
    if (parsed.contains("sample_max_value")) {
        utils::read_json_param(parsed, "sample_max_value", sample_max_value);
    }
    if (parsed.contains("predict_x0")) {
        utils::read_json_param(parsed, "predict_x0", predict_x0);
    }
    if (parsed.contains("solver_type")) {
        std::string solver_type_str;
        utils::read_json_param(parsed, "solver_type", solver_type_str);
        solver_type = parse_solver_type(solver_type_str);
    }
    if (parsed.contains("lower_order_final")) {
        utils::read_json_param(parsed, "lower_order_final", lower_order_final);
    }
    if (parsed.contains("disable_corrector")) {
        utils::read_json_param(parsed, "disable_corrector", disable_corrector);
    }
    if (parsed.contains("use_karras_sigmas")) {
        utils::read_json_param(parsed, "use_karras_sigmas", use_karras_sigmas);
    }
    if (parsed.contains("use_exponential_sigmas")) {
        utils::read_json_param(parsed, "use_exponential_sigmas", use_exponential_sigmas);
    }
    if (parsed.contains("use_beta_sigmas")) {
        utils::read_json_param(parsed, "use_beta_sigmas", use_beta_sigmas);
    }
    if (parsed.contains("use_flow_sigmas")) {
        utils::read_json_param(parsed, "use_flow_sigmas", use_flow_sigmas);
    }
    if (parsed.contains("flow_shift")) {
        utils::read_json_param(parsed, "flow_shift", flow_shift);
    }
    if (parsed.contains("timestep_spacing")) {
        std::string timestep_spacing_str;
        utils::read_json_param(parsed, "timestep_spacing", timestep_spacing_str);
        timestep_spacing = parse_timestep_spacing(timestep_spacing_str);
    }
    if (parsed.contains("steps_offset")) {
        utils::read_json_param(parsed, "steps_offset", steps_offset);
    }
    if (parsed.contains("final_sigma_type")) {
        std::string final_sigma_type_str;
        utils::read_json_param(parsed, "final_sigma_type", final_sigma_type_str);
        final_sigma_type = parse_final_sigma_type(final_sigma_type_str);
    }
    if (parsed.contains("rescale_betas_zero_snr")) {
        utils::read_json_param(parsed, "rescale_betas_zero_snr", rescale_betas_zero_snr);
    }
    if (parsed.contains("use_dynamic_shifting")) {
        utils::read_json_param(parsed, "use_dynamic_shifting", use_dynamic_shifting);
    }
    if (parsed.contains("time_shift_type")) {
        std::string time_shift_type_str;
        utils::read_json_param(parsed, "time_shift_type", time_shift_type_str);
        time_shift_type = parse_time_shift_type(time_shift_type_str);
    }
    if (parsed.contains("sigma_min")) {
        utils::read_json_param(parsed, "sigma_min", sigma_min);
    }
    if (parsed.contains("sigma_max")) {
        utils::read_json_param(parsed, "sigma_max", sigma_max);
    }
}

UniPCMultistepScheduler::UniPCMultistepScheduler(const std::filesystem::path &config_path, const std::string &device)
    : UniPCMultistepScheduler(Config(config_path), device) {}

UniPCMultistepScheduler::UniPCMultistepScheduler(const Config &config, const std::string &device)
    : m_config(config), m_device(device) {
    int sigmas_count = 0;
    if (m_config.use_beta_sigmas) {
        sigmas_count++;
    }
    if (m_config.use_exponential_sigmas) {
        sigmas_count++;
    }
    if (config.use_karras_sigmas) {
        sigmas_count++;
    }
    if (sigmas_count > 1) {
        OPENVINO_THROW("Only one of 'use_beta_sigmas', 'use_exponential_sigmas' or 'use_karras_sigmas' can be true");
    }

    if (m_config.trained_betas.has_value()) {
        m_config.trained_betas.value().copy_to(m_betas);
    } else if (m_config.beta_schedule == BetaSchedule::LINEAR) {
        auto betas = numpy_utils::linspace<float>(
            m_config.beta_start, 
            m_config.beta_end, 
            static_cast<size_t>(m_config.num_train_timesteps), 
            true);
        m_betas = ov::Tensor(ov::element::f32, {betas.size()});
        std::memcpy(m_betas.data<float>(), betas.data(), betas.size() * sizeof(float));
    } else if (m_config.beta_schedule == BetaSchedule::SCALED_LINEAR) {
        auto betas = numpy_utils::linspace<float>(
            std::sqrt(m_config.beta_start), 
            std::sqrt(m_config.beta_end), 
            static_cast<size_t>(m_config.num_train_timesteps), 
            true);
        m_betas = ov::Tensor(ov::element::f32, {betas.size()});
        auto betas_data = m_betas.data<float>();
        for (size_t i = 0; i < betas.size(); ++i) {
            betas_data[i] = betas[i] * betas[i];
        }
    } else if (m_config.beta_schedule == BetaSchedule::SQUAREDCOS_CAP_V2) {
        m_betas = betas_for_alpha_bar(static_cast<size_t>(m_config.num_train_timesteps));
    } else {
        OPENVINO_THROW("Unsupported beta_schedule: " + to_string(m_config.beta_schedule));
    }

    if (m_config.rescale_betas_zero_snr) {
        rescale_zero_terminal_snr();
    }

    m_alphas = ov::Tensor(m_betas.get_element_type(), m_betas.get_shape());
    m_alphas_cumprod = ov::Tensor(m_betas.get_element_type(), m_betas.get_shape());
    m_alpha_t = ov::Tensor(m_betas.get_element_type(), m_betas.get_shape());
    m_sigma_t = ov::Tensor(m_betas.get_element_type(), m_betas.get_shape());
    m_lambda_t = ov::Tensor(m_betas.get_element_type(), m_betas.get_shape());
    m_sigmas = ov::Tensor(m_betas.get_element_type(), m_betas.get_shape());
    auto betas_data = m_betas.data<const float>();
    auto alphas_data = m_alphas.data<float>();
    auto alphas_cumprod_data = m_alphas_cumprod.data<float>();
    auto alpha_t_data = m_alpha_t.data<float>();
    auto sigma_t_data = m_sigma_t.data<float>();
    auto lambda_t_data = m_lambda_t.data<float>();
    auto sigmas_data = m_sigmas.data<float>();

    for (size_t i = 0; i < m_betas.get_size(); i++) {
        alphas_data[i] = 1.0f - betas_data[i];
        if (i == 0) {
            alphas_cumprod_data[i] = alphas_data[i];
        } else {
            alphas_cumprod_data[i] = alphas_cumprod_data[i - 1] * alphas_data[i];
        }
        if (i == (m_betas.get_size() - 1) && m_config.rescale_betas_zero_snr) {
            alphas_cumprod_data[i] = static_cast<float>(std::pow(2, -24));
        }
        alpha_t_data[i] = std::sqrt(alphas_cumprod_data[i]);
        sigma_t_data[i] = std::sqrt(1.0f - alphas_cumprod_data[i]);
        lambda_t_data[i] = std::log(alpha_t_data[i]) - std::log(sigma_t_data[i]);
        sigmas_data[i] = std::sqrt((1.0f - alphas_cumprod_data[i]) / alphas_cumprod_data[i]);
    }

    m_init_noise_sigma = 1.0f;

    if (m_config.solver_type != SolverType::BH1 && m_config.solver_type != SolverType::BH2) {
        GENAI_INFO("Using bh2 solver");
        m_config.solver_type = SolverType::BH2;
    }

    m_timesteps = numpy_utils::linspace<int64_t>(
        0, 
        m_config.num_train_timesteps - 1, 
        static_cast<size_t>(m_config.num_train_timesteps), 
        true);
    std::reverse(m_timesteps.begin(), m_timesteps.end());

    m_model_outputs.resize(m_config.solver_order, std::nullopt);
    m_timestep_list.resize(m_config.solver_order, std::nullopt);
    m_lower_order_nums = 0;

    init_solver_models();
}

void UniPCMultistepScheduler::init_solver_models() {
    auto c_input_R = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
    auto c_input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    
    auto c_axis_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
    auto c_b_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(c_input_b, c_axis_1);
    auto c_R_inv = std::make_shared<ov::op::v14::Inverse>(c_input_R, false);
    auto c_x = std::make_shared<ov::op::v0::MatMul>(c_R_inv, c_b_unsqueezed);
    auto c_x_final = std::make_shared<ov::op::v0::Squeeze>(c_x, c_axis_1);
    auto c_result = std::make_shared<ov::op::v0::Result>(c_x_final);
    auto c_model = std::make_shared<ov::Model>(ov::ResultVector{c_result}, ov::ParameterVector{c_input_R, c_input_b});
    ov::CompiledModel c_solver_model = ov::genai::utils::singleton_core().compile_model(
        c_model, m_device);
    m_c_solver = c_solver_model.create_infer_request();

    auto p_input_R = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
    auto p_input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});

    auto start_R = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 0});
    auto stop_R = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, -1});
    auto step_R = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 1});
    
    std::vector<int64_t> begin_mask = {0, 0};
    std::vector<int64_t> end_mask = {0, 0};
    auto p_R_sliced = std::make_shared<ov::op::v1::StridedSlice>(
        p_input_R, start_R, stop_R, step_R, begin_mask, end_mask);
    
    auto start_b = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto stop_b = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    auto step_b = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    std::vector<int64_t> begin_mask_b = {0};
    std::vector<int64_t> end_mask_b = {0};

    auto p_b_sliced = std::make_shared<ov::op::v1::StridedSlice>(
        p_input_b, start_b, stop_b, step_b, begin_mask_b, end_mask_b);
    
    auto p_axis_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
    auto p_b_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(p_b_sliced, p_axis_1);
    auto p_R_inv = std::make_shared<ov::op::v14::Inverse>(p_R_sliced, false);
    auto p_x = std::make_shared<ov::op::v0::MatMul>(p_R_inv, p_b_unsqueezed);
    auto p_x_final = std::make_shared<ov::op::v0::Squeeze>(p_x, p_axis_1);
    auto p_result = std::make_shared<ov::op::v0::Result>(p_x_final);
    auto p_model = std::make_shared<ov::Model>(ov::ResultVector{p_result}, ov::ParameterVector{p_input_R, p_input_b});
    ov::CompiledModel p_solver_model = ov::genai::utils::singleton_core().compile_model(
        p_model, m_device);
    m_p_solver = p_solver_model.create_infer_request();
}

void UniPCMultistepScheduler::set_timesteps(size_t num_inference_steps, std::optional<float> mu) {
    if (mu.has_value()) {
        OPENVINO_ASSERT(m_config.use_dynamic_shifting && m_config.time_shift_type == TimeShiftType::EXPONENTIAL);
        m_config.flow_shift = std::exp(mu.value());
    }

    std::vector<int64_t> timesteps;
    if (m_config.timestep_spacing == TimestepSpacing::LINSPACE) {
        timesteps.clear();
        auto timesteps_float = numpy_utils::linspace<float>(
            0, 
            m_config.num_train_timesteps - 1, 
            static_cast<size_t>(m_config.num_train_timesteps), 
            true);
        std::transform(timesteps_float.begin(), timesteps_float.end(), std::back_inserter(timesteps),
                       [](float t) { return static_cast<int64_t>(std::round(t)); });
        std::reverse(timesteps.begin(), timesteps.end());
        timesteps.pop_back();
    } else if (m_config.timestep_spacing == TimestepSpacing::LEADING) {
        float step_ratio = static_cast<float>(m_config.num_train_timesteps) / static_cast<float>(num_inference_steps + 1);
        timesteps.resize(num_inference_steps + 1);
        timesteps.clear();

        for (size_t i = num_inference_steps; i >= 0; i--) {
            timesteps[i] = static_cast<int64_t>(std::round(static_cast<float>(i) * step_ratio) + static_cast<float>(m_config.steps_offset));
        }
        timesteps.pop_back();
    } else if (m_config.timestep_spacing == TimestepSpacing::TRAILING) {
        float step_ratio = static_cast<float>(m_config.num_train_timesteps) / static_cast<float>(num_inference_steps);
        timesteps.clear();
        for (size_t i = 0; i < num_inference_steps; i++) {
            timesteps.push_back(static_cast<int64_t>(std::round(static_cast<float>(m_config.num_train_timesteps) - static_cast<float>(i) * step_ratio) - 1));
        }
    } else {
        OPENVINO_THROW("Unsupported timestep spacing: " + to_string(m_config.timestep_spacing));
    }

    ov::Tensor sigmas(m_betas.get_element_type(), m_betas.get_shape());
    auto sigmas_data = sigmas.data<float>();
    auto alphas_cumprod_data = m_alphas_cumprod.data<const float>();
    for (size_t i = 0; i < m_betas.get_size(); i++) {
        sigmas_data[i] = std::sqrt((1.0f - alphas_cumprod_data[i]) / alphas_cumprod_data[i]);
    }

    if (m_config.use_karras_sigmas) {
        std::tie(m_sigmas, m_timesteps) = convert_to_karras(sigmas, num_inference_steps);
    } else if (m_config.use_exponential_sigmas) {
        // TODO: implement exponential sigmas conversion
    } else if (m_config.use_beta_sigmas) {
        // TODO: implement beta sigmas conversion
    } else if (m_config.use_flow_sigmas) {
        std::tie(m_sigmas, m_timesteps) = convert_to_flow(num_inference_steps);
    } else {
        // TODO: implement default sigmas conversion
    }

    m_num_inference_steps = m_timesteps.size();
    m_model_outputs.clear();
    m_model_outputs.resize(m_config.solver_order, std::nullopt);
    m_timestep_list.clear();
    m_timestep_list.resize(m_config.solver_order, std::nullopt);
    m_last_sample = std::nullopt;
    m_lower_order_nums = 0;
    m_step_index = std::nullopt;
    m_begin_index = std::nullopt;
}

UniPCMultistepScheduler::BetaSchedule UniPCMultistepScheduler::parse_beta_schedule(const std::string &value) {
    if (value == "linear") {
        return BetaSchedule::LINEAR;
    } else if (value == "scaled_linear") {
        return BetaSchedule::SCALED_LINEAR;
    } else if (value == "squaredcos_cap_v2") {
        return BetaSchedule::SQUAREDCOS_CAP_V2;
    } else {
        OPENVINO_THROW("Unsupported beta_schedule value: " + value);
    }
}

UniPCMultistepScheduler::PredictionType UniPCMultistepScheduler::parse_prediction_type(const std::string &value) {
    if (value == "epsilon") {
        return PredictionType::EPSILON;
    } else if (value == "sample") {
        return PredictionType::SAMPLE;
    } else if (value == "v_prediction") {
        return PredictionType::V_PREDICTION;
    } else if (value == "flow_prediction") {
        return PredictionType::FLOW_PREDICTION;
    } else {
        OPENVINO_THROW("Unsupported prediction_type value: " + value);
    }
}

UniPCMultistepScheduler::SolverType UniPCMultistepScheduler::parse_solver_type(const std::string &value) {
    if (value == "bh1") {
        return SolverType::BH1;
    } else if (value == "bh2") {
        return SolverType::BH2;
    } else if (value == "midpoint") {
        return SolverType::MIDPOINT;
    } else if (value == "heun") {
        return SolverType::HEUN;
    } else if (value == "logrho") {
        return SolverType::LOGRHO;
    } else {
        OPENVINO_THROW("Unsupported solver_type value: " + value);
    }
}

UniPCMultistepScheduler::TimestepSpacing UniPCMultistepScheduler::parse_timestep_spacing(const std::string &value) {
    if (value == "linspace") {
        return TimestepSpacing::LINSPACE;
    } else if (value == "leading") {
        return TimestepSpacing::LEADING;
    } else if (value == "trailing") {
        return TimestepSpacing::TRAILING;
    } else {
        OPENVINO_THROW("Unsupported timestep_spacing value: " + value);
    }
}

UniPCMultistepScheduler::FinalSigmaType UniPCMultistepScheduler::parse_final_sigma_type(const std::string &value) {
    if (value == "zero") {
        return FinalSigmaType::ZERO;
    } else if (value == "sigma_min") {
        return FinalSigmaType::SIGMA_MIN;
    } else {
        OPENVINO_THROW("Unsupported final_sigma_type value: " + value);
    }
}

UniPCMultistepScheduler::TimeShiftType UniPCMultistepScheduler::parse_time_shift_type(const std::string &value) {
    if (value == "exponential") {
        return TimeShiftType::EXPONENTIAL;
    } else {
        OPENVINO_THROW("Unsupported time_shift_type value: " + value);
    }
}

std::string UniPCMultistepScheduler::to_string(BetaSchedule schedule) {
    switch (schedule) {
        case BetaSchedule::LINEAR:
            return "linear";
        case BetaSchedule::SCALED_LINEAR:
            return "scaled_linear";
        case BetaSchedule::SQUAREDCOS_CAP_V2:
            return "squaredcos_cap_v2";
        default:
            return "unknown";
    }
}

std::string UniPCMultistepScheduler::to_string(PredictionType type) {
    switch (type) {
        case PredictionType::EPSILON:
            return "epsilon";
        case PredictionType::SAMPLE:
            return "sample";
        case PredictionType::V_PREDICTION:
            return "v_prediction";
        case PredictionType::FLOW_PREDICTION:
            return "flow_prediction";
        default:
            return "unknown";
    }
}

std::string UniPCMultistepScheduler::to_string(SolverType type) {
    switch (type) {
        case SolverType::BH1:
            return "bh1";
        case SolverType::BH2:
            return "bh2";
        case SolverType::MIDPOINT:
            return "midpoint";
        case SolverType::HEUN:
            return "heun";
        case SolverType::LOGRHO:
            return "logrho";
        default:
            return "unknown";
    }
}

std::string UniPCMultistepScheduler::to_string(TimestepSpacing spacing) {
    switch (spacing) {
        case TimestepSpacing::LINSPACE:
            return "linspace";
        case TimestepSpacing::LEADING:
            return "leading";
        case TimestepSpacing::TRAILING:
            return "trailing";
        default:
            return "unknown";
    }
}

std::string UniPCMultistepScheduler::to_string(FinalSigmaType type) {
    switch (type) {
        case FinalSigmaType::ZERO:
            return "zero";
        case FinalSigmaType::SIGMA_MIN:
            return "sigma_min";
        default:
            return "unknown";
    }
}

std::string UniPCMultistepScheduler::to_string(TimeShiftType type) {
    switch (type) {
        case TimeShiftType::EXPONENTIAL:
            return "exponential";
        default:
            return "unknown";
    }
}

ov::Tensor UniPCMultistepScheduler::betas_for_alpha_bar(size_t num_diffusion_timesteps, float max_beta, const std::string &alpha_transform_type) {
    std::function<float(float)> alpha_bar_fn;
    if (alpha_transform_type == "cosine") {
        alpha_bar_fn = [](float t) {
            return std::cos((t + 0.008f) / 1.008f * M_PI / 2.0f) * std::cos((t + 0.008f) / 1.008f * M_PI / 2.0f);
        };
    } else if (alpha_transform_type == "exp") {
        alpha_bar_fn = [](float t) {
            return std::exp(t * -12.0f);
        };
    } else {
        OPENVINO_THROW("Unsupported alpha_transform_type: " + alpha_transform_type);
    }

    ov::Tensor betas(ov::element::f32, {num_diffusion_timesteps});
    float * betas_data = betas.data<float>();
    for (size_t i = 0; i < num_diffusion_timesteps; i++) {
        float t1 = static_cast<float>(i) / static_cast<float>(num_diffusion_timesteps);
        float t2 = static_cast<float>(i + 1) / static_cast<float>(num_diffusion_timesteps);
        betas_data[i] = std::min(1.0f - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta);
    }
    return betas;
}

void UniPCMultistepScheduler::rescale_zero_terminal_snr() {
    ov::Tensor alphas(m_betas.get_element_type(), m_betas.get_shape());
    ov::Tensor alphas_cumprod(m_betas.get_element_type(), m_betas.get_shape());
    ov::Tensor alphas_bar_sqrt(m_betas.get_element_type(), m_betas.get_shape());
    auto alphas_data = alphas.data<float>();
    auto betas_data = m_betas.data<float>();
    auto alphas_cumprod_data = alphas_cumprod.data<float>();
    auto alphas_bar_sqrt_data = alphas_bar_sqrt.data<float>();

    for (size_t i = 0; i < m_betas.get_size(); i++) {
        alphas_data[i] = 1.0f - betas_data[i];
        if (i == 0) {
            alphas_cumprod_data[i] = alphas_data[i];
        } else {
            alphas_cumprod_data[i] = alphas_cumprod_data[i - 1] * alphas_data[i];
        }
        alphas_bar_sqrt_data[i] = std::sqrt(alphas_cumprod_data[i]);
    }

    float alphas_bar_sqrt_0 = alphas_bar_sqrt_data[0];
    float alphas_bar_sqrt_T = alphas_bar_sqrt_data[alphas_bar_sqrt.get_size() - 1];

    ov::Tensor alphas_bar(m_betas.get_element_type(), m_betas.get_shape());
    auto alphas_bar_data = alphas_bar.data<float>();
    for (size_t i = 0; i < m_betas.get_size(); i++) {
        alphas_bar_sqrt_data[i] -= alphas_bar_sqrt_T;
        alphas_bar_sqrt_data[i] *= (alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T));
        alphas_bar_data[i] = alphas_bar_sqrt_data[i] * alphas_bar_sqrt_data[i];

        if (i >= 1) {
            alphas_data[i] = alphas_bar_data[i] / alphas_bar_data[i - 1];
            betas_data[i] = 1.0f - alphas_data[i];
        } else {
            alphas_data[0] = alphas_bar_data[0];
            betas_data[0] = 1.0f - alphas_data[0];
        }
    }
}

float UniPCMultistepScheduler::sigma_to_t(float sigma, const ov::Tensor &log_sigmas) {
    auto log_sigmas_data = log_sigmas.data<const float>();
    float log_sigma = std::log(std::max(sigma, 1e-10f));
    ov::Tensor dists(ov::element::f32, log_sigmas.get_shape());
    auto dists_data = dists.data<float>();
    ov::Tensor cumsum(ov::element::i32, dists.get_shape());
    auto cumsum_data = cumsum.data<int32_t>();
    size_t low_index = 0;


    for (size_t i = 0; i < dists.get_shape()[0]; i++) {
        dists_data[i] = log_sigma - log_sigmas_data[i];
        if (dists_data[i] >= 0.0f) {
            if (i == 0) {
                cumsum_data[i] = 1;
            } else {
                cumsum_data[i] = 1 + cumsum_data[i - 1];
            }
        } else {
            if (i == 0) {
                cumsum_data[i] = 0;
            } else {
                cumsum_data[i] = cumsum_data[i - 1];
            }
        }
        if (cumsum_data[low_index] < cumsum_data[i]) {
            low_index = i;
        }
    }

    if (low_index > log_sigmas.get_size() - 2) {
        low_index = log_sigmas.get_size() - 2;
    }
    size_t high_index = low_index + 1;

    float low = log_sigmas_data[low_index];
    float high = log_sigmas_data[high_index];

    float w = (low - log_sigma) / (low - high);
    w = std::clamp(w, 0.0f, 1.0f);

    float t = (1 - w) * static_cast<float>(low_index) + w * static_cast<float>(high_index);

    return t;
}

std::vector<int64_t> UniPCMultistepScheduler::sigma_to_t(const ov::Tensor &sigmas, const ov::Tensor &log_sigmas) {
    auto sigmas_data = sigmas.data<const float>();
    auto log_sigmas_data = log_sigmas.data<const float>();
    ov::Tensor log_sigma(ov::element::f32, sigmas.get_shape());
    auto log_sigma_data = log_sigma.data<float>();
    for (size_t i = 0; i < sigmas.get_size(); i++) {
        log_sigma_data[i] = std::log(std::max(sigmas_data[i], 1e-10f));
    }

    ov::Tensor dists(ov::element::f32, {log_sigmas.get_shape()[0], log_sigma.get_shape()[0]});
    auto dists_data = dists.data<float>();
    ov::Tensor cumsum(ov::element::i32, dists.get_shape());
    auto cumsum_data = cumsum.data<int32_t>();
    std::vector<size_t> low_index(dists.get_shape()[1], 0);;
    for (size_t i = 0; i < dists.get_shape()[0]; i++) {
        for (size_t j = 0; j < dists.get_shape()[1]; j++) {
            dists_data[i * dists.get_shape()[1] + j] = log_sigma_data[j] - log_sigmas_data[i];

            if (dists_data[i * dists.get_shape()[1] + j] >= 0.0f) {
                if (i == 0) {
                    cumsum_data[i * dists.get_shape()[1] + j] = 1;
                } else {
                    cumsum_data[i * dists.get_shape()[1] + j] = 1 + cumsum_data[(i - 1) * dists.get_shape()[1] + j];
                }
            } else {
                if (i == 0) {
                    cumsum_data[i * dists.get_shape()[1] + j] = 0;
                } else {
                    cumsum_data[i * dists.get_shape()[1] + j] = cumsum_data[(i - 1) * dists.get_shape()[1] + j];
                }
            }
            if (cumsum_data[low_index[j] * dists.get_shape()[1] + j] < cumsum_data[i * dists.get_shape()[1] + j]) {
                low_index[j] = i;
            }
        }
    }

    std::vector<size_t> high_index(dists.get_shape()[1], 0);
    for (size_t i = 0; i < low_index.size(); i++) {
        if (low_index[i] > log_sigmas.get_size() - 2) {
            low_index[i] = log_sigmas.get_size() - 2;
        }
        high_index[i] = low_index[i] + 1;
    }

    std::vector<float> low(sigmas.get_size(), 0.0f);
    std::vector<float> high(sigmas.get_size(), 0.0f);
    for (size_t i = 0; i < low_index.size(); i++) {
        low[i] = log_sigmas_data[low_index[i]];
        high[i] = log_sigmas_data[high_index[i]];
    }

    std::vector<float> w(sigmas.get_size(), 0.0f);
    for (size_t i = 0; i < w.size(); i++) {
        w[i] = (low[i] - log_sigma_data[i]) / (low[i] - high[i]);
        w[i] = std::clamp(w[i], 0.0f, 1.0f);
    }

    std::vector<int64_t> t(sigmas.get_size(), 0);
    for (size_t i = 0; i < t.size(); i++) {
        t[i] = static_cast<int64_t>(
            std::round((1 - w[i]) * static_cast<float>(low_index[i]) + w[i] * static_cast<float>(high_index[i])));
    }

    return t;
}

std::pair<ov::Tensor, std::vector<int64_t>> UniPCMultistepScheduler::convert_to_karras(ov::Tensor &sigmas, size_t num_inference_steps) {
    auto sigmas_data = sigmas.data<float>();

    ov::Tensor log_sigmas(sigmas.get_element_type(), sigmas.get_shape());
    auto log_sigmas_data = log_sigmas.data<float>();
    for (size_t i = 0; i < sigmas.get_size(); i++) {
        log_sigmas_data[i] = std::log(sigmas_data[i]);
    }

    std::reverse(sigmas_data, sigmas_data + sigmas.get_size());

    float sigma_min = sigmas_data[sigmas.get_size() - 1];
    float sigma_max = sigmas_data[0];

    if (m_config.sigma_min.has_value()) {
        sigma_min = m_config.sigma_min.value();
    }
    if (m_config.sigma_max.has_value()) {
        sigma_max = m_config.sigma_max.value();
    }

    float rho = 7.0f;
    ov::Shape sigmas_shape = sigmas.get_shape();
    std::vector<int64_t> timesteps;
    timesteps.reserve(num_inference_steps);
    auto ramp = numpy_utils::linspace<float>(
        0.0f, 
        1.0f, 
        static_cast<size_t>(num_inference_steps), 
        true);
    ov::Tensor output_sigmas(sigmas.get_element_type(), {ramp.size()});
    auto output_sigmas_data = output_sigmas.data<float>();
    float min_inv_rho = std::pow(sigma_min, 1.0f / rho);
    float max_inv_rho = std::pow(sigma_max, 1.0f / rho);
    for (size_t i = 0; i < num_inference_steps; i++) {
        float inv_rho = max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho);
        output_sigmas_data[i] = std::pow(inv_rho, rho);
        if (m_config.use_flow_sigmas) {
            output_sigmas_data[i] = output_sigmas_data[i] / (1.0f + output_sigmas_data[i]);
            timesteps.push_back(static_cast<int64_t>(
                        output_sigmas_data[i] * static_cast<float>(m_config.num_train_timesteps)));
        }
    }
    if (!m_config.use_flow_sigmas) {
        timesteps = sigma_to_t(output_sigmas, log_sigmas);
    }
    float sigma_last = 0.0f;
    if (m_config.final_sigma_type == FinalSigmaType::ZERO) {
        sigma_last = 0.0f;
    } else if (m_config.final_sigma_type == FinalSigmaType::SIGMA_MIN) {
        sigma_last = output_sigmas_data[num_inference_steps - 1];
    } else {
        OPENVINO_THROW("Unsupported final_sigma_type: " + to_string(m_config.final_sigma_type));
    }

    ov::Shape final_sigmas_shape = output_sigmas.get_shape();
    final_sigmas_shape[0]++;
    ov::Tensor final_sigmas(sigmas.get_element_type(), final_sigmas_shape);
    auto final_sigmas_data = final_sigmas.data<float>();
    std::memcpy(final_sigmas_data, output_sigmas_data, output_sigmas.get_size() * sizeof(float));
    final_sigmas_data[final_sigmas.get_size() - 1] = sigma_last;
    return {final_sigmas, timesteps};
}

std::pair<ov::Tensor, std::vector<int64_t>> UniPCMultistepScheduler::convert_to_exponential(ov::Tensor &sigmas, size_t num_inference_steps) {
    auto sigmas_data = sigmas.data<float>();
    ov::Tensor log_sigmas(sigmas.get_element_type(), sigmas.get_shape());
    auto log_sigmas_data = log_sigmas.data<float>();
    for (size_t i = 0; i < sigmas.get_size(); i++) {
        log_sigmas_data[i] = std::log(sigmas_data[i]);
    }

    std::reverse(sigmas_data, sigmas_data + sigmas.get_size());

    // TODO: implement exponential sigmas conversion
    return {sigmas, std::vector<int64_t>()};
}

std::pair<ov::Tensor, std::vector<int64_t>> UniPCMultistepScheduler::convert_to_flow(size_t num_inference_steps) {
    auto alphas = numpy_utils::linspace<float>(
        1.0f, 
        1.0f / static_cast<float>(m_config.num_train_timesteps), 
        num_inference_steps + 1, 
        true);
    ov::Tensor sigmas(ov::element::f32, {num_inference_steps + 1});
    auto sigmas_data = sigmas.data<float>();
    for (size_t i = 0; i < sigmas.get_size(); i++) {
        float sigma = 1.0f - alphas[i];
        sigmas_data[i] = m_config.flow_shift * sigma / 
                            (1.0f + (m_config.flow_shift - 1.0f) * sigma);
    }
    std::reverse(sigmas_data, sigmas_data + sigmas.get_size());
    float sigma_last = 0.0f;
    if (m_config.final_sigma_type == FinalSigmaType::ZERO) {
        sigma_last = 0.0f;
    } else if (m_config.final_sigma_type == FinalSigmaType::SIGMA_MIN) {
        sigma_last = sigmas_data[num_inference_steps - 1];
    } else {
        OPENVINO_THROW("Unsupported final_sigma_type: " + to_string(m_config.final_sigma_type));
    }
    sigmas_data[num_inference_steps] = sigma_last;
    std::vector<int64_t> timesteps(num_inference_steps);
    for (size_t i = 0; i < num_inference_steps; i++) {
        timesteps[i] = static_cast<int64_t>(sigmas_data[i] * static_cast<float>(m_config.num_train_timesteps));
    }
    return {sigmas, timesteps};
}

std::pair<ov::Tensor, ov::Tensor> UniPCMultistepScheduler::sigma_to_alpha_sigma_t(const ov::Tensor &sigma) {
    ov::Tensor alpha_t(sigma.get_element_type(), sigma.get_shape());
    ov::Tensor sigma_t(sigma.get_element_type(), sigma.get_shape());
    auto sigma_data = sigma.data<const float>();
    auto alpha_t_data = alpha_t.data<float>();
    auto sigma_t_data = sigma_t.data<float>();
    if (m_config.use_flow_sigmas) {
        for (size_t i = 0; i < sigma.get_size(); i++) {
            alpha_t_data[i] = 1.0f - sigma_data[i];
            sigma_t_data[i] = sigma_data[i];
        }
    } else {
        for (size_t i = 0; i < sigma.get_size(); i++) {
            alpha_t_data[i] = 1.0f / (std::sqrt(sigma_data[i] * sigma_data[i] + 1.0f));
            sigma_t_data[i] = sigma_data[i] * alpha_t_data[i];
        }
    }

    return {alpha_t, sigma_t};
}

std::pair<float, float> UniPCMultistepScheduler::sigma_to_alpha_sigma_t(float sigma) {
    float alpha_t = 0.0f;
    float sigma_t = 0.0f;
    if (m_config.use_flow_sigmas) {
        alpha_t = 1.0f - sigma;
        sigma_t = sigma;
    } else {
        alpha_t = 1.0f / (std::sqrt(sigma * sigma + 1.0f));
        sigma_t = sigma * alpha_t;
    }
    return {alpha_t, sigma_t};
}

size_t UniPCMultistepScheduler::index_for_timestep(
    int64_t timestep, std::optional<std::vector<int64_t>> schedule_timesteps) {
    std::vector<int64_t> &schedule_timesteps_local = m_timesteps;
    if (schedule_timesteps.has_value()) {
        schedule_timesteps_local = schedule_timesteps.value();
    }

    std::vector<size_t> index_candidates;
    for (size_t i = 0; i < schedule_timesteps_local.size(); i++) {
        if (schedule_timesteps_local[i] == timestep) {
            index_candidates.push_back(i);
        }
    }

    if (index_candidates.empty()) {
        return m_timesteps.size() - 1;
    } else if (index_candidates.size() > 1) {
        return index_candidates[1];
    } else {
        return index_candidates[0];
    }
}

void UniPCMultistepScheduler::init_step_index(int64_t timestep) {
    if (!m_begin_index.has_value()) {
        m_step_index = index_for_timestep(timestep);
    } else {
        m_step_index = m_begin_index.value();
    }
}

ov::Tensor UniPCMultistepScheduler::convert_model_output(const ov::Tensor &model_output, const ov::Tensor &sample) {
    auto sigmas_data = m_sigmas.data<const float>();
    auto model_output_data = model_output.data<const float>();
    auto sample_data = sample.data<const float>();

    float sigma = sigmas_data[m_step_index.value()];
    float alpha_t, sigma_t;
    std::tie(alpha_t, sigma_t) = sigma_to_alpha_sigma_t(sigma);

    if (m_config.predict_x0) {
        ov::Tensor x0_pred(model_output.get_element_type(), model_output.get_shape());
        auto x0_pred_data = x0_pred.data<float>();
        if (m_config.prediction_type == PredictionType::EPSILON) {
            for (size_t i = 0; i < model_output.get_size(); i++) {
                x0_pred_data[i] = (sample_data[i] - sigma_t * model_output_data[i]) / alpha_t;
            }
        } else if (m_config.prediction_type == PredictionType::SAMPLE) {
            for (size_t i = 0; i < model_output.get_size(); i++) {
                x0_pred_data[i] = model_output_data[i];
            }
        } else if (m_config.prediction_type == PredictionType::V_PREDICTION) {
            for (size_t i = 0; i < model_output.get_size(); i++) {
                x0_pred_data[i] = alpha_t * sample_data[i] - sigma_t * model_output_data[i];
            }
        } else if (m_config.prediction_type == PredictionType::FLOW_PREDICTION) {
            for (size_t i = 0; i < model_output.get_size(); i++) {
                sigma_t = sigmas_data[m_step_index.value()];
                x0_pred_data[i] = sample_data[i] - sigma_t * model_output_data[i];
            }
        }

        if (m_config.thresholding) {
            thresholding_sample(x0_pred);
        }
        return x0_pred;
    } else {
        if (m_config.prediction_type == PredictionType::EPSILON) {
            return model_output;
        } else if (m_config.prediction_type == PredictionType::SAMPLE) {
            ov::Tensor epsilon(model_output.get_element_type(), model_output.get_shape());
            auto epsilon_data = epsilon.data<float>();
            for (size_t i = 0; i < model_output.get_size(); i++) {
                epsilon_data[i] = (sample_data[i] - alpha_t * model_output_data[i]) / sigma_t;
            }
            return epsilon;
        } else if (m_config.prediction_type == PredictionType::V_PREDICTION) {
            ov::Tensor epsilon(model_output.get_element_type(), model_output.get_shape());
            auto epsilon_data = epsilon.data<float>();
            for (size_t i = 0; i < model_output.get_size(); i++) {
                epsilon_data[i] = alpha_t * model_output_data[i] + sigma_t * sample_data[i];
            }
            return epsilon;
        } else {
            OPENVINO_THROW("Unsupported prediction_type: " + to_string(m_config.prediction_type));
        }
    }
}

void UniPCMultistepScheduler::thresholding_sample(ov::Tensor &sample) {
    const ov::Shape &sample_shape = sample.get_shape();

    size_t batch_size = sample_shape[0];

    size_t elements_per_sample = 1;
    for (size_t i = 1; i < sample_shape.size(); i++) {
        elements_per_sample *= sample_shape[static_cast<long>(i)];
    }

    auto sample_data = sample.data<float>();

    std::vector<float> abs(elements_per_sample);
    for (size_t i = 0; i < batch_size; i++) {
        size_t offset = i * elements_per_sample;
        for (size_t j = 0; j < elements_per_sample; j++) {
            abs[j] = std::abs(sample_data[offset + j]);
        }

        float s = compute_quantile(abs, m_config.dynamic_thresholding_ratio);
        s = std::clamp(s, 1.0f, m_config.sample_max_value);
        for (size_t j = 0; j < elements_per_sample; j++) {
            sample_data[offset + j] = std::clamp(sample_data[offset + j], -s, s) / s;
        }
    }
}

float UniPCMultistepScheduler::compute_quantile(std::vector<float> &data, float quantile) {
    if (data.empty()) {
        return 0.0f;
    }

    size_t n = data.size();
    float pos = quantile * (static_cast<float>(n) - 1.0f);
    size_t idx = static_cast<size_t>(pos);
    float frac = pos - static_cast<float>(idx);

    if (idx >= n - 1) {
        std::nth_element(data.begin(), data.begin() + static_cast<long>(n) - 1, data.end());
        return data[n - 1];
    }

    std::nth_element(data.begin(), data.begin() + static_cast<long>(idx), data.end());
    float lower = data[idx];
    float upper = *std::min_element(data.begin() + static_cast<long>(idx) + 1, data.end());
    return lower * (1.0f - frac) + upper * frac;
}

ov::Tensor UniPCMultistepScheduler::einsum_k_bkc_to_bc(const ov::Tensor &D1s, const ov::Tensor &rhos) {
    ov::Shape result_shape = D1s.get_shape();
    if (result_shape.size() < 3) {
        OPENVINO_THROW("D1s tensor must be more than 3-dimensional for einsum_k_bkc_to_bc");
    }
    size_t B = result_shape[0];
    size_t K = result_shape[1];
    size_t C = result_shape[2];
    size_t tail = 1;
    for (size_t i = 3; i < result_shape.size(); i++) {
        tail *= result_shape[static_cast<long>(i)];
    }

    result_shape.erase(result_shape.begin() + 1);
    ov::Tensor result(ov::element::f32, result_shape);
    auto result_data = result.data<float>();
    auto D1s_data = D1s.data<const float>();
    auto rhos_data = rhos.data<const float>();

    for (size_t b = 0; b < B; b++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t t  = 0; t < tail; t++) {
                float accumulated = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    accumulated += (D1s_data[((b * K + k) * C + c) * tail + t] * rhos_data[k]);
                }
                result_data[(b * C + c) * tail + t] = accumulated;
            }
            
        }
    }
    return result;
}

ov::Tensor UniPCMultistepScheduler::multistep_uni_c_bh_update(
    ov::Tensor &this_model_output, ov::Tensor &last_sample, ov::Tensor &this_sample, int order) {
    if (m_model_outputs.empty() || !m_model_outputs[m_model_outputs.size() - 1].has_value()) {
        OPENVINO_THROW("Model outputs are empty, cannot perform multistep uni c bh update");
    }
    ov::Tensor &m0 = m_model_outputs[m_model_outputs.size() - 1].value();
    auto m0_data = m0.data<const float>();
    ov::Tensor &x = last_sample;
    auto x_data = x.data<const float>();
    ov::Tensor &x_t = this_sample;
    auto x_t_data = x_t.data<const float>();
    ov::Tensor &model_t = this_model_output;
    auto model_t_data = model_t.data<const float>();

    float sigma_t = m_sigmas.data<const float>()[m_step_index.value()];
    if (m_step_index.value() == 0) {
        OPENVINO_THROW("Step index is zero, cannot perform multistep uni c bh update");
    }
    float sigma_s0 = m_sigmas.data<const float>()[m_step_index.value() - 1];
    float alpha_t;
    std::tie(alpha_t, sigma_t) = sigma_to_alpha_sigma_t(sigma_t);
    float alpha_s0;
    std::tie(alpha_s0, sigma_s0) = sigma_to_alpha_sigma_t(sigma_s0);

    float lambda_t = std::log(alpha_t) - std::log(sigma_t);
    float lambda_s0 = std::log(alpha_s0) - std::log(sigma_s0);

    float h = lambda_t - lambda_s0;

    std::vector<float> rks;
    ov::Tensor D1s;
    std::vector<ov::Tensor> D1_list;
    D1_list.reserve(order - 1);
    for (size_t i = 1; i < order; i++) {
        size_t si = m_step_index.value() - (i + 1);
        ov::Tensor &mi = m_model_outputs[m_model_outputs.size() - (i + 1)].value();
        float alpha_si, sigma_si;
        std::tie(alpha_si, sigma_si) = sigma_to_alpha_sigma_t(m_sigmas.data<const float>()[si]);
        float lambda_si = std::log(alpha_si) - std::log(sigma_si);
        float rk = (lambda_si - lambda_s0) / h;
        rks.push_back(rk);
        auto mi_data = mi.data<const float>();
        ov::Tensor D1(ov::element::f32, m0.get_shape());
        auto D1_data = D1.data<float>();
        for (size_t j = 0; j < m0.get_size(); j++) {
            D1_data[j] = (mi_data[j] - m0_data[j]) / rk;
        }
        D1_list.push_back(D1);
    }

    rks.push_back(1.0f);
    
    float hh = h;
    if (m_config.predict_x0) {
        hh = -h;
    }
    float h_phi_1 = std::expm1(hh);
    float h_phi_k = h_phi_1 / hh - 1;
    int factorial_i = 1;

    float B_h = hh;
    if (m_config.solver_type == SolverType::BH1) {
        B_h = hh;
    } else if (m_config.solver_type == SolverType::BH2) {
        B_h = std::expm1(hh);
    } else {
        OPENVINO_THROW("Unsupported solver_type for multistep uni c bh update: " + to_string(m_config.solver_type));
    }

    ov::Shape r_shape = {static_cast<size_t>(order), rks.size()};
    ov::Tensor R(ov::element::f32, r_shape);
    auto R_data = R.data<float>();
    ov::Tensor b(ov::element::f32, {static_cast<size_t>(order)});
    auto b_data = b.data<float>();

    for (size_t i = 1; i < order + 1; i++) {
        for (size_t j = 0; j < rks.size(); j++) {
            R_data[(i - 1) * rks.size() + j] = static_cast<float>(std::pow(rks[j], i - 1));
        }
        b_data[i - 1] = h_phi_k * static_cast<float>(factorial_i) / B_h;
        factorial_i *= (static_cast<int>(i) + 1);
        h_phi_k = h_phi_k / hh - 1.0f / static_cast<float>(factorial_i);
    }

    if (!D1_list.empty()) {
        D1s = tensor_utils::stack(D1_list, 1);
    }

    ov::Tensor rhos_c;
    if (order == 1) {
        rhos_c = ov::Tensor(ov::element::f32, {1});
        rhos_c.data<float>()[0] = 0.5f;
    } else {
        m_c_solver.set_input_tensor(0, R);
        m_c_solver.set_input_tensor(1, b);
        m_c_solver.infer();
        rhos_c = m_c_solver.get_output_tensor(0);
    }

    if (m_config.predict_x0) {
        ov::Tensor x_t_(x.get_element_type(), x.get_shape());
        auto x_t_data_1 = x_t_.data<float>();
        for (size_t i = 0; i < x_t_.get_size(); i++) {
            x_t_data_1[i] = sigma_t / sigma_s0 * x_data[i] - alpha_t * h_phi_1 * m0_data[i];
        }
        ov::Tensor corr_res;
        if (!D1_list.empty()) {
            corr_res = einsum_k_bkc_to_bc(D1s, rhos_c);
        } else {
            corr_res = ov::Tensor(ov::element::f32, m0.get_shape());
            auto corr_res_data = corr_res.data<float>();
            for (size_t i = 0; i < corr_res.get_size(); i++) {
                corr_res_data[i] = 0.0f;
            }
        }
        ov::Tensor x_t_final(x_t.get_element_type(), x_t.get_shape());
        auto x_t_final_data = x_t_final.data<float>();
        auto corr_res_data = corr_res.data<const float>();
        float rhos = rhos_c.data<const float>()[rhos_c.get_size() - 1];
        for (size_t i = 0; i < x_t_final.get_size(); i++) {
            float D1_t = model_t_data[i] - m0_data[i];
            x_t_final_data[i] = x_t_data_1[i] - alpha_t * B_h * (corr_res_data[i] + rhos * D1_t);
        }
        return x_t_final;
    } else {
        ov::Tensor x_t_(x.get_element_type(), x.get_shape());
        auto x_t_data_1 = x_t_.data<float>();
        for (size_t i = 0; i < x_t_.get_size(); i++) {
            x_t_data_1[i] = alpha_t / alpha_s0 * x_data[i] - sigma_t * h_phi_1 * m0_data[i];
        }
        ov::Tensor corr_res;
        if (!D1_list.empty()) {
            corr_res = einsum_k_bkc_to_bc(D1s, rhos_c);
        } else {
            corr_res = ov::Tensor(ov::element::f32, m0.get_shape());
            auto corr_res_data = corr_res.data<float>();
            for (size_t i = 0; i < corr_res.get_size(); i++) {
                corr_res_data[i] = 0.0f;
            }
        }
        ov::Tensor x_t_final(x_t.get_element_type(), x_t.get_shape());
        auto x_t_final_data = x_t_final.data<float>();
        auto corr_res_data = corr_res.data<const float>();
        float rhos = rhos_c.data<const float>()[rhos_c.get_size() - 1];
        for (size_t i = 0; i < x_t_final.get_size(); i++) {
            float D1_t = model_t_data[i] - m0_data[i];
            x_t_final_data[i] = x_t_data_1[i] - sigma_t * B_h * (corr_res_data[i] + rhos * D1_t);
        }
        return x_t_final;
    }

}

ov::Tensor UniPCMultistepScheduler::multistep_uni_p_bh_update(
    ov::Tensor &model_output, ov::Tensor &sample, int order) {
    ov::Tensor &m0 = m_model_outputs[m_model_outputs.size() - 1].value();
    auto m0_data = m0.data<const float>();
    ov::Tensor &x = sample;
    auto x_data = x.data<const float>();

    // TODO: Add solver_p operation

    float sigma_t = m_sigmas.data<const float>()[m_step_index.value() + 1];
    float sigma_s0 = m_sigmas.data<const float>()[m_step_index.value()];
    float alpha_t;
    std::tie(alpha_t, sigma_t) = sigma_to_alpha_sigma_t(sigma_t);
    float alpha_s0;
    std::tie(alpha_s0, sigma_s0) = sigma_to_alpha_sigma_t(sigma_s0);
    float lambda_t = std::log(alpha_t) - std::log(sigma_t);
    float lambda_s0 = std::log(alpha_s0) - std::log(sigma_s0);
    float h = lambda_t - lambda_s0;

    std::vector<float> rks;
    ov::Tensor D1s;
    std::vector<ov::Tensor> D1_list;
    D1_list.reserve(order - 1);
    for (size_t i = 1; i < order; i++) {
        size_t si = m_step_index.value() - i;
        ov::Tensor &mi = m_model_outputs[m_model_outputs.size() - (i + 1)].value();
        float alpha_si, sigma_si;
        std::tie(alpha_si, sigma_si) = sigma_to_alpha_sigma_t(m_sigmas.data<const float>()[si]);
        float lambda_si = std::log(alpha_si) - std::log(sigma_si);
        float rk = (lambda_si - lambda_s0) / h;
        rks.push_back(rk);
        auto mi_data = mi.data<const float>();
        ov::Tensor D1(ov::element::f32, m0.get_shape());
        auto D1_data = D1.data<float>();
        for (size_t j = 0; j < m0.get_size(); j++) {
            D1_data[j] = (mi_data[j] - m0_data[j]) / rk;
        }
        D1_list.push_back(D1);
    }

    rks.push_back(1.0f);

    float hh = h;
    if (m_config.predict_x0) {
        hh = -h;
    }
    float h_phi_1 = std::expm1(hh);
    float h_phi_k = h_phi_1 / hh - 1;
    int factorial_i = 1;

    float B_h = hh;
    if (m_config.solver_type == SolverType::BH1) {
        B_h = hh;
    } else if (m_config.solver_type == SolverType::BH2) {
        B_h = std::expm1(hh);
    } else {
        OPENVINO_THROW("Unsupported solver_type for multistep uni c bh update: " + to_string(m_config.solver_type));
    }

    ov::Shape r_shape = {static_cast<size_t>(order), rks.size()};
    ov::Tensor R(ov::element::f32, r_shape);
    auto R_data = R.data<float>();
    ov::Tensor b(ov::element::f32, {static_cast<size_t>(order)});
    auto b_data = b.data<float>();

    for (size_t i = 1; i < order + 1; i++) {
        for (size_t j = 0; j < rks.size(); j++) {
            R_data[(i - 1) * rks.size() + j] = static_cast<float>(std::pow(rks[j], i - 1));
        }
        b_data[i - 1] = h_phi_k * static_cast<float>(factorial_i) / B_h;
        factorial_i *= (static_cast<int>(i) + 1);
        h_phi_k = h_phi_k / hh - 1.0f / static_cast<float>(factorial_i);
    }

    ov::Tensor rhos_p;
    if (!D1_list.empty()) {
        D1s = tensor_utils::stack(D1_list, 1);
        if (order == 2) {
            rhos_p = ov::Tensor(ov::element::f32, {1});
            rhos_p.data<float>()[0] = 0.5f;
        } else {
            m_p_solver.set_input_tensor(0, R);
            m_p_solver.set_input_tensor(1, b);
            m_p_solver.infer();
            rhos_p = m_p_solver.get_output_tensor(0);
        }
    }

    if (m_config.predict_x0) {
        ov::Tensor x_t_(x.get_element_type(), x.get_shape());
        auto x_t_data_1 = x_t_.data<float>();
        for (size_t i = 0; i < x_t_.get_size(); i++) {
            x_t_data_1[i] = sigma_t / sigma_s0 * x_data[i] - alpha_t * h_phi_1 * m0_data[i];
        }
        ov::Tensor pred_res;
        if (!D1_list.empty()) {
            pred_res = einsum_k_bkc_to_bc(D1s, rhos_p);
        } else {
            pred_res = ov::Tensor(ov::element::f32, m0.get_shape());
            auto pred_res_data = pred_res.data<float>();
            for (size_t i = 0; i < pred_res.get_size(); i++) {
                pred_res_data[i] = 0.0f;
            }
        }
        ov::Tensor x_t(x.get_element_type(), x.get_shape());
        auto x_t_data = x_t.data<float>();
        auto pred_res_data = pred_res.data<const float>();
        for (size_t i = 0; i < x_t.get_size(); i++) {
            x_t_data[i] = x_t_data_1[i] - alpha_t * B_h * pred_res_data[i];
        }
        return x_t;
    } else {
        ov::Tensor x_t_(x.get_element_type(), x.get_shape());
        auto x_t_data_1 = x_t_.data<float>();
        for (size_t i = 0; i < x_t_.get_size(); i++) {
            x_t_data_1[i] = alpha_t / alpha_s0 * x_data[i] - sigma_t * h_phi_1 * m0_data[i];
        }
        ov::Tensor pred_res;
        if (!D1_list.empty()) {
            pred_res = einsum_k_bkc_to_bc(D1s, rhos_p);
        } else {
            pred_res = ov::Tensor(ov::element::f32, m0.get_shape());
            auto pred_res_data = pred_res.data<float>();
            for (size_t i = 0; i < pred_res.get_size(); i++) {
                pred_res_data[i] = 0.0f;
            }
        }
        ov::Tensor x_t(x.get_element_type(), x.get_shape());
        auto x_t_data = x_t.data<float>();
        auto pred_res_data = pred_res.data<const float>();
        for (size_t i = 0; i < x_t.get_size(); i++) {
            x_t_data[i] = x_t_data_1[i] - sigma_t * B_h * pred_res_data[i];
        }
        return x_t;
    }
}

std::map<std::string, ov::Tensor> UniPCMultistepScheduler::step(
    ov::Tensor &model_output,
    int64_t timestep,
    ov::Tensor sample) {
    if (!m_num_inference_steps.has_value()) {
        OPENVINO_THROW("Number of inference steps is empty, , you need to run 'set_timesteps' after creating the scheduler");
    }

    if (!m_step_index.has_value()) {
        init_step_index(timestep);
    }

    bool use_corrector = m_step_index.value() > 0 &&
                         std::find(
                            m_config.disable_corrector.begin(),
                            m_config.disable_corrector.end(),
                            m_step_index.value() - 1) == m_config.disable_corrector.end() &&
                         m_last_sample.has_value();
    ov::Tensor model_output_convert = convert_model_output(model_output, sample);
    if (use_corrector) {
        sample = multistep_uni_c_bh_update(
            model_output_convert,
            m_last_sample.value(),
            sample,
            m_this_order.value());
    }

    for (size_t i = 0; i < m_config.solver_order - 1; i++) {
        m_model_outputs[i] = m_model_outputs[i + 1];
        m_timestep_list[i] = m_timestep_list[i + 1];
    }

    m_model_outputs[m_config.solver_order - 1] = model_output_convert;
    m_timestep_list[m_config.solver_order - 1] = timestep;

    int this_order;
    if (m_config.lower_order_final) {
        this_order = std::min(m_config.solver_order, static_cast<int>(m_timesteps.size() - m_step_index.value()));
    } else {
        this_order = m_config.solver_order;
    }

    m_this_order = std::min(this_order, static_cast<int>(m_lower_order_nums.value() + 1));
    OPENVINO_ASSERT(m_this_order > 0);

    m_last_sample = sample;
    ov::Tensor prev_sample = multistep_uni_p_bh_update(
        model_output, sample, m_this_order.value());
    
    if (m_lower_order_nums.value() < m_config.solver_order) {
        m_lower_order_nums = m_lower_order_nums.value() + 1;
    }

    m_step_index = m_step_index.value() + 1;
    return {{"prev_sample", prev_sample}};
}

}
