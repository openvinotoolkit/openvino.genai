// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief a header file for SD pipeline
 * @file stable_diffusion.hpp
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <openvino/openvino.hpp>
#include <random>
#include <regex>
#include <stack>
#include <string>
#include <unordered_map>
#include <utils.hpp>
#include <vector>

#include "logger.hpp"
#include "lora_cpp.hpp"
#include "process_bar.hpp"
#include "write_bmp.hpp"

Logger logger("log.txt");

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

std::vector<float> LMSDiscreteScheduler(int32_t num_train_timesteps = 1000,
                                        float beta_start = 0.00085f,
                                        float beta_end = 0.012f,
                                        std::string beta_schedule = "scaled_linear",
                                        std::string prediction_type = "epsilon",
                                        std::vector<float> trained_betas = std::vector<float>{}) {
    std::string _predictionType = prediction_type;
    auto Derivatives = std::vector<std::vector<float>>{};
    auto Timesteps = std::vector<int>();

    auto alphas = std::vector<float>();
    auto betas = std::vector<float>();
    if (!trained_betas.empty()) {
        auto betas = trained_betas;
    } else if (beta_schedule == "linear") {
        for (int32_t i = 0; i < num_train_timesteps; i++) {
            betas.push_back(beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1));
        }
    } else if (beta_schedule == "scaled_linear") {
        float start = sqrt(beta_start);
        float end = sqrt(beta_end);
        std::vector<float> temp = linspace(start, end, num_train_timesteps);
        for (float b : temp) {
            betas.push_back(b * b);
        }
    } else {
        std::cout << " beta_schedule must be one of 'linear' or 'scaled_linear' " << std::endl;
    }
    for (float b : betas) {
        alphas.push_back(1 - b);
    }
    std::vector<float> log_sigma_vec;
    for (int32_t i = 1; i <= (int)alphas.size(); i++) {
        float alphas_cumprod =
            std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
        float sigma = sqrt((1 - alphas_cumprod) / alphas_cumprod);
        log_sigma_vec.push_back(std::log(sigma));
    }
    return log_sigma_vec;
}

std::vector<float> std_randn_function(uint32_t seed, uint32_t h, uint32_t w) {
    std::vector<float> noise;
    {
        std::mt19937 gen{static_cast<unsigned long>(seed)};
        std::normal_distribution<float> normal{0.0f, 1.0f};
        noise.resize(h / 8 * w / 8 * 4 * 1);
        std::for_each(noise.begin(), noise.end(), [&](float& x) {
            x = normal(gen);
        });
    }
    return noise;
}

std::vector<uint8_t> vae_decoder_function(ov::CompiledModel& decoder_compiled_model,
                                          std::vector<float>& sample,
                                          uint32_t h,
                                          uint32_t w) {
    logger.log_vector(LogLevel::DEBUG, "DEBUG-sample.values: ", sample, 0, 5);

    auto decoder_input_port = decoder_compiled_model.input();
    auto decoder_output_port = decoder_compiled_model.output();
    auto shape = decoder_input_port.get_partial_shape();
    logger.log_value(LogLevel::DEBUG, "decoder_input_port.get_partial_shape(): ", shape);

    const ov::element::Type type = decoder_input_port.get_element_type();

    float coeffs_const{1 / 0.18215};
    std::for_each(sample.begin(), sample.end(), [coeffs_const](float& i) {
        i *= coeffs_const;
    });
    ov::Shape sample_shape = {1, 4, h / 8, w / 8};
    ov::Tensor decoder_input_tensor(type, sample_shape, sample.data());

    ov::InferRequest infer_request = decoder_compiled_model.create_infer_request();
    infer_request.set_tensor(decoder_input_port, decoder_input_tensor);
    // infer_request.start_async();
    // infer_request.wait();
    infer_request.infer();
    ov::Tensor decoder_output_tensor = infer_request.get_tensor(decoder_output_port);
    auto decoder_output_ptr = decoder_output_tensor.data<float>();
    std::vector<float> decoder_output_vec;
    std::vector<uint8_t> output_vec;

    for (size_t i = 0; i < 3 * h * w; i++) {
        decoder_output_vec.push_back(decoder_output_ptr[i]);
    }

    // np.clip(image / 2 + 0.5, 0, 1)
    logger.log_vector(LogLevel::DEBUG, "decoder_output_vec: ", decoder_output_vec, 0, 5);
    float mul_const{0.5};
    std::transform(decoder_output_vec.begin(),
                   decoder_output_vec.end(),
                   decoder_output_vec.begin(),
                   [&mul_const](auto& c) {
                       return c * mul_const;
                   });
    float add_const{0.5};
    std::transform(decoder_output_vec.begin(),
                   decoder_output_vec.end(),
                   decoder_output_vec.begin(),
                   [&add_const](auto& c) {
                       return c + add_const;
                   });
    std::transform(decoder_output_vec.begin(), decoder_output_vec.end(), decoder_output_vec.begin(), [=](auto i) {
        return std::clamp(i, 0.0f, 1.0f);
    });

    logger.log_vector(LogLevel::DEBUG, "image post-process to set values [0,1]: ", decoder_output_vec, 0, 5);

    for (size_t i = 0; i < decoder_output_vec.size(); i++) {
        output_vec.push_back(static_cast<uint8_t>(decoder_output_vec[i] * 255.0f));
    }

    return output_vec;
}

std::vector<int64_t> sigma_to_t(std::vector<float>& log_sigmas, float sigma) {
    double log_sigma = std::log(sigma);
    std::vector<float> dists(1000);
    for (int32_t i = 0; i < 1000; i++) {
        if (log_sigma - log_sigmas[i] >= 0)
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
    float low = log_sigmas[low_idx];
    float high = log_sigmas[high_idx];
    // interpolate sigmas
    double w = (low - log_sigma) / (low - high);
    w = std::max(0.0, std::min(1.0, w));

    int64_t t = std::llround((1 - w) * low_idx + w * high_idx);
    std::vector<int64_t> vector_t{t};
    return vector_t;
}

std::vector<float> unet_infer_function(ov::CompiledModel& unet_model,
                                       std::vector<int64_t>& vector_t,
                                       std::vector<float>& latent_input_1d,
                                       std::vector<float>& text_embedding_1d,
                                       uint32_t u_h,
                                       uint32_t u_w) {
    auto t0 = std::chrono::steady_clock::now();

    ov::InferRequest unet_infer_request = unet_model.create_infer_request();

    auto t1 = std::chrono::steady_clock::now();
    auto duration_create_infer_request = std::chrono::duration_cast<std::chrono::duration<float>>(t1 - t0);
    logger.log_value(LogLevel::DEBUG, "duration of create_infer_request(s): ", duration_create_infer_request.count());

    auto input_port = unet_model.inputs();
    uint32_t latent_h = u_h / 8;
    uint32_t latent_w = u_w / 8;

    for (auto input : unet_model.inputs()) {
        const ov::element::Type type = input.get_element_type();
        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        logger.log_value(LogLevel::DEBUG, "unet.get_partial_shape(): ", input.get_partial_shape());

        if (name == "sample") {  // latent_model_input
            ov::Shape latent_shape = {2, 4, latent_h, latent_w};
            ov::Tensor input_tensor_0 = ov::Tensor(type, latent_shape, latent_input_1d.data());

            unet_infer_request.set_tensor(name, input_tensor_0);
        }
        if (name == "timestep") {  // t
            ov::Shape ts_shape = {1};
            ov::Tensor input_tensor_1 = ov::Tensor(type, ts_shape, vector_t.data());

            unet_infer_request.set_tensor(name, input_tensor_1);
        }
        if (name == "encoder_hidden_states") {
            ov::Shape encoder_shape = {2, 77, 768};
            ov::Tensor input_tensor_2 = ov::Tensor(type, encoder_shape, text_embedding_1d.data());

            unet_infer_request.set_tensor(name, input_tensor_2);
        }
    }

    // unet_infer_request.start_async();
    // unet_infer_request.wait();
    auto t2 = std::chrono::steady_clock::now();

    unet_infer_request.infer();

    auto t3 = std::chrono::steady_clock::now();
    auto duration_set_tensor = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);
    logger.log_value(LogLevel::DEBUG, "duration of set_tensor(s): ", duration_set_tensor.count());
    // std::cout << "duration of set_tensor(s): " << duration_set_tensor.count() << std::endl;

    auto duration_infer = std::chrono::duration_cast<std::chrono::duration<float>>(t3 - t2);
    logger.log_value(LogLevel::DEBUG, "duration of unet_infer_request.infer()(s): ", duration_infer.count());
    // std::cout << "duration of infer(s): " << duration_infer.count() << std::endl;

    std::vector<ov::Output<const ov::Node>> output_port = unet_model.outputs();
    ov::Tensor noise_pred_tensor = unet_infer_request.get_output_tensor();

    auto noise_pred_ptr = noise_pred_tensor.data<float>();

    std::vector<float> noise_pred_uncond_vec(noise_pred_ptr, noise_pred_ptr + (latent_h * latent_w * 4));
    std::vector<float> noise_pred_text_vec(noise_pred_ptr + (latent_h * latent_w * 4),
                                           noise_pred_ptr + (latent_h * latent_w * 4 * 2));
    std::vector<float> noise_pred_vec;
    logger.log_value(LogLevel::DEBUG, "DEBUG-noise_pred_uncond_vec.size(): ", noise_pred_uncond_vec.size());

    float guidance_scale = 7.5;

    for (int32_t i = 0; i < (int)noise_pred_uncond_vec.size(); i++) {
        float result = noise_pred_uncond_vec[i] + guidance_scale * (noise_pred_text_vec[i] - noise_pred_uncond_vec[i]);
        noise_pred_vec.push_back(result);
    }
    logger.log_string(LogLevel::DEBUG, "DEBUG-perform guidance: ");
    logger.log_vector(LogLevel::DEBUG, "uncond: ", noise_pred_uncond_vec, 0, 5);
    logger.log_vector(LogLevel::DEBUG, "text: ", noise_pred_text_vec, 0, 5);
    logger.log_vector(LogLevel::DEBUG, "noise_pred with post_process: ", noise_pred_vec, 0, 5);

    return noise_pred_vec;
}

float lms_derivative_function(float tau, int32_t order, int32_t curr_order, std::vector<float> sigma_vec, int32_t t) {
    float prod = 1.0;

    for (int32_t k = 0; k < order; k++) {
        if (curr_order == k) {
            continue;
        }
        prod *= (tau - sigma_vec[t - k]) / (sigma_vec[t - curr_order] - sigma_vec[t - k]);
    }
    return prod;
}

std::vector<float> np_randn_function() {
    // read np generated latents with defaut seed 42
    std::vector<float> latent_vector_1d;
    std::ifstream latent_copy_file;
    latent_copy_file.open("../scripts/np_latents_512x512.txt");
    std::vector<std::string> latent_vector_new;
    if (latent_copy_file.is_open()) {
        std::string word;
        while (latent_copy_file >> word)
            latent_vector_new.push_back(word);
        latent_copy_file.close();
    } else {
        std::cout << "could not find the np_latents_512x512.txt" << std::endl;
        exit(0);
    }

    latent_vector_new.insert(latent_vector_new.begin(), latent_vector_new.begin(), latent_vector_new.end());

    for (int i = 0; i < (int)latent_vector_new.size() / 2; i++) {
        latent_vector_1d.push_back(std::stof(latent_vector_new[i]));
    }

    return latent_vector_1d;
}

void convertBGRtoRGB(std::vector<unsigned char>& image, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        // Swap the red and blue components (BGR to RGB)
        unsigned char temp = image[i * 3];
        image[i * 3] = image[i * 3 + 2];
        image[i * 3 + 2] = temp;
    }
}

std::vector<float> diffusion_function(ov::CompiledModel& unet_compiled_model,
                                      uint32_t seed,
                                      int32_t step,
                                      uint32_t d_h,
                                      uint32_t d_w,
                                      std::vector<float>& latent_vector_1d,
                                      std::vector<float>& text_embeddings_2_77_768) {
    std::vector<float> log_sigma_vec = LMSDiscreteScheduler();

    // t_to_sigma
    std::vector<float> sigma(step);
    float delta = -999.0f / (step - 1);

    // transform interpolation to time range

    for (int32_t i = 0; i < step; i++) {
        float t = 999.0 + i * delta;
        int32_t low_idx = std::floor(t);
        int32_t high_idx = std::ceil(t);
        float w = t - low_idx;
        sigma[i] = std::exp((1 - w) * log_sigma_vec[low_idx] + w * log_sigma_vec[high_idx]);
    }

    sigma.push_back(0.f);

    logger.log_vector(LogLevel::DEBUG, "sigma: ", sigma, 0, 20);

    // LMSDiscreteScheduler: latents are multiplied by sigmas
    double n{sigma[0]};  // 14.6146
    std::vector<float> latent_vector_1d_new = latent_vector_1d;
    std::transform(latent_vector_1d.begin(), latent_vector_1d.end(), latent_vector_1d_new.begin(), [&n](auto& c) {
        return c * n;
    });

    std::vector<std::vector<float>> derivative_list;

    process_bar bar(sigma.size());

    for (int32_t i = 0; i < step; i++) {
        bar.progress(i);

        logger.log_string(LogLevel::DEBUG, "------------------------------------");
        logger.log_value(LogLevel::DEBUG, "step: ", i);

        std::vector<int64_t> t = sigma_to_t(log_sigma_vec, sigma[i]);
        logger.log_value(LogLevel::DEBUG, "t: ", t[0]);

        std::vector<float> latent_model_input;
        for (int32_t j = 0; j < static_cast<int>(latent_vector_1d_new.size()); j++) {
            latent_model_input.push_back(latent_vector_1d_new[j]);
        }

        // expand the latents for classifier free guidance:
        latent_model_input.insert(latent_model_input.end(), latent_model_input.begin(), latent_model_input.end());

        // scale_model_input
        for (int32_t j = 0; j < static_cast<int>(latent_model_input.size()); j++) {
            latent_model_input[j] = latent_model_input[j] / sqrt((sigma[i] * sigma[i] + 1));
        }

        logger.log_value(LogLevel::DEBUG, "DEBUG-latent_model_input.size(): ", latent_model_input.size());
        logger.log_vector(LogLevel::DEBUG, "text_em0: ", text_embeddings_2_77_768, 0, 5);
        logger.log_vector(LogLevel::DEBUG, "text_em1: ", text_embeddings_2_77_768, 77 * 768, 5);
        logger.log_vector(LogLevel::DEBUG, "latent: ", latent_model_input, 0, 5);

        // std::cout << "text_em1:  " ;
        // for (int32_t i= 0; i <5 ; i++ ) {
        //     // std::cout << text_embeddings_2_77_768[i*768+ 77*768] << " ";
        // }

        auto start = std::chrono::steady_clock::now();

        std::vector<float> noise_pred_1d =
            unet_infer_function(unet_compiled_model, t, latent_model_input, text_embeddings_2_77_768, d_h, d_w);

        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
        // std::cout << "duration of unet_infer_function(s): " << duration.count() << std::endl;
        logger.log_value(LogLevel::DEBUG, "duration of unet_infer_function(s): ", duration.count());
        logger.log_value(LogLevel::DEBUG,
                         "DEBUG-noise_pred_1d.size() after unet_infer_function: ",
                         noise_pred_1d.size());

        auto start_post = std::chrono::steady_clock::now();
        // LMS step function:
        std::vector<float> derivative_vec_1d;
        int32_t order = 4;
        for (int32_t j = 0; j < static_cast<int>(latent_vector_1d.size()); j++) {
            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            // defaut "epsilon"
            float pred_latent = latent_vector_1d[j] - sigma[i] * noise_pred_1d[j];

            // 2. Convert to an ODE derivative
            derivative_vec_1d.push_back((latent_vector_1d[j] - pred_latent) / sigma[i]);
        }

        derivative_list.push_back(derivative_vec_1d);
        // keep the list size within 4
        if ((int)derivative_list.size() > order) {
            derivative_list.erase(derivative_list.begin());
        }

        for (int32_t m = 0; m < (int32_t)derivative_list.size(); m++) {
            logger.log_vector(LogLevel::DEBUG, "DEBUG-derivative_list: ", derivative_list[m], 0, 5);
        }

        // 3. Compute linear multistep coefficients
        order = std::min(i + 1, order);
        logger.log_value(LogLevel::DEBUG, "Debug-order: ", order);

        std::vector<float> lms_coeffs;
        for (int32_t curr_order = 0; curr_order < order; curr_order++) {
            auto f = [order, curr_order, sigma, i](float tau) {
                return lms_derivative_function(tau, order, curr_order, sigma, i);
            };
            // auto start1 = std::chrono::steady_clock::now();
            // auto integrated_coeff = boost::math::quadrature::trapezoidal(f,
            //                                                              static_cast<double>(sigma[i]),
            //                                                              static_cast<double>(sigma[i + 1]),
            //                                                              1e-4);
            // auto end1 = std::chrono::steady_clock::now();
            // auto duration1 = std::chrono::duration_cast<std::chrono::duration<float>>(end1 - start1);
            auto start2 = std::chrono::steady_clock::now();
            auto integrated_coeff_new =
                trapezoidal(f, static_cast<double>(sigma[i]), static_cast<double>(sigma[i + 1]), 1e-4);
            auto end2 = std::chrono::steady_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::duration<float>>(end2 - start2);

            lms_coeffs.push_back(integrated_coeff_new);
            // logger.log_value(LogLevel::DEBUG, "Debug-integrated_coeff: ", integrated_coeff);
            logger.log_value(LogLevel::DEBUG, "Debug-integrated_coeff_new: ", integrated_coeff_new);
            // logger.log_value(LogLevel::DEBUG, "Debug-integrated_coeff time: ", duration1.count());
            logger.log_value(LogLevel::DEBUG, "Debug-integrated_coeff_new time : ", duration2.count());
        }

        // 4. Compute previous sample based on the derivatives path
        // prev_sample = sample + sum(coeff * derivative for coeff, derivative in zip(lms_coeffs,
        // reversed(self.derivatives))) Reverse list of tensors this.derivatives
        std::vector<std::vector<float>> rev_derivative = derivative_list;
        std::reverse(rev_derivative.begin(), rev_derivative.end());
        // derivative * coeffs
        for (int32_t m = 0; m < order; m++) {
            float coeffs_const{lms_coeffs[m]};
            std::for_each(rev_derivative[m].begin(), rev_derivative[m].end(), [coeffs_const](float& i) {
                i *= coeffs_const;
            });
        }
        // sum of derivative
        std::vector<float> derivative_sum = rev_derivative[0];
        if (order > 1) {
            for (int32_t d = 0; d < order - 1; d++) {
                std::transform(derivative_sum.begin(),
                               derivative_sum.end(),
                               rev_derivative[d + 1].begin(),
                               derivative_sum.begin(),
                               [](float x, float y) {
                                   return x + y;
                               });
            }
        }
        // latent + sum of derivative
        std::transform(derivative_sum.begin(),
                       derivative_sum.end(),
                       latent_vector_1d_new.begin(),
                       latent_vector_1d_new.begin(),
                       [](float x, float y) {
                           return x + y;
                       });
        auto end_post = std::chrono::steady_clock::now();
        auto duration_post = std::chrono::duration_cast<std::chrono::duration<float>>(end_post - start_post);
        // std::cout << "duration of unet post integration(s): " << duration_post.count() << std::endl;
        logger.log_value(LogLevel::DEBUG, "duration of unet post integration(s): ", duration_post.count());
        logger.log_vector(LogLevel::DEBUG, "Debug-latent_vector_1d_new: ", latent_vector_1d_new, 0, 5);
    }
    bar.finish();
    return latent_vector_1d_new;
}

std::vector<std::pair<std::string, float>> parse_prompt_attention(std::string& texts) {
    std::vector<std::pair<std::string, float>> res;
    std::stack<int> round_brackets;
    std::stack<int> square_brackets;
    const float round_bracket_multiplier = 1.1;
    const float square_bracket_multiplier = 1 / 1.1;
    std::vector<std::string> ms;

    for (char c : texts) {
        std::string s = std::string(1, c);

        if (s == "(" || s == "[" || s == ")" || s == "]") {
            ms.push_back(s);
        }

        else {
            if (ms.size() < 1)
                ms.push_back("");

            std::string last = ms[ms.size() - 1];

            if (last == "(" || last == "[" || last == ")" || last == "]") {
                ms.push_back("");
            }

            ms[ms.size() - 1] += s;
        }
    }

    for (std::string text : ms) {
        if (text == "(") {
            round_brackets.push(res.size());
        }

        else if (text == "[") {
            square_brackets.push(res.size());
        }

        else if (text == ")" && round_brackets.size() > 0) {
            for (unsigned long p = round_brackets.top(); p < res.size(); p++) {
                res[p].second *= round_bracket_multiplier;
            }

            round_brackets.pop();
        }

        else if (text == "]" && square_brackets.size() > 0) {
            for (unsigned long p = square_brackets.top(); p < res.size(); p++) {
                res[p].second *= square_bracket_multiplier;
            }

            square_brackets.pop();
        }

        else {
            res.push_back(make_pair(text, 1.0));
        }
    }

    while (!round_brackets.empty()) {
        for (unsigned long p = round_brackets.top(); p < res.size(); p++) {
            res[p].second *= round_bracket_multiplier;
        }

        round_brackets.pop();
    }

    while (!square_brackets.empty()) {
        for (unsigned long p = square_brackets.top(); p < res.size(); p++) {
            res[p].second *= square_bracket_multiplier;
        }

        square_brackets.pop();
    }

    unsigned long i = 0;

    while (i + 1 < res.size()) {
        if (res[i].second == res[i + 1].second) {
            res[i].first += res[i + 1].first;
            auto it = res.begin();
            res.erase(it + i + 1);
        }

        else {
            i += 1;
        }
    }

    return res;
}

// splid and add </w>
std::vector<std::string> split(std::string str) {
    // ys: erase all the "."
    while (str.find(".") != std::string::npos) {
        str.erase(str.find("."), 1);
    }
    logger.log_value(LogLevel::DEBUG, "DEBUG-erased prompt: ", str);

    std::string::size_type pos;
    std::vector<std::string> result;
    str += " ";
    int32_t size = str.size();

    for (int32_t i = 0; i < size; i++) {
        pos = std::min(str.find(" ", i), str.find(",", i));

        if (pos < str.size()) {
            std::string s = str.substr(i, pos - i);
            std::string pat = std::string(1, str[pos]);

            if (s.length() > 0)
                result.push_back(s + "</w>");

            if (pat != " ")
                result.push_back(pat + "</w>");

            i = pos;
        }
    }

    return result;
}

std::vector<std::vector<int32_t>> tokenizer_infer_function(ov::CompiledModel& tokenizer_model, std::string prompt) {
    logger.log_value(LogLevel::DEBUG, "DEBUG-prompt: ", prompt);

    auto tokenizer_encoder_infer_request = tokenizer_model.create_infer_request();

    int32_t batch_size = 1;
    int32_t offset = 0;
    int32_t length = prompt.length();

    int32_t eos = 49407;
    int32_t max_token_length = 77;
    int32_t default_input_ids = eos;
    int32_t default_attention_mask = 0;

    unsigned long total_byte_length = 4 + 4 + 4 + prompt.length();
    // unsigned char bytes[total_byte_length];
    std::vector<uint8_t> bytes_vec(total_byte_length);
    StringToByteArray_vec(prompt, bytes_vec, batch_size, offset, length);
    // std::string str = ByteArrayToString(bytes+4+4+4, length);
    // std::cout << "str: " << str << "\n";

    // Get input port for model with one input
    auto tokenizer_encoder_input_port = tokenizer_model.input();
    logger.log_value(LogLevel::DEBUG,
                     "tokenizer_encoder_input_port.get_partial_shape(): ",
                     tokenizer_encoder_input_port.get_partial_shape());

    auto tokenizer_encoder_output_ports = tokenizer_model.outputs();

    ov::Tensor input_tensor(tokenizer_encoder_input_port.get_element_type(), {total_byte_length}, bytes_vec.data());

    // Set input tensor for model with one input
    tokenizer_encoder_infer_request.set_input_tensor(input_tensor);
    // tokenizer_encoder_infer_request.start_async();
    // tokenizer_encoder_infer_request.wait();
    tokenizer_encoder_infer_request.infer();

    // Get output tensor by tensor name
    auto tokenizer_encoder_output_1 = tokenizer_encoder_infer_request.get_tensor(tokenizer_encoder_output_ports[0]);
    auto tokenizer_encoder_output_2 = tokenizer_encoder_infer_request.get_tensor(tokenizer_encoder_output_ports[1]);

    logger.log_value(LogLevel::DEBUG, "tokenzier_output_1.get_shape(): ", tokenizer_encoder_output_1.get_shape());
    logger.log_value(LogLevel::DEBUG, "tokenzier_output_2.get_shape(): ", tokenizer_encoder_output_2.get_shape());

    const int32_t* tokenizer_encoder_output_buffer_1 = tokenizer_encoder_output_1.data<const int32_t>();
    const int32_t* tokenizer_encoder_output_buffer_2 = tokenizer_encoder_output_2.data<const int32_t>();

    std::vector<int32_t> input_ids(max_token_length, default_input_ids);
    std::vector<int32_t> attention_mask(max_token_length, default_attention_mask);

    // std::vector<int32_t> input_ids(output_buffer_1, output_buffer_1 + tokenizer_output_1.get_shape()[1]);
    // std::vector<int32_t> attention_mask(output_buffer_2, output_buffer_2 + tokenizer_output_2.get_shape()[1]);
    std::copy(tokenizer_encoder_output_buffer_1,
              tokenizer_encoder_output_buffer_1 + tokenizer_encoder_output_1.get_shape()[1],
              input_ids.begin());

    std::copy(tokenizer_encoder_output_buffer_2,
              tokenizer_encoder_output_buffer_2 + tokenizer_encoder_output_2.get_shape()[1],
              attention_mask.begin());

    return std::vector<std::vector<int32_t>>{input_ids};
}

std::vector<float> clip_infer_function(ov::CompiledModel& prompt_model, std::vector<int32_t> current_tokens)

{
    ov::InferRequest infer_request = prompt_model.create_infer_request();
    auto clip_input_port = prompt_model.input();
    auto shape = clip_input_port.get_partial_shape();
    logger.log_value(LogLevel::DEBUG, "clip_input_port.get_partial_shape(): ", shape);
    ov::Shape clip_input_shape = {1, current_tokens.size()};
    ov::Tensor text_embeddings_input_tensor(clip_input_port.get_element_type(),
                                            clip_input_shape,
                                            current_tokens.data());
    infer_request.set_tensor(clip_input_port, text_embeddings_input_tensor);
    // infer_request.start_async();
    // infer_request.wait();
    infer_request.infer();

    auto output_port_0 = prompt_model.outputs()[0];

    ov::Tensor text_embeddings_tensor = infer_request.get_tensor(output_port_0);
    auto text_em_ptr = text_embeddings_tensor.data<float>();
    std::vector<float> text_embeddings;
    for (size_t i = 0; i < 77 * 768; i++) {
        text_embeddings.push_back(text_em_ptr[i]);
    }
    logger.log_vector(LogLevel::DEBUG, "text_embeddings: ", text_embeddings, 0, 5);

    return text_embeddings;
}

std::vector<int32_t> pre_process_function(std::unordered_map<std::string, int>& tokenizer_token2idx,
                                          std::string prompt) {
    // parse attention, `()` to improve and `[]` to reduce
    std::vector<std::pair<std::string, float>> parsed = parse_prompt_attention(prompt);
    logger.log_value(LogLevel::DEBUG, "DEBUG-prompt: ", prompt);

    // token2ids
    std::vector<std::vector<int>> tokenized;
    {
        for (auto p : parsed) {
            std::vector<std::string> tokens = split(p.first);
            std::vector<int> ids;

            for (std::string token : tokens) {
                // vocab.txt is lower only
                std::transform(token.begin(), token.end(), token.begin(), ::tolower);
                ids.push_back(tokenizer_token2idx[token]);
                // std::cout << "DEBUG: " << token << " - " << tokenizer_token2idx[token] << std::endl;
            }
            tokenized.push_back(ids);
        }
    }

    logger.log_value(LogLevel::DEBUG, "DEBUG-tokenized.size(): ", tokenized[0].size());

    std::vector<int> remade_tokens;
    std::vector<float> multipliers;
    {
        int32_t last_comma = -1;

        for (unsigned long it_tokenized = 0; it_tokenized < tokenized.size(); it_tokenized++) {
            std::vector<int> tokens = tokenized[it_tokenized];
            float weight = parsed[it_tokenized].second;
            unsigned long i = 0;

            while (i < tokens.size()) {
                int32_t token = tokens[i];

                if (token == 267) {
                    last_comma = remade_tokens.size();
                }

                else if ((std::max(int(remade_tokens.size()), 1) % 75 == 0) && (last_comma != -1) &&
                         (remade_tokens.size() - last_comma <= 20)) {
                    last_comma += 1;
                    std::vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
                    std::vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
                    std::vector<int> _remade_tokens_(remade_tokens.begin(), remade_tokens.begin() + last_comma);
                    remade_tokens = _remade_tokens_;
                    int32_t length = remade_tokens.size();
                    int32_t rem = std::ceil(length / 75.0) * 75 - length;
                    std::vector<int> tmp_token(rem, 49407);
                    remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
                    remade_tokens.insert(remade_tokens.end(), reloc_tokens.begin(), reloc_tokens.end());
                    std::vector<float> _multipliers_(multipliers.begin(), multipliers.end() + last_comma);
                    std::vector<int> tmp_multipliers(rem, 1.0f);
                    _multipliers_.insert(_multipliers_.end(), tmp_multipliers.begin(), tmp_multipliers.end());
                    _multipliers_.insert(_multipliers_.end(), reloc_mults.begin(), reloc_mults.end());
                    multipliers = _multipliers_;
                }

                remade_tokens.push_back(token);
                multipliers.push_back(weight);
                i += 1;
            }
        }

        int32_t prompt_target_length = std::ceil(std::max(int(remade_tokens.size()), 1) / 75.0) * 75;
        int32_t tokens_to_add = prompt_target_length - remade_tokens.size();
        std::vector<int> tmp_token(tokens_to_add, 49407);
        remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
        std::vector<int> tmp_multipliers(tokens_to_add, 1.0f);
        multipliers.insert(multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());
    }

    std::vector<int32_t> current_tokens;
    current_tokens.insert(current_tokens.begin(), 49406);
    current_tokens.insert(current_tokens.begin() + 1, remade_tokens.begin(), remade_tokens.end());
    current_tokens.insert(current_tokens.end(), 49407);
    logger.log_vector(LogLevel::DEBUG, "DEBUG-current_tokens.values: ", current_tokens, 0, 77);

    return current_tokens;
}

std::vector<float> prompt_function(ov::CompiledModel& text_encoder_compiled_model,
                                   std::string const& prompt_positive_str,
                                   std::string const& prompt_negative_str) {
    // read vocab

    std::unordered_map<std::string, int> tokenizer_token2idx;
    {
        std::ifstream vocab_file;
        std::string vocab_path = "../models/vocab.txt";
        vocab_file.open(vocab_path.data());
        if (vocab_file.is_open()) {
            std::string s;
            int32_t idx = 0;

            while (getline(vocab_file, s)) {
                tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
                idx++;
            }
        } else {
            std::cout << "could not find the vocab.txt" << std::endl;
            exit(0);
        }
        vocab_file.close();
        // std::cout << "DEBUG-tokenizer_token2idx.size(): " << tokenizer_token2idx.size() << std::endl;
    }

    // without OVTokenizer
    std::vector<int32_t> prompt_positive_vec = pre_process_function(tokenizer_token2idx, prompt_positive_str);
    std::vector<int32_t> prompt_negative_vec = pre_process_function(tokenizer_token2idx, prompt_negative_str);

    std::vector<float> text_embeddings_pos = clip_infer_function(text_encoder_compiled_model, prompt_positive_vec);
    std::vector<float> text_embeddings_neg = clip_infer_function(text_encoder_compiled_model, prompt_negative_vec);
    text_embeddings_neg.insert(text_embeddings_neg.end(), text_embeddings_pos.begin(), text_embeddings_pos.end());

    logger.log_value(LogLevel::DEBUG, "DEBUG-text_embeddings_pos.size(): ", text_embeddings_neg.size());  // 118272

    return text_embeddings_neg;
}

std::vector<ov::CompiledModel> SD_init(const std::string& model_path,
                                       const std::string& device,
                                       const std::string& type,
                                       const std::map<std::string, float>& lora_models,
                                       bool use_ov_extension,
                                       bool use_cache) {
    ov::Core core;

    if (use_cache) {
        core.set_property(ov::cache_dir("./cache_dir"));
    }

    std::vector<ov::CompiledModel> SD_compiled_models;

    std::shared_ptr<ov::Model> text_encoder_model =
        core.read_model((model_path + "/" + type + "/text_encoder/openvino_model.xml").c_str());
    std::shared_ptr<ov::Model> unet_model =
        core.read_model((model_path + "/" + type + "/unet/openvino_model.xml").c_str());
    std::shared_ptr<ov::Model> decoder_model =
        core.read_model((model_path + "/" + type + "/vae_decoder/openvino_model.xml").c_str());

    SD_compiled_models = load_lora_weights_cpp(core, text_encoder_model, unet_model, device, lora_models);

    ov::preprocess::PrePostProcessor ppp(decoder_model);
    ppp.output().model().set_layout("NCHW");
    ppp.output().tensor().set_layout("NHWC");
    decoder_model = ppp.build();

    ov::CompiledModel decoder_compiled_model = core.compile_model(decoder_model, device);
    SD_compiled_models.push_back(decoder_compiled_model);

    if (use_ov_extension) {
        std::string tokenizer_encoder_model_path = "../models/tokenizer/tokenizer_encoder.xml"; 
        std::cout << "Initialize SDTokenizer Model from path: " << tokenizer_encoder_model_path << std::endl;
        core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in CMakeLists.txt
        ov::CompiledModel compiled_tokenizer_encoder = core.compile_model(tokenizer_encoder_model_path, device);
        SD_compiled_models.push_back(compiled_tokenizer_encoder);
    }

    return SD_compiled_models;
}

void stable_diffusion(const std::string& positive_prompt = std::string{},
                      const std::vector<std::string>& output_vec = {},
                      const std::string& device = std::string{},
                      int32_t step = 20,
                      const std::vector<uint32_t>& seed_vec = {},
                      uint32_t num = 1,
                      uint32_t height = 512,
                      uint32_t width = 512,
                      std::string negative_prompt = std::string{},
                      bool use_logger = false,
                      bool use_cache = false,
                      const std::string& model_path = std::string{},
                      const std::string& type = std::string{},
                      const std::string& lora_path = std::string{},
                      float alpha = 0.75,
                      bool use_ov_extension = false,
                      bool read_np_latent = false) {
    logger.setLoggingEnabled(use_logger);
    logger.log_time(LogLevel::DEBUG);
    logger.log_string(LogLevel::DEBUG, "Welcome to use Stable-Diffusion-OV.");
    logger.log_string(LogLevel::INFO, "----------------[start]------------------");
    logger.log_value(LogLevel::INFO, "positive_prompt: ", positive_prompt);
    logger.log_value(LogLevel::INFO, "negative_prompt: ", negative_prompt);
    logger.log_value(LogLevel::INFO, "Device: ", device);
    logger.log_value(LogLevel::INFO, "step: ", step);
    logger.log_value(LogLevel::INFO, "num: ", num);
    logger.log_value(LogLevel::INFO, "height: ", height);
    logger.log_value(LogLevel::INFO, "width: ", width);
    logger.log_value(LogLevel::INFO, "model_path: ", model_path);
    logger.log_value(LogLevel::INFO, "type: ", type);
    logger.log_value(LogLevel::INFO, "lora_path: ", lora_path);
    logger.log_value(LogLevel::INFO, "use_ov_extension: ", use_ov_extension);
    logger.log_value(LogLevel::INFO, "read_np_latent: ", read_np_latent);
    logger.log_value(LogLevel::INFO, "use_logger: ", use_logger);
    logger.log_value(LogLevel::INFO, "use_cache: ", use_cache);

    std::cout << "----------------[start]------------------" << std::endl;

    std::cout << "openvino version: " << ov::get_openvino_version() << std::endl;
    std::cout << "positive_prompt: " << positive_prompt << std::endl;
    std::cout << "negative_prompt: " << negative_prompt << std::endl;
    std::cout << "Device: " << device << std::endl;
    std::cout << "output_png_path: ./build/images/" << std::endl;
    std::cout << "step: " << step << std::endl;
    std::cout << "num: " << num << std::endl;
    std::cout << "height: " << height << std::endl;
    std::cout << "width: " << width << std::endl;
    std::cout << "model_path: " << model_path << std::endl;
    std::cout << "type: " << type << std::endl;
    std::cout << "lora_path: " << lora_path << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << std::boolalpha << "use_ov_extension: " << use_ov_extension << std::endl;
    std::cout << std::boolalpha << "read_np_latent: " << read_np_latent << std::endl;
    std::cout << std::boolalpha << "use_logger: " << use_logger << std::endl;
    std::cout << std::boolalpha << "use_cache: " << use_cache << std::endl;

    logger.log_string(LogLevel::INFO, "----------------[Model Init]------------------");
    std::cout << "----------------[Model Init]------------------" << std::endl;

    std::map<std::string, float> lora_models;
    lora_models.insert(std::pair<std::string, float>(lora_path, alpha));

    auto start_SDinit = std::chrono::steady_clock::now();
    std::vector<ov::CompiledModel> SD_models =
        SD_init(model_path, device, type, lora_models, use_ov_extension, use_cache);
    auto end_SDinit = std::chrono::steady_clock::now();
    auto duration_SDinit = std::chrono::duration_cast<std::chrono::duration<float>>(end_SDinit - start_SDinit);

    logger.log_value(LogLevel::DEBUG, "duration of SD_init(s): ", duration_SDinit.count());
    auto start_tokenizer = std::chrono::steady_clock::now();

    std::vector<float> text_embeddings;
    if (use_ov_extension) {
        // OVTokenizer (WIP)
        logger.log_string(LogLevel::INFO, "----------------[tokenizer]------------------");
        std::cout << "----------------[tokenizer]------------------" << std::endl;
        std::vector<std::vector<int32_t>> pos_infered_token = tokenizer_infer_function(SD_models[3], positive_prompt);
        std::vector<std::vector<int32_t>> neg_infered_token = tokenizer_infer_function(SD_models[3], negative_prompt);

        logger.log_string(LogLevel::INFO, "----------------[text embedding]------------------");
        std::cout << "----------------[text embedding]------------------" << std::endl;

        // auto start_clip = std::chrono::steady_clock::now();
        std::vector<float> text_embeddings_pos = clip_infer_function(SD_models[0], pos_infered_token[0]);
        std::vector<float> text_embeddings_neg = clip_infer_function(SD_models[0], neg_infered_token[0]);
        text_embeddings = std::vector<float>(text_embeddings_neg);
        text_embeddings.insert(text_embeddings.end(), text_embeddings_pos.begin(), text_embeddings_pos.end());
        // auto end_clip = std::chrono::steady_clock::now();
        // auto duration_clip = std::chrono::duration_cast<std::chrono::duration<float>>(end_clip - start_clip);
        // std::cout << "duration (pos + neg prompt): " << duration_clip.count() << " s" << std::endl;
    } else {
        // not use OVTokenizer
        logger.log_string(LogLevel::INFO, "----------------[prompt_function]------------------");
        std::cout << "----------------[prompt_function]------------------" << std::endl;
        text_embeddings = prompt_function(SD_models[0], positive_prompt, negative_prompt);
    }

    auto end_tokenizer = std::chrono::steady_clock::now();
    auto duration_tokenizer = std::chrono::duration_cast<std::chrono::duration<float>>(end_tokenizer - start_tokenizer);
    std::cout << "duration (pos + neg prompt): " << duration_tokenizer.count() << " s" << std::endl;

    logger.log_string(LogLevel::INFO, "----------------[diffusion]------------------");
    std::cout << "----------------[diffusion]---------------" << std::endl;

    for (uint32_t n = 0; n < num; n++) {
        logger.log_value(LogLevel::INFO, "seed: ", seed_vec[n]);
        std::cout << "image No." << n << ", seed = " << seed_vec[n] << std::endl;

        std::vector<float> latent_vector_1d;
        if (read_np_latent) {
            latent_vector_1d = np_randn_function();
        } else {
            latent_vector_1d = std_randn_function(seed_vec[n], height, width);
        }
        logger.log_vector(LogLevel::DEBUG, "randn output: ", latent_vector_1d, 0, 20);

        auto start_diffusion = std::chrono::steady_clock::now();
        auto sample =
            diffusion_function(SD_models[1], seed_vec[n], step, height, width, latent_vector_1d, text_embeddings);
        auto end_diffusion = std::chrono::steady_clock::now();
        auto duration_diffusion =
            std::chrono::duration_cast<std::chrono::duration<float>>(end_diffusion - start_diffusion);
        std::cout << "duration (all " << step << " steps): " << duration_diffusion.count()
                  << " s, each step: " << duration_diffusion.count() / step << " s" << std::endl;

        logger.log_string(LogLevel::INFO, "----------------[decode]------------------");
        std::cout << "----------------[decode]------------------" << std::endl;
        auto start_decode = std::chrono::steady_clock::now();
        auto output_decoder = vae_decoder_function(SD_models[2], sample, height, width);
        auto end_decode = std::chrono::steady_clock::now();
        auto duration_decode = std::chrono::duration_cast<std::chrono::duration<float>>(end_decode - start_decode);
        std::cout << "duration: " << duration_decode.count() << " s" << std::endl;

        logger.log_string(LogLevel::INFO, "----------------[save]------------------");
        std::cout << "----------------[save]--------------------" << std::endl;
        auto start_save = std::chrono::steady_clock::now();

        std::vector<uint8_t> output_decoder_int = std::vector<uint8_t>(output_decoder.begin(), output_decoder.end());

        convertBGRtoRGB(output_decoder_int, width, height);

        writeOutputBmp(output_vec[n], output_decoder_int.data(), height, width);

        auto end_save = std::chrono::steady_clock::now();
        auto duration_save = std::chrono::duration_cast<std::chrono::duration<float>>(end_save - start_save);

        auto duration_total = std::chrono::duration_cast<std::chrono::duration<float>>(end_decode - start_diffusion +
                                                                                       end_tokenizer - start_tokenizer);
        std::cout << "duration of one image generation without model compiling: " << duration_total.count() << " s\n\n"
                  << std::endl;
    }

    logger.log_string(LogLevel::INFO, "----------------[close]------------------");
    std::cout << "----------------[close]-------------------" << std::endl;
}