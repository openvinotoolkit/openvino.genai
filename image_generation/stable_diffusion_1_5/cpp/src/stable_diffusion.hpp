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
#include <random>
#include <regex>
#include <stack>
#include <string>
#include <unordered_map>
#include <utils.hpp>
#include <vector>

#include "openvino/openvino.hpp"

#include <openvino_extensions/strings.hpp>

#include "lora_cpp.hpp"

#include "logger.hpp"
#include "progress_bar.hpp"
#include "imwrite.hpp"

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

    ProgressBar bar(sigma.size());

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

std::vector<std::vector<int32_t>> tokenizer_infer_function(ov::CompiledModel& tokenizer_model, std::string prompt) {
    constexpr int32_t BATCH_SIZE = 1, MAX_LENGTH = 77, EOS = 49407, DEFAULT_INPUT_IDS = EOS;
    const ov::Shape input_ids_shape({BATCH_SIZE, MAX_LENGTH});

    auto tokenizer_request = tokenizer_model.create_infer_request();
    auto input_ids_tensor = tokenizer_request.get_tensor("input_ids");

    // we need to pre-fill 'input_ids' with default tokens value
    input_ids_tensor.set_shape(input_ids_shape);
    std::fill_n(input_ids_tensor.data<int32_t>(), MAX_LENGTH, DEFAULT_INPUT_IDS);

    ov::Tensor packed_string = tokenizer_request.get_input_tensor();
    openvino_extensions::pack_strings(std::array<std::string, BATCH_SIZE>{prompt}, tokenizer_request.get_input_tensor());

    tokenizer_request.infer();

    // restore shape to CLIP expected input shape
    input_ids_tensor.set_shape(input_ids_shape);

    const int32_t* input_ids_data = input_ids_tensor.data<const int32_t>();
    std::vector<int32_t> input_ids(input_ids_tensor.get_shape()[1]);

    std::copy(input_ids_data,
              input_ids_data + input_ids.size(),
              input_ids.begin());

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

struct StableDiffusionModels {
    ov::CompiledModel text_encoder;
    ov::CompiledModel unet;
    ov::CompiledModel vae_decoder;
    ov::CompiledModel tokenizer;
};

StableDiffusionModels compile_models(const std::string& model_path,
                                     const std::string& device,
                                     const std::string& type,
                                     const std::string& lora_path,
                                     float alpha,
                                     bool use_cache) {
    StableDiffusionModels models;

    ov::Core core;
    if (use_cache)
        core.set_property(ov::cache_dir("./cache_dir"));
    core.add_extension(TOKENIZERS_LIBRARY_PATH);

    std::map<std::string, float> lora_models;
    lora_models[lora_path] = alpha;

    // CLIP and UNet
    auto text_encoder_model = core.read_model((model_path + "/" + type + "/text_encoder/openvino_model.xml").c_str());
    auto unet_model = core.read_model((model_path + "/" + type + "/unet/openvino_model.xml").c_str());
    auto text_encoder_and_unet = load_lora_weights_cpp(core, text_encoder_model, unet_model, device, lora_models);
    models.text_encoder = text_encoder_and_unet[0];
    models.unet = text_encoder_and_unet[1];

    // VAE decoder
    auto vae_decoder_model = core.read_model((model_path + "/" + type + "/vae_decoder/openvino_model.xml").c_str());
    ov::preprocess::PrePostProcessor ppp(vae_decoder_model);
    ppp.output().model().set_layout("NCHW");
    ppp.output().tensor().set_layout("NHWC");
    models.vae_decoder = core.compile_model(vae_decoder_model = ppp.build(), device);

    // tokenizer
    std::string tokenizer_model_path = "../models/tokenizer/tokenizer_encoder.xml";
    models.tokenizer = core.compile_model(tokenizer_model_path, device);

    return models;
}

void stable_diffusion(const std::string& positive_prompt = std::string{},
                      const std::vector<std::string>& output_images = {},
                      const std::string& device = std::string{},
                      int32_t steps = 20,
                      const std::vector<uint32_t>& seed_vec = {},
                      uint32_t num_images = 1,
                      uint32_t height = 512,
                      uint32_t width = 512,
                      std::string negative_prompt = std::string{},
                      bool use_logger = false,
                      bool use_cache = false,
                      const std::string& model_base_path = std::string{},
                      const std::string& model_type = std::string{},
                      const std::string& lora_path = std::string{},
                      float alpha = 0.75f,
                      bool read_np_latent = false) {
    auto start_SDinit = std::chrono::steady_clock::now();
    StableDiffusionModels models = compile_models(model_base_path, device, model_type, lora_path, alpha, use_cache);
    auto duration_SDinit = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::steady_clock::now() - start_SDinit);

    logger.log_value(LogLevel::DEBUG, "duration of SD_init(s): ", duration_SDinit.count());
    auto start_tokenizer = std::chrono::steady_clock::now();

    std::vector<float> text_embeddings;
    logger.log_string(LogLevel::INFO, "----------------[tokenizer]------------------");
    std::cout << "----------------[tokenizer]------------------" << std::endl;
    std::vector<std::vector<int32_t>> pos_infered_token = tokenizer_infer_function(models.tokenizer, positive_prompt);
    std::vector<std::vector<int32_t>> neg_infered_token = tokenizer_infer_function(models.tokenizer, negative_prompt);

    logger.log_string(LogLevel::INFO, "----------------[text embedding]------------------");
    std::cout << "----------------[text embedding]------------------" << std::endl;

    std::vector<float> text_embeddings_pos = clip_infer_function(models.text_encoder, pos_infered_token[0]);
    std::vector<float> text_embeddings_neg = clip_infer_function(models.text_encoder, neg_infered_token[0]);
    text_embeddings = std::vector<float>(text_embeddings_neg);
    text_embeddings.insert(text_embeddings.end(), text_embeddings_pos.begin(), text_embeddings_pos.end());

    auto end_tokenizer = std::chrono::steady_clock::now();
    auto duration_tokenizer = std::chrono::duration_cast<std::chrono::duration<float>>(end_tokenizer - start_tokenizer);
    std::cout << "duration (pos + neg prompt): " << duration_tokenizer.count() << " s" << std::endl;

    logger.log_string(LogLevel::INFO, "----------------[diffusion]------------------");
    std::cout << "----------------[diffusion]---------------" << std::endl;

    for (uint32_t n = 0; n < num_images; n++) {
        logger.log_value(LogLevel::INFO, "seed: ", seed_vec[n]);
        std::cout << "image No." << n << ", seed = " << seed_vec[n] << std::endl;

        std::vector<float> latent_vector_1d = read_np_latent ? np_randn_function() : std_randn_function(seed_vec[n], height, width);
        logger.log_vector(LogLevel::DEBUG, "randn output: ", latent_vector_1d, 0, 20);

        auto start_diffusion = std::chrono::steady_clock::now();
        auto sample = diffusion_function(models.unet, seed_vec[n], steps, height, width, latent_vector_1d, text_embeddings);
        auto end_diffusion = std::chrono::steady_clock::now();
        auto duration_diffusion =
            std::chrono::duration_cast<std::chrono::duration<float>>(end_diffusion - start_diffusion);
        std::cout << "duration (all " << steps << " steps): " << duration_diffusion.count()
                  << " s, each step: " << duration_diffusion.count() / steps << " s" << std::endl;

        logger.log_string(LogLevel::INFO, "----------------[decode]------------------");
        std::cout << "----------------[decode]------------------" << std::endl;
        auto start_decode = std::chrono::steady_clock::now();
        auto output_decoder = vae_decoder_function(models.vae_decoder, sample, height, width);
        auto end_decode = std::chrono::steady_clock::now();
        auto duration_decode = std::chrono::duration_cast<std::chrono::duration<float>>(end_decode - start_decode);
        std::cout << "duration: " << duration_decode.count() << " s" << std::endl;

        logger.log_string(LogLevel::INFO, "----------------[save]------------------");
        std::cout << "----------------[save]--------------------" << std::endl;
        auto start_save = std::chrono::steady_clock::now();

        std::vector<uint8_t> output_decoder_int = std::vector<uint8_t>(output_decoder.begin(), output_decoder.end());

        convertBGRtoRGB(output_decoder_int, width, height);

        imwrite(output_images[n], output_decoder_int.data(), height, width);

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
