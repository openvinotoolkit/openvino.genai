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

ov::Tensor sigma_to_t(std::vector<float>& log_sigmas, float sigma) {
    ov::Tensor timestemp(ov::element::i64, {1});

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

    timestemp.data<int64_t>()[0] = std::llround((1 - w) * low_idx + w * high_idx);

    return timestemp;
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct StableDiffusionModels {
    ov::CompiledModel text_encoder;
    ov::CompiledModel unet;
    ov::CompiledModel vae_decoder;
    ov::CompiledModel tokenizer;
};

std::vector<uint8_t> vae_decoder_function(ov::CompiledModel& decoder_compiled_model,
                                          ov::Tensor sample,
                                          uint32_t h, uint32_t w) {
    auto decoder_input_port = decoder_compiled_model.input();
    auto decoder_output_port = decoder_compiled_model.output();
    auto shape = decoder_input_port.get_partial_shape();
    logger.log_value(LogLevel::DEBUG, "decoder_input_port.get_partial_shape(): ", shape);

    const ov::element::Type type = decoder_input_port.get_element_type();

    float coeffs_const{1 / 0.18215};
    for (size_t i = 0; i < sample.get_size(); ++i) {
        sample.data<float>()[i] *= coeffs_const;
    }

    ov::Shape sample_shape = {1, 4, h / 8, w / 8};
    ov::Tensor decoder_input_tensor(type, sample_shape, sample.data());

    std::cout << "samples shape " << sample.get_shape() << std::endl;
    std::cout << "sample_shape " << sample_shape << std::endl;

    ov::InferRequest infer_request = decoder_compiled_model.create_infer_request();
    infer_request.set_tensor(decoder_input_port, decoder_input_tensor);
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

ov::Tensor unet_infer_function(ov::CompiledModel& unet_model,
                               ov::Tensor timestemp,
                               ov::Tensor latent_input_1d,
                               ov::Tensor text_embedding_1d,
                               const uint32_t u_h,
                               const uint32_t u_w) {
    ov::InferRequest unet_infer_request = unet_model.create_infer_request();

    const uint32_t latent_h = u_h / 8;
    const uint32_t latent_w = u_w / 8;

    unet_infer_request.set_tensor("sample", latent_input_1d);
    unet_infer_request.set_tensor("timestep", timestemp);
    unet_infer_request.set_tensor("encoder_hidden_states", text_embedding_1d);

    unet_infer_request.infer();

    ov::Tensor noise_pred_tensor = unet_infer_request.get_output_tensor();
    ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
    noise_pred_shape[0] = 1;

    const float guidance_scale = 7.5f;
    auto noise_pred_uncond = noise_pred_tensor.data<const float>();
    auto noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

    ov::Tensor noise_pred(noise_pred_tensor.get_element_type(), noise_pred_shape);
    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i)
        noise_pred.data<float>()[i] = noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);

    return noise_pred;
}

ov::Tensor diffusion_function(ov::CompiledModel unet_compiled_model,
                              uint32_t seed,
                              int32_t step,
                              uint32_t d_h,
                              uint32_t d_w,
                              ov::Tensor latent_vector_1d,
                              ov::Tensor text_embeddings_2_77_768) {
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

    // LMSDiscreteScheduler: latents are multiplied by sigmas
    double n = sigma[0];  // 14.6146

    ov::Shape latent_shape = latent_vector_1d.get_shape(), latent_model_input_shape = latent_shape;
    latent_model_input_shape[0] = 2; // Unet accepts batch 2
    const ov::element::Type latent_type = latent_vector_1d.get_element_type();
    ov::Tensor latent_vector_1d_new(latent_type, latent_shape), latent_model_input(latent_type, latent_model_input_shape);
    for (size_t i = 0; i < latent_vector_1d.get_size(); ++i) {
        latent_vector_1d_new.data<float>()[i]  = latent_vector_1d.data<float>()[i] * n;
    }

    std::vector<std::vector<float>> derivative_list;

    ProgressBar bar(sigma.size());

    for (size_t i = 0; i < step; i++) {
        bar.progress(i);

        // 'sample'
        double scale = 1.0 / sqrt((sigma[i] * sigma[i] + 1));
        for (size_t j = 0, lanent_size = latent_vector_1d_new.get_size(); j < lanent_size; j++) {
            latent_model_input.data<float>()[j + lanent_size] = latent_model_input.data<float>()[j] = latent_vector_1d_new.data<float>()[j] * scale;
        }

        // 'timestep'
        ov::Tensor timestemp = sigma_to_t(log_sigma_vec, sigma[i]);

        ov::Tensor noise_pred_1d = unet_infer_function(unet_compiled_model, timestemp, latent_model_input, text_embeddings_2_77_768, d_h, d_w);

        // LMS step function:
        std::vector<float> derivative_vec_1d;
        size_t order = 4;
        for (size_t j = 0; j < latent_vector_1d.get_size(); j++) {
            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise defaut "epsilon"
            float pred_latent = latent_vector_1d.data<float>()[j] - sigma[i] * noise_pred_1d.data<float>()[j];
            // 2. Convert to an ODE derivative
            derivative_vec_1d.push_back((latent_vector_1d.data<float>()[j] - pred_latent) / sigma[i]);
        }

        derivative_list.push_back(derivative_vec_1d);
        // keep the list size within 4
        if (order < derivative_list.size()) {
            derivative_list.erase(derivative_list.begin());
        }

        // 3. Compute linear multistep coefficients
        order = std::min(i + 1, order);

        std::vector<float> lms_coeffs;
        for (int32_t curr_order = 0; curr_order < order; curr_order++) {
            auto functor = [order, curr_order, sigma, i](float tau) {
                return lms_derivative_function(tau, order, curr_order, sigma, i);
            };
            auto integrated_coeff_new = trapezoidal(functor, static_cast<double>(sigma[i]), static_cast<double>(sigma[i + 1]), 1e-4);
            lms_coeffs.push_back(integrated_coeff_new);
        }

        // 4. Compute previous sample based on the derivatives path
        // prev_sample = sample + sum(coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives)))
        // Reverse list of tensors this.derivatives
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
                       latent_vector_1d_new.data<float>(),
                       latent_vector_1d_new.data<float>(),
                       [](float x, float y) {
                           return x + y;
                       });
    }
    bar.finish();
    return latent_vector_1d_new;
}

std::vector<float> text_encoder_infer_function(StableDiffusionModels models, const std::string& prompt) {
    const size_t MAX_LENGTH = models.text_encoder.input().get_shape()[1], BATCH_SIZE = 1;
    const ov::Shape input_ids_shape({BATCH_SIZE, MAX_LENGTH});

    // Tokenization

    ov::InferRequest tokenizer_req = models.tokenizer.create_infer_request();
    ov::Tensor input_ids_tensor = tokenizer_req.get_tensor("input_ids");

    input_ids_tensor.set_shape(input_ids_shape);
    std::fill_n(input_ids_tensor.data<int32_t>(), ov::shape_size(input_ids_shape), 49407);

    ov::Tensor packed_strings = tokenizer_req.get_input_tensor();
    openvino_extensions::pack_strings(std::array<std::string, BATCH_SIZE>{prompt}, packed_strings);

    tokenizer_req.infer();
    // restore shape to CLIP expected input shape
    input_ids_tensor.set_shape(input_ids_shape);

    // Text embedding

    std::vector<float> text_embeddings;
    text_embeddings.reserve(ov::shape_size(input_ids_shape));

    ov::InferRequest text_encoder_req = models.text_encoder.create_infer_request();
    text_encoder_req.set_input_tensor(input_ids_tensor);
    text_encoder_req.infer();

    ov::Tensor text_embeddings_tensor = text_encoder_req.get_output_tensor(0);
    for (size_t i = 0; i < text_embeddings_tensor.get_size(); i++)
        text_embeddings.push_back(text_embeddings_tensor.data<const float>()[i]);

    return text_embeddings;
}

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
    StableDiffusionModels models = compile_models(model_base_path, device, model_type, lora_path, alpha, use_cache);

    std::vector<float> text_embeddings_pos = text_encoder_infer_function(models, positive_prompt);
    std::vector<float> text_embeddings = text_encoder_infer_function(models, negative_prompt);
    std::copy_n(text_embeddings_pos.begin(), text_embeddings_pos.size(), std::back_inserter(text_embeddings));
    ov::Tensor text_embeddings_(ov::element::f32, {2,77,768}, text_embeddings.data());

    for (uint32_t n = 0; n < num_images; n++) {
        std::vector<float> latent_vector_1d = read_np_latent ? np_randn_function() : std_randn_function(seed_vec[n], height, width);
        ov::Shape latent_shape = models.unet.input("sample").get_shape();
        latent_shape[0] = 1;
        ov::Tensor latent_vector_1d_(ov::element::f32, latent_shape, latent_vector_1d.data());

        ov::Tensor sample = diffusion_function(models.unet, seed_vec[n], steps, height, width, latent_vector_1d_, text_embeddings_);
        auto output_decoder = vae_decoder_function(models.vae_decoder, sample, height, width);

        std::vector<uint8_t> output_decoder_int = std::vector<uint8_t>(output_decoder.begin(), output_decoder.end());

        convertBGRtoRGB(output_decoder_int, width, height);
        imwrite(output_images[n], output_decoder_int.data(), height, width);
    }
}
