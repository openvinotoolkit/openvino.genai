// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief a header file for SD pipeline
 * @file stable_diffusion.hpp
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include <openvino_extensions/strings.hpp>

#include "lora_cpp.hpp"

#include "progress_bar.hpp"
#include "imwrite.hpp"

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

void convert_bgr2rgb(ov::Tensor image) {
    auto shape = image.get_shape();
    OPENVINO_ASSERT(image.get_element_type() == ov::element::u8 &&
        shape.size() == 4 && shape[0] == 1 && shape[3] == 3,
        "Image of u8 type and [1, H, W, 3] shape is expected");

    auto image_data = image.data<std::uint8_t>();
    for (size_t i = 0, spatial_dims = shape[1] * shape[2]; i < spatial_dims; ++i) {
        // Swap the red and blue components (BGR to RGB)
        std::swap(image_data[i * 3], image_data[i * 3 + 2]);
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

ov::Tensor vae_decoder_function(ov::CompiledModel& decoder_compiled_model, ov::Tensor sample) {
    const float coeffs_const{1 / 0.18215};
    for (size_t i = 0; i < sample.get_size(); ++i)
        sample.data<float>()[i] *= coeffs_const;

    ov::InferRequest req = decoder_compiled_model.create_infer_request();
    req.set_input_tensor(sample);
    req.infer();

    ov::Tensor decoded_image = req.get_output_tensor();
    ov::Tensor generated_image(ov::element::u8, decoded_image.get_shape());

    // convert to u8 image
    const float* decoded_data = decoded_image.data<const float>();
    std::uint8_t* generated_data = generated_image.data<std::uint8_t>();
    for (size_t i = 0; i < decoded_image.get_size(); ++i) {
        generated_data[i] = static_cast<std::uint8_t>(std::clamp(decoded_data[i] * 0.5f + 0.5f, 0.0f, 1.0f) * 255);
    }

    return generated_image;
}

ov::Tensor unet_infer_function(ov::CompiledModel unet_model,
                               ov::Tensor timestep,
                               ov::Tensor latent_input_1d,
                               ov::Tensor text_embedding_1d) {
    ov::InferRequest unet_infer_request = unet_model.create_infer_request();

    unet_infer_request.set_tensor("sample", latent_input_1d);
    unet_infer_request.set_tensor("timestep", timestep);
    unet_infer_request.set_tensor("encoder_hidden_states", text_embedding_1d);

    unet_infer_request.infer();

    ov::Tensor noise_pred_tensor = unet_infer_request.get_output_tensor();
    ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
    noise_pred_shape[0] = 1;

    const float guidance_scale = 7.5f;
    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
    const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

    ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);
    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i)
        noisy_residual.data<float>()[i] = noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);

    return noisy_residual;
}

class Scheduler {
    // https://gist.github.com/lorenzoriano/5414671
    template <typename T>
    static std::vector<T> linspace(T a, T b, size_t N) {
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
    static Real trapezoidal(F f, Real a, Real b, Real tol = 1e-6, int max_refinements = 100) {
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

    enum class BetaSchedule {
        LINEAR = 0,
        SCALED_LINEAR,
    };

    enum class PredictionType {
        EPSILON = 0,
        SAMPLE, 
        V_PREDICTION
    };

    static float lms_derivative_function(float tau, int32_t order, int32_t curr_order, const std::vector<float>& sigma_vec, int32_t t) {
        float prod = 1.0;

        for (int32_t k = 0; k < order; k++) {
            if (curr_order == k) {
                continue;
            }
            prod *= (tau - sigma_vec[t - k]) / (sigma_vec[t - curr_order] - sigma_vec[t - k]);
        }
        return prod;
    }

    // https://github.com/huggingface/diffusers/blob/v0.24.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L130
    void init(int32_t num_train_timesteps = 1000,
              float beta_start = 0.00085f,
              float beta_end = 0.012f,
              BetaSchedule beta_schedule = BetaSchedule::SCALED_LINEAR,
              PredictionType prediction_type = PredictionType::EPSILON,
              const std::vector<float>& trained_betas = {}) {
        std::vector<float> alphas, betas;

        if (!trained_betas.empty()) {
            betas = trained_betas;
        } else if (beta_schedule == BetaSchedule::LINEAR) {
            for (int32_t i = 0; i < num_train_timesteps; i++) {
                betas.push_back(beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1));
            }
        } else if (beta_schedule == BetaSchedule::SCALED_LINEAR) {
            float start = std::sqrt(beta_start);
            float end = std::sqrt(beta_end);
            std::vector<float> temp = linspace(start, end, num_train_timesteps);
            for (float b : temp) {
                betas.push_back(b * b);
            }
        } else {
            OPENVINO_THROW("beta_schedule must be one of 'linear' or 'scaled_linear'");
        }

        for (float b : betas) {
            alphas.push_back(1.0f - b);
        }

        std::vector<float> log_sigma_vec;
        for (size_t i = 1; i <= alphas.size(); i++) {
            float alphas_cumprod =
                std::accumulate(alphas.begin(), alphas.begin() + i, 1.0f, std::multiplies<float>{});
            float sigma = std::sqrt((1 - alphas_cumprod) / alphas_cumprod);
            m_sigmas.push_back(sigma);
            m_log_sigmas.push_back(std::log(sigma));
        }
    }

    int64_t _sigma_to_t(float sigma) const {
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

    std::vector<float> m_log_sigmas;
    std::vector<float> m_sigmas;
    std::vector<int64_t> m_timesteps;

    std::vector<std::vector<float>> m_derivative_list;

public:
    Scheduler() {
        init();
    }

    // https://github.com/huggingface/diffusers/blob/v0.24.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L173C9-L173C25
    float get_init_noise_sigma() const {
        // 14.6146
        return m_sigmas[0];
    }

    // https://github.com/huggingface/diffusers/blob/v0.24.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L187
    void scale_model_input(ov::Tensor sample, size_t inference_step) {
        const double scale = 1.0 / std::sqrt((m_sigmas[inference_step] * m_sigmas[inference_step] + 1));
        float* sample_data = sample.data<float>();
        for (size_t i = 0; i < sample.get_size(); i++) {
            sample_data[i] *= scale;
        }
    }

    // https://github.com/huggingface/diffusers/blob/v0.24.0/src/diffusers/schedulers/scheduling_lms_discrete.py#L271
    void set_timesteps(size_t num_inference_steps) {
        float delta = -999.0f / (num_inference_steps - 1);
        // transform interpolation to time range
        for (size_t i = 0; i < num_inference_steps; i++) {
            float t = 999.0 + i * delta;
            int32_t low_idx = std::floor(t);
            int32_t high_idx = std::ceil(t);
            float w = t - low_idx;
            m_sigmas[i] = std::exp((1 - w) * m_log_sigmas[low_idx] + w * m_log_sigmas[high_idx]);
        }
        m_sigmas.push_back(0.f);

        // initialize timesteps
        for (size_t i = 0; i < num_inference_steps; ++i) {
            int64_t timestep = _sigma_to_t(m_sigmas[i]);
            m_timesteps.push_back(timestep);
        }
    }

    ov::Tensor diffusion_function(ov::CompiledModel unet_compiled_model,
                                  size_t num_inference_steps,
                                  ov::Tensor latent_vector_1d,
                                  ov::Tensor text_embeddings) {
        // latents are multiplied by sigmas
        ov::Shape latent_shape = latent_vector_1d.get_shape(), latent_model_input_shape = latent_shape;
        latent_model_input_shape[0] = 2; // Unet accepts batch 2
        const ov::element::Type latent_type = latent_vector_1d.get_element_type();
        ov::Tensor latent_vector_1d_new(latent_type, latent_shape), latent_model_input(latent_type, latent_model_input_shape);
        for (size_t i = 0; i < latent_vector_1d.get_size(); ++i) {
            latent_vector_1d_new.data<float>()[i]  = latent_vector_1d.data<float>()[i] * get_init_noise_sigma();
        }

        ProgressBar bar(num_inference_steps);

        for (size_t inference_step = 0; inference_step < num_inference_steps; inference_step++) {
            bar.progress(inference_step);

            // 'sample'
            float* latent_model_input_data = latent_model_input.data<float>();
            const float* sample_data = latent_vector_1d_new.data<const float>();
            for (size_t i = 0, lanent_size = ov::shape_size(latent_shape); i < lanent_size; ++i)
                latent_model_input_data[i] = latent_model_input_data[i + lanent_size] = sample_data[i];

            scale_model_input(latent_model_input, inference_step);
            // 'timestep'
            ov::Tensor timestep(ov::element::i64, {1}, &m_timesteps[inference_step]);

            ov::Tensor noisy_residual = unet_infer_function(unet_compiled_model, timestep, latent_model_input, text_embeddings);

            // LMS step function:
            std::vector<float> derivative;
            for (size_t j = 0; j < latent_vector_1d_new.get_size(); j++) {
                // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise default "epsilon"
                float pred_latent = latent_vector_1d_new.data<float>()[j] - m_sigmas[inference_step] * noisy_residual.data<float>()[j];
                // 2. Convert to an ODE derivative
                derivative.push_back((latent_vector_1d_new.data<float>()[j] - pred_latent) / m_sigmas[inference_step]);
            }

            m_derivative_list.push_back(derivative);
            // keep the list size within 4
            size_t order = 4;
            if (order < m_derivative_list.size()) {
                m_derivative_list.erase(m_derivative_list.begin());
            }

            // 3. Compute linear multistep coefficients
            order = std::min(inference_step + 1, order);

            std::vector<float> lms_coeffs(order);
            for (size_t curr_order = 0; curr_order < order; curr_order++) {
                auto lms_derivative_functor = [order, curr_order, sigmas = this->m_sigmas, inference_step](float tau) {
                    return lms_derivative_function(tau, order, curr_order, sigmas, inference_step);
                };
                lms_coeffs[curr_order] = trapezoidal(lms_derivative_functor, static_cast<double>(m_sigmas[inference_step]), static_cast<double>(m_sigmas[inference_step + 1]), 1e-4);
            }

            // 4. Compute previous sample based on the derivatives path
            // prev_sample = sample + sum(coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives)))
            float * new_sample_data = latent_vector_1d_new.data<float>();
            for (size_t i = 0; i < latent_vector_1d_new.get_size(); ++i) {
                // derivative * coeffs
                float derivative_sum = 0.0f;
                for (size_t curr_order = 0; curr_order < order; curr_order++) {
                    derivative_sum += m_derivative_list[curr_order][i] * lms_coeffs[order - curr_order - 1];
                }
                new_sample_data[i] += derivative_sum;
            }
        }
        bar.finish();
        return latent_vector_1d_new;
    }
};

std::vector<float> text_encoder_infer_function(StableDiffusionModels models, const std::string& prompt) {
    const size_t MAX_LENGTH = 77 /* 'model_max_length' from 'tokenizer_config.json' */, BATCH_SIZE = 1;
    const int32_t EOS_TOKEN_ID = 49407, PAD_TOKEN_ID = EOS_TOKEN_ID;
    const ov::Shape input_ids_shape({BATCH_SIZE, MAX_LENGTH});

    // Tokenization

    ov::InferRequest tokenizer_req = models.tokenizer.create_infer_request();
    ov::Tensor input_ids_tensor = tokenizer_req.get_tensor("input_ids");

    input_ids_tensor.set_shape(input_ids_shape);
    // need to pre-fill 'input_ids_tensor' with 'PAD' tokens
    std::fill_n(input_ids_tensor.data<int32_t>(), ov::shape_size(input_ids_shape), PAD_TOKEN_ID);

    ov::Tensor packed_strings = tokenizer_req.get_input_tensor();
    openvino_extensions::pack_strings(std::array<std::string, BATCH_SIZE>{prompt}, packed_strings);

    tokenizer_req.infer();
    // restore shape to CLIP's expected input shape if actual shape is smaller than 'input_ids_shape'
    input_ids_tensor.set_shape(input_ids_shape);

    // Text embedding

    std::vector<float> text_embeddings;
    text_embeddings.reserve(ov::shape_size(input_ids_shape));

    ov::InferRequest text_encoder_req = models.text_encoder.create_infer_request();
    text_encoder_req.set_input_tensor(input_ids_tensor);
    text_encoder_req.infer();

    ov::Tensor text_embeddings_tensor = text_encoder_req.get_output_tensor(0);
    std::cout << "text_embeddings_tensor shape " << text_embeddings_tensor.get_shape() << std::endl;
    for (size_t i = 0; i < text_embeddings_tensor.get_size(); i++)
        text_embeddings.push_back(text_embeddings_tensor.data<const float>()[i]);

    return text_embeddings;
}

StableDiffusionModels compile_models(const std::string& model_path,
                                     const std::string& device,
                                     const std::string& type,
                                     const std::string& lora_path,
                                     const float alpha,
                                     const bool use_cache) {
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

    // Tokenizer
    std::string tokenizer_model_path = "../models/tokenizer/tokenizer_encoder.xml";
    models.tokenizer = core.compile_model(tokenizer_model_path, device);

    return models;
}

void stable_diffusion(const std::string& positive_prompt = std::string{},
                      const std::vector<std::string>& output_images = {},
                      const std::string& device = std::string{},
                      const size_t inference_steps = 20,
                      const std::vector<uint32_t>& seed_vec = {},
                      const uint32_t num_images = 1,
                      const uint32_t height = 512,
                      const uint32_t width = 512,
                      const std::string negative_prompt = std::string{},
                      const bool use_cache = false,
                      const std::string& model_base_path = std::string{},
                      const std::string& model_type = std::string{},
                      const std::string& lora_path = std::string{},
                      const float alpha = 0.75f,
                      const bool read_np_latent = false) {
    StableDiffusionModels models = compile_models(model_base_path, device, model_type, lora_path, alpha, use_cache);

    ov::PartialShape sample_shape = models.unet.input("sample").get_partial_shape();
    ov::PartialShape text_embedding_partial_shape = models.unet.input("encoder_hidden_states").get_partial_shape();
    OPENVINO_ASSERT(sample_shape.is_dynamic() || (sample_shape[2] * 8 == width && sample_shape[3] * 8 == height),
        "UNet model has static shapes [1, 4, H/8, W/8] or dynamic shapes [?, 4, ?, ?]");

    ov::Shape latent_vector_shape{1, 4, height / 8, width / 8};
    ov::Shape text_embedding_shape{2, // batch 2, because we need to combine 'negative' and 'positive' prompt
        static_cast<size_t>(text_embedding_partial_shape[1].get_length()),
        static_cast<size_t>(text_embedding_partial_shape[2].get_length())};

    std::vector<float> text_embeddings_pos = text_encoder_infer_function(models, positive_prompt);
    std::vector<float> text_embeddings = text_encoder_infer_function(models, negative_prompt);
    std::copy_n(text_embeddings_pos.begin(), text_embeddings_pos.size(), std::back_inserter(text_embeddings));
    std::copy_n(text_embeddings_pos.begin(), text_embeddings_pos.size(), std::back_inserter(text_embeddings));
    ov::Tensor text_embeddings_(ov::element::f32, text_embedding_shape, text_embeddings.data());

    for (uint32_t n = 0; n < num_images; n++) {
        // TODO: check that 'np_randn_function' can generate enough data
        std::vector<float> latent_vector_1d = read_np_latent ? np_randn_function() : std_randn_function(seed_vec[n], height, width);
        ov::Tensor latent_vector_1d_(ov::element::f32, latent_vector_shape, latent_vector_1d.data());

        Scheduler scheduler;
        scheduler.set_timesteps(inference_steps);
        ov::Tensor sample = scheduler.diffusion_function(models.unet, inference_steps, latent_vector_1d_, text_embeddings_);
        ov::Tensor generated_image = vae_decoder_function(models.vae_decoder, sample);

        convert_bgr2rgb(generated_image);
        imwrite(output_images[n], generated_image);
    }
}
