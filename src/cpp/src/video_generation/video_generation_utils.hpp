// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>

#include <openvino/op/transpose.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

#include "image_generation/image_processor.hpp"
#include "image_generation/schedulers/ischeduler.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

namespace ov::genai::video_generation_utils {

inline std::shared_ptr<IScheduler> cast_scheduler(std::shared_ptr<Scheduler>&& scheduler) {
    auto casted = std::dynamic_pointer_cast<IScheduler>(std::move(scheduler));
    OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
    return casted;
}

// Rescales the CFG noise prediction to fix overexposure when using zero terminal SNR
// noise_cfg and noise_pred_text each contain batch_size * elements_per_sample consecutive floats.
inline void rescale_noise_cfg(float* noise_cfg,
                              const float* noise_pred_text,
                              size_t batch_size,
                              size_t elements_per_sample,
                              float guidance_rescale) {
    if (elements_per_sample == 0) {
        return;
    }
    for (size_t b = 0; b < batch_size; ++b) {
        float* cfg_sample = noise_cfg + b * elements_per_sample;
        const float* text_sample = noise_pred_text + b * elements_per_sample;

        double text_mean = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            text_mean += text_sample[i];
        }
        text_mean /= static_cast<double>(elements_per_sample);

        double text_var = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const double diff = text_sample[i] - text_mean;
            text_var += diff * diff;
        }
        const float std_text = static_cast<float>(std::sqrt(text_var / static_cast<double>(elements_per_sample)));

        double cfg_mean = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            cfg_mean += cfg_sample[i];
        }
        cfg_mean /= static_cast<double>(elements_per_sample);

        double cfg_var = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const double diff = cfg_sample[i] - cfg_mean;
            cfg_var += diff * diff;
        }
        const float std_cfg = static_cast<float>(std::sqrt(cfg_var / static_cast<double>(elements_per_sample)));

        const float scale = std_cfg > 0.0f ? std_text / std_cfg : 1.0f;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const float rescaled = cfg_sample[i] * scale;
            cfg_sample[i] = guidance_rescale * rescaled + (1.0f - guidance_rescale) * cfg_sample[i];
        }
    }
}

// Unpacked latents of shape [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p,
// p]. The patch dimensions are then permuted and collapsed into the channel dimension of shape: [B, F // p_t * H // p *
// W // p, C * p_t * p * p] (a 3 dimensional tensor). dim=0 is the batch size, dim=1 is the effective video sequence
// length, dim=2 is the effective number of input features
inline ov::Tensor pack_latents(ov::Tensor& latents, size_t patch_size, size_t patch_size_t) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape.at(0), num_channels = latents_shape.at(1), num_frames = latents_shape.at(2),
           height = latents_shape.at(3), width = latents_shape.at(4);
    size_t post_patch_num_frames = num_frames / patch_size_t;
    size_t post_patch_height = height / patch_size;
    size_t post_patch_width = width / patch_size;
    latents.set_shape({batch_size,
                       num_channels,
                       post_patch_num_frames,
                       patch_size_t,
                       post_patch_height,
                       patch_size,
                       post_patch_width,
                       patch_size});
    std::array<int64_t, 8> order = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(ov::element::f32, {})};
    ov::op::v1::Transpose{}.evaluate(outputs,
                                     {latents, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, order.data())});
    ov::Shape permuted_shape = outputs.at(0).get_shape();
    outputs.at(0).set_shape({permuted_shape.at(0),
                             permuted_shape.at(1) * permuted_shape.at(2) * permuted_shape.at(3),
                             permuted_shape.at(4) * permuted_shape.at(5) * permuted_shape.at(6)});
    return outputs.at(0);
}

// Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
// are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of what happens
// in the `_pack_latents` method.
inline ov::Tensor unpack_latents(const ov::Tensor& latents,
                                 size_t num_frames,
                                 size_t height,
                                 size_t width,
                                 size_t patch_size = 1,
                                 size_t patch_size_t = 1) {
    const ov::Shape in_shape = latents.get_shape();
    OPENVINO_ASSERT(in_shape.size() == 3, "unpack_latents expects [B, S, D] input shape");
    const size_t batch_size = in_shape.at(0), sequence_length = in_shape.at(1), feature_dimensions = in_shape.at(2);

    const size_t patch_volume = patch_size_t * patch_size * patch_size;
    OPENVINO_ASSERT(feature_dimensions % patch_volume == 0, "D must be divisible by patch_size_t * patch_size * patch_size");
    const size_t num_channels = feature_dimensions / patch_volume;

    // Zero-copy view: the Transpose below reads the input and writes to a separate output tensor
    ov::Tensor reshaped(latents.get_element_type(),
                        {batch_size, num_frames, height, width, num_channels, patch_size_t, patch_size, patch_size},
                        latents.data());

    // permute(0, 4, 1, 5, 2, 6, 3, 7) -> [B, C, F//patch_size_t, patch_size_t, H//patch_size, patch_size, W//patch_size, patch_size]
    const std::array<int64_t, 8> order = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(reshaped.get_element_type(), {})};
    ov::op::v1::Transpose{}.evaluate(
        outputs,
        {reshaped, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, const_cast<int64_t*>(order.data()))}
    );

    // (F//patch_size_t, patch_size_t) -> F, (H//patch_size, patch_size) -> H, (W//patch_size, patch_size) -> W
    const ov::Shape perm = outputs[0].get_shape(); // [B, C, F//patch_size_t, patch_size_t, H//patch_size, patch_size, W//patch_size, patch_size]
    OPENVINO_ASSERT(perm.size() == 8, "Unexpected rank after transpose");

    const size_t F = perm[2] * perm[3]; // (F//patch_size_t) * patch_size_t
    const size_t H = perm[4] * perm[5]; // (H//patch_size) * patch_size
    const size_t W = perm[6] * perm[7]; // (W//patch_size) * patch_size

    outputs[0].set_shape({perm[0], perm[1], F, H, W}); // [B, C, F, H, W]
    return outputs[0];
}

inline void reshape_to_1C111(ov::Tensor& t, size_t C) {
    size_t elems = 1;
    for (auto d : t.get_shape())
        elems *= d;

    OPENVINO_ASSERT(elems == C, "latents_mean/std must contain exactly C elements (got ", elems, ", expected ", C, ")");

    t.set_shape({1, C, 1, 1, 1});
}

inline void check_video_size(int64_t height, int64_t width, int64_t divisor) {
    OPENVINO_ASSERT(height > 0, "Height must be positive");
    OPENVINO_ASSERT(height % divisor == 0, "Height have to be divisible by ", divisor, " but got ", height);
    OPENVINO_ASSERT(width > 0, "Width must be positive");
    OPENVINO_ASSERT(width % divisor == 0, "Width have to be divisible by ", divisor, " but got ", width);
}

inline ov::Tensor make_i64_scalar(int64_t value) {
    ov::Tensor scalar(ov::element::i64, {});
    *scalar.data<int64_t>() = value;
    return scalar;
}

inline ov::Tensor make_scalar(const ov::element::Type& et, float v) {
    ov::Tensor s(et, {});
    if (et == ov::element::f32) {
        *s.data<float>() = v;
    } else if (et == ov::element::f16) {
        *s.data<ov::float16>() = static_cast<ov::float16>(v);
    } else if (et == ov::element::bf16) {
        *s.data<ov::bfloat16>() = static_cast<ov::bfloat16>(v);
    } else {
        OPENVINO_ASSERT(false, "Unsupported element type for scalar scaling_factor");
    }
    return s;
}

// Denormalize latents across channel dim: [B, C, F, H, W]
// latents = latents * latents_std / scaling_factor + latents_mean
inline ov::Tensor denormalize_latents(const ov::Tensor& latents,
                                      ov::Tensor latents_mean,
                                      ov::Tensor latents_std,
                                      float scaling_factor = 1.0f) {
    const ov::Shape latents_shape = latents.get_shape();
    OPENVINO_ASSERT(latents_shape.size() == 5, "denormalize_latents expects [B, C, F, H, W]");
    const size_t num_channels = latents_shape[1];

    // .view(1, -1, 1, 1, 1)
    reshape_to_1C111(latents_mean, num_channels);
    reshape_to_1C111(latents_std, num_channels);

    const auto latents_type = latents.get_element_type();
    ov::Tensor scale = make_scalar(latents_type, scaling_factor);

    // latents * latents_std
    std::vector<ov::Tensor> tmp{ov::Tensor(latents_type, {})};
    ov::op::v1::Multiply{}.evaluate(tmp, {latents, latents_std});  // NUMPY broadcast

    // (...) / scaling_factor
    std::vector<ov::Tensor> tmp2{ov::Tensor(latents_type, {})};
    ov::op::v1::Divide{}.evaluate(tmp2, {tmp[0], scale});

    // (...) + latents_mean
    std::vector<ov::Tensor> result{ov::Tensor(latents_type, {})};
    ov::op::v1::Add{}.evaluate(result, {tmp2[0], latents_mean});

    return result[0];  // [B, C, F, H, W]
}

// (latents - latents_mean) * scaling_factor / latents_std, the inverse of denormalize_latents
inline ov::Tensor normalize_latents(const ov::Tensor& latents,
                                    ov::Tensor latents_mean,
                                    ov::Tensor latents_std,
                                    float scaling_factor = 1.0f) {
    const ov::Shape latents_shape = latents.get_shape();
    OPENVINO_ASSERT(latents_shape.size() == 5, "normalize_latents expects [B, C, F, H, W]");
    const size_t num_channels = latents_shape[1];

    reshape_to_1C111(latents_mean, num_channels);
    reshape_to_1C111(latents_std, num_channels);

    const auto latents_type = latents.get_element_type();
    ov::Tensor scale = make_scalar(latents_type, scaling_factor);

    std::vector<ov::Tensor> tmp{ov::Tensor(latents_type, {})};
    ov::op::v1::Subtract{}.evaluate(tmp, {latents, latents_mean});

    std::vector<ov::Tensor> tmp2{ov::Tensor(latents_type, {})};
    ov::op::v1::Multiply{}.evaluate(tmp2, {tmp[0], scale});

    std::vector<ov::Tensor> result{ov::Tensor(latents_type, {})};
    ov::op::v1::Divide{}.evaluate(result, {tmp2[0], latents_std});

    return result[0];  // [B, C, F, H, W]
}

// Splits VAE encoder 'latent_parameters' [B, 2C, ...] into mean and logvar halves
class DiagonalGaussianDistribution {
public:
    explicit DiagonalGaussianDistribution(ov::Tensor parameters) : m_params(std::move(parameters)) {
        OPENVINO_ASSERT(m_params.get_element_type() == ov::element::f32,
            "DiagonalGaussianDistribution requires f32 encoder output, got ",
            m_params.get_element_type());
        const ov::Shape& full_shape = m_params.get_shape();
        OPENVINO_ASSERT(full_shape.size() >= 2, "Parameters tensor rank must be at least 2");
        OPENVINO_ASSERT(full_shape[1] % 2 == 0, "Channel dimension must be even to split mean and logvar");
        m_channels = full_shape[1] / 2;
        m_spatial = 1;
        for (size_t i = 2; i < full_shape.size(); ++i)
            m_spatial *= full_shape[i];

        ov::Shape std_shape = full_shape;
        std_shape[1] = m_channels;
        m_std = ov::Tensor(m_params.get_element_type(), std_shape);

        const float* src = m_params.data<float>();
        float* std_data = m_std.data<float>();
        const size_t batch = full_shape[0];
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < m_channels; ++c) {
                const size_t lvar_off = (b * full_shape[1] + m_channels + c) * m_spatial;
                const size_t dst_off  = (b * m_channels + c) * m_spatial;
                for (size_t s = 0; s < m_spatial; ++s) {
                    const float logvar = std::min(std::max(src[lvar_off + s], -30.0f), 20.0f);
                    std_data[dst_off + s] = std::exp(0.5f * logvar);
                }
            }
        }
    }

    ov::Tensor sample(std::shared_ptr<Generator> generator) const {
        OPENVINO_ASSERT(generator, "Generator must not be nullptr");

        ov::Shape sample_shape = m_params.get_shape();
        sample_shape[1] = m_channels;
        ov::Tensor result = generator->randn_tensor(sample_shape);
        OPENVINO_ASSERT(result.get_element_type() == ov::element::f32,
            "Generator::randn_tensor() must return an f32 tensor, got ",
            result.get_element_type());

        const float* params_data = m_params.data<float>();
        const float* std_data = m_std.data<float>();
        float* result_data = result.data<float>();
        const size_t batch = m_params.get_shape()[0];
        const size_t full_channels = m_params.get_shape()[1];

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < m_channels; ++c) {
                const size_t mean_off = (b * full_channels + c) * m_spatial;
                const size_t dst_off  = (b * m_channels + c) * m_spatial;
                for (size_t s = 0; s < m_spatial; ++s) {
                    result_data[dst_off + s] = params_data[mean_off + s] + std_data[dst_off + s] * result_data[dst_off + s];
                }
            }
        }

        return result;
    }

    // The distribution mean (diffusers' sample_mode="argmax")
    ov::Tensor mode() const {
        ov::Shape mean_shape = m_params.get_shape();
        const size_t full_channels = mean_shape[1];
        mean_shape[1] = m_channels;
        ov::Tensor result(m_params.get_element_type(), mean_shape);

        const float* params_data = m_params.data<float>();
        float* result_data = result.data<float>();
        const size_t mean_elems = m_channels * m_spatial;
        for (size_t b = 0; b < mean_shape[0]; ++b) {
            std::memcpy(result_data + b * mean_elems, params_data + b * full_channels * m_spatial, mean_elems * sizeof(float));
        }
        return result;
    }

private:
    ov::Tensor m_params, m_std;
    size_t m_channels, m_spatial;
};

// Validates a u8 NHWC conditioning image, resizes it to height x width and returns the normalized
// f32 [N, 3, 1, H, W] single-frame VAE encoder input
inline ov::Tensor image_to_encoder_input(const ov::Tensor& image,
                                         int64_t height,
                                         int64_t width,
                                         ImageResizer& resizer,
                                         ImageProcessor& processor) {
    // ov::Tensor copies share the underlying memory, so set_shape() here would promote the
    // caller's rank-3 tensor in place. Wrap the same memory in a rank-4 view instead.
    ov::Tensor img = image;
    if (image.get_shape().size() == 3) {
        const auto s = image.get_shape();
        img = ov::Tensor(image.get_element_type(), ov::Shape{1, s[0], s[1], s[2]}, image.data());
    }

    const auto& img_shape = img.get_shape();
    OPENVINO_ASSERT(img_shape.size() == 4,
                    "Conditioning image must have shape [H, W, 3] or [1, H, W, 3] (NHWC), got rank ",
                    img_shape.size());
    OPENVINO_ASSERT(img.get_element_type() == ov::element::u8,
                    "Conditioning image must have element type u8 (uint8), got ",
                    img.get_element_type());
    OPENVINO_ASSERT(img_shape[3] == 3,
                    "Conditioning image must have 3 channels in the last dimension (NHWC), got ",
                    img_shape[3]);

    ov::Tensor resized = (img_shape[1] == static_cast<size_t>(height) && img_shape[2] == static_cast<size_t>(width))
                             ? img
                             : resizer.execute(img, height, width);
    ov::Tensor processed = processor.execute(resized);

    OPENVINO_ASSERT(processed.get_element_type() == ov::element::f32,
                    "ImageProcessor must return f32, got ", processed.get_element_type());
    OPENVINO_ASSERT(processed.get_shape().size() == 4,
                    "ImageProcessor must return rank-4 [N,C,H,W], got rank ", processed.get_shape().size());
    const auto& proc_shape = processed.get_shape();
    ov::Tensor encoder_input(ov::element::f32, {proc_shape[0], proc_shape[1], 1, proc_shape[2], proc_shape[3]});
    std::memcpy(encoder_input.data<float>(), processed.data<const float>(), processed.get_byte_size());
    return encoder_input;
}

// Converts between the numeric element types the exported IRs use for masks (e.g. f32/i64/i32)
inline ov::Tensor convert_tensor(const ov::Tensor& tensor, const ov::element::Type& target_type) {
    if (tensor.get_element_type() == target_type) {
        return tensor;
    }
    std::vector<ov::Tensor> converted{ov::Tensor(target_type, tensor.get_shape())};
    ov::op::v0::Convert{}.evaluate(converted, {tensor});
    return converted[0];
}

inline ov::Tensor tensor_from_vector(const std::vector<float>& data) {
    ov::Tensor t{ov::element::f32, ov::Shape{data.size()}};
    if (!data.empty()) {
        std::memcpy(t.data<float>(), data.data(), data.size() * sizeof(float));
    }
    return t;
}

}  // namespace ov::genai::video_generation_utils
